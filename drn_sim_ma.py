import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import marimo
    import visualize
    import matplotlib.pyplot as plt

    class DroneSimulator:
        def __init__(
            self,
            mass=0.027,
            inertia=np.diag([0.000014, 0.000014, 0.000022]),
            g=9.81,
            dt=0.01,
            ctrl=None,
        ):
            self.m = mass
            self.I = inertia
            self.invI = np.linalg.inv(self.I)
            self.g = g
            self.dt = dt
            # state: [px,py,pz, vx,vy,vz, φ,θ,ψ, ωx,ωy,ωz]
            self.state = np.zeros(12)
            self.ctrl = ctrl

            # PD gains for stable cascaded control
            # altitude
            self.kpz = 2.0
            self.kdz = 1.0
            # horizontal
            self.kpx = 1.0
            self.kdx = 0.5
            self.kpy = 1.5  # increased for better y stability
            self.kdy = 1.0  # increased damping
            # attitude
            self.kpr = 12.0  # increased damping
            self.kdr = 8.0
            self.kpp = 12.0
            self.kdp = 8.0
            self.kpyaw = 6.0
            self.kdyaw = 3.0

        def reset(self, state=None):
            """Optionally set initial state vector of length 12."""
            self.ctrl.reset()
            if state is not None:
                self.state = np.array(state, dtype=float)
            else:
                self.state.fill(0.0)

        def rotation_matrix(self, angles):
            phi, theta, psy = angles
            cphi = np.cos(phi)
            sphi = np.sin(phi)
            cthe = np.cos(theta)
            sthe = np.sin(theta)
            cpsy = np.cos(psy)
            spsy = np.sin(psy)
            # Rb2i
            R = np.array(
                [
                    [
                        cthe * cpsy,
                        sphi * sthe * cpsy - cphi * spsy,
                        cphi * sthe * cpsy + sphi * spsy,
                    ],
                    [
                        cthe * spsy,
                        sphi * sthe * spsy + cphi * cpsy,
                        cphi * sthe * spsy - sphi * cpsy,
                    ],
                    [-sthe, sphi * cthe, cphi * cthe],
                ]
            )
            return R

        def euler_rates(self, angles, omega):
            phi, theta, _ = angles
            cphi = np.cos(phi)
            sphi = np.sin(phi)
            tthe = np.tan(theta)
            secthe = 1 / np.cos(theta)
            # mapping body ω -> euler angle rates
            T = np.array(
                [
                    [1, sphi * tthe, cphi * tthe],
                    [0, cphi, -sphi],
                    [0, sphi * secthe, cphi * secthe],
                ]
            )
            return T.dot(omega)

        def step(self, thrust, torque):
            """
            thrust: scalar force along body z-axis
            torque: np.array([τx, τy, τz])
            """
            s = self.state
            # unpack
            p = s[0:3]
            v = s[3:6]
            angles = s[6:9]
            omega = s[9:12]

            # translational dynamics
            R = self.rotation_matrix(angles)
            F_body = np.array([0.0, 0.0, thrust])
            acc = R.dot(F_body) / self.m - np.array([0, 0, self.g])

            # rotational dynamics
            omega_dot = self.invI.dot(torque - np.cross(omega, self.I.dot(omega)))

            # kinematics
            angles_dot = self.euler_rates(angles, omega)

            # Euler integration
            p = p + v * self.dt
            v = v + acc * self.dt
            angles = self.angle_wrap(angles + angles_dot * self.dt)
            omega = omega + omega_dot * self.dt

            self.state = np.hstack([p, v, angles, omega])
            return self.state

        @staticmethod
        def angle_wrap(angle):
            """Wrap angle to [-pi, pi]."""
            return (angle + np.pi) % (2 * np.pi) - np.pi


    class PID:
        def __init__(
            self,
            kp: float,
            ki: float,
            kd: float,
            dt: float,
            windup: float = None,
            beta: float = 1.0,
        ):
            self.kp = kp
            self.ki = ki
            self.kd = kd
            self.dt = dt
            self.windup = windup
            # derivative filter coefficient (0=noisy, 1=no filtering)
            self.beta = beta
            self.integral = 0.0
            self.prev_error = 0.0
            self.prev_deriv = 0.0

        def reset(self) -> None:
            self.integral = 0.0
            self.prev_error = 0.0

        def __call__(self, sp: float, pv: float) -> float:
            """
            sp: setpoint, pv: process variable
            returns control action
            """
            err = sp - pv
            self.integral += err * self.dt
            if self.windup is not None:
                self.integral = np.clip(self.integral, -self.windup, self.windup)
            # raw derivative
            deriv_raw = (err - self.prev_error) / self.dt
            # filtered derivative
            deriv = self.beta * self.prev_deriv + (1 - self.beta) * deriv_raw
            self.prev_error = err
            self.prev_deriv = deriv
            return self.kp * err + self.ki * self.integral + self.kd * deriv

    class CascadePIDController:
        def __init__(self, pid_alt, pid_att, pid_xy, dt: float = 0.01, mass: float = 0.027, gravity: float = 9.81):
            # outer‐loop
            self.alt_pid = PID(kp=pid_alt[0], ki=pid_alt[1], kd=pid_alt[2], dt=dt)
            # small xy→tilt mapping
            self.kpxy = 0.3
            # inner‐loop
            self.roll_pid = PID(kp=pid_att[0], ki=pid_att[1], kd=pid_att[2], dt=dt, windup=0.5)
            self.pitch_pid = PID(kp=pid_att[0], ki=pid_att[1], kd=pid_att[2], dt=dt, windup=0.5)
            self.yaw_pid = PID(kp=pid_att[0], ki=pid_att[1], kd=pid_att[2], dt=dt, windup=0.5)
            # physics parameters
            self.m = mass
            self.g = gravity
            # outer-loop horizontal PID (adds integral to fix steady-state y drift)
            # horizontal position controllers
            self.pid_x = PID(kp=pid_xy[0], ki=pid_xy[1], kd=pid_xy[2], dt=dt, windup=0.5, beta=0.3)
            self.pid_y = PID(kp=pid_xy[0], ki=pid_xy[1], kd=pid_xy[2], dt=dt, windup=0.5, beta=0.3)
            # altitude-loop PD

        def reset(self) -> None:
            self.alt_pid.reset()
            self.pid_x.reset()
            self.pid_y.reset()
            self.roll_pid.reset()
            self.pitch_pid.reset()
            self.yaw_pid.reset()

        # def compute(
        #     self,
        #     target_pos: np.ndarray,  # [x, y, z]
        #     target_yaw: float,
        #     state: np.ndarray,  # sim.state = [p(3), v(3), ang(3), ω(3)]
        # ) -> tuple[float, np.ndarray]:
        #     # unpack state
        #     pos = state[0:3]
        #     vel = state[3:6]
        #     ang = state[6:9]
        #     rates = state[9:12]

        #     # 1) Altitude PD -> thrust
        #     ez = target_pos[2] - pos[2]
        #     az = self.alt_pid.kp * ez - self.alt_pid.kd * vel[2]
        #     thrust = self.m * (self.g + az)
        #     thrust = np.clip(thrust, 0.0, 2.0 * self.m * self.g)

        #     # 2) Horizontal PID -> desired roll/pitch using arctan2 mapping
        #     ax_cmd = self.pid_x(target_pos[0], pos[0])
        #     ay_cmd = self.pid_y(target_pos[1], pos[1])
        #     des_roll = np.arctan2(ay_cmd, self.g)
        #     des_pitch = np.arctan2(-ax_cmd, self.g)
        #     # limit desired tilt to ±15° for better damping
        #     max_ang = np.radians(15)
        #     des_roll = np.clip(des_roll, -max_ang, max_ang)
        #     des_pitch = np.clip(des_pitch, -max_ang, max_ang)

        #     # 3) Attitude PID -> body torques using PID instances (smoothes angular rates)
        #     tau_x = self.roll_pid(des_roll, ang[0])
        #     tau_y = self.pitch_pid(des_pitch, ang[1])
        #     tau_z = self.yaw_pid(target_yaw, ang[2])
        #     # saturate torques
        #     max_tau = 0.03  # tighter torque limits
        #     tau_x = np.clip(tau_x, -max_tau, max_tau)
        #     tau_y = np.clip(tau_y, -max_tau, max_tau)
        #     tau_z = np.clip(tau_z, -max_tau, max_tau)
        #     torque = np.array([tau_x, tau_y, tau_z])

        #     return thrust, torque


        def compute(
            self,
            target_pos: np.ndarray,  # [x, y, z]
            target_yaw: float,
            state: np.ndarray,  # sim.state = [p(3), v(3), ang(3), ω(3)]
        ) -> tuple[float, np.ndarray]:
            """
            Cascaded PID control: position -> attitude -> thrust/torque
            Returns: (thrust, torque_vector)
            """
            # Unpack state
            pos = state[0:3]
            vel = state[3:6] 
            angles = DroneSimulator.angle_wrap(state[6:9])  # [roll, pitch, yaw]
            omega = state[9:12]

            # === OUTER LOOP: Position Control ===

            # Altitude control (generates thrust)
            thrust = self.alt_pid(target_pos[2], pos[2])
            # Add feedforward term for gravity compensation
            thrust += self.m * self.g

            # Horizontal position control (generates desired roll/pitch)
            x_error = self.pid_x(target_pos[0], pos[0])
            y_error = self.pid_y(target_pos[1], pos[1])

            # Map horizontal errors to desired attitude angles
            # Small angle approximation: desired_roll ≈ -y_accel/g, desired_pitch ≈ x_accel/g
            desired_roll = -self.kpxy * y_error  # negative for proper direction
            desired_pitch = self.kpxy * x_error
            desired_yaw = target_yaw

            # === INNER LOOP: Attitude Control ===

            # Attitude PID controllers generate torques
            torque_x = self.roll_pid(desired_roll, angles[0])
            torque_y = self.pitch_pid(desired_pitch, angles[1]) 
            torque_z = self.yaw_pid(desired_yaw, angles[2])
            max_tau = 0.03  # or even lower for initial tests
            torque_x = np.clip(torque_x, -max_tau, max_tau)
            torque_y = np.clip(torque_y, -max_tau, max_tau)
            torque_z = np.clip(torque_z, -max_tau, max_tau)
            torque = np.array([torque_x, torque_y, torque_z])


            torque = np.array([torque_x, torque_y, torque_z])

            return thrust, torque
    return CascadePIDController, DroneSimulator, marimo, np, plt, visualize


@app.cell
def _(sim):
    sim.ctrl.pid_x.kp
    return


@app.cell
def _(kd_alt, ki_alt, kp_alt):
    kp_alt, ki_alt, kd_alt
    return


@app.cell
def _(kd_xy, ki_xy, kp_xy):
    kp_xy, ki_xy, kd_xy
    return


@app.cell
def _(kd_att, ki_att, kp_att):
    kp_att, ki_att, kd_att
    return


@app.cell
def _(
    CascadePIDController,
    DroneSimulator,
    kd_alt,
    kd_att,
    kd_xy,
    ki_alt,
    ki_att,
    ki_xy,
    kp_alt,
    kp_att,
    kp_xy,
    np,
    visualize,
):
    viewer = visualize.States3DPlotter("Drone Simulation")
    sim = DroneSimulator(dt=0.01, ctrl=CascadePIDController( 
    10**np.array((kp_alt.value, ki_alt.value, kd_alt.value)), 
    10**np.array((kp_att.value, ki_att.value, kd_att.value)),
    10**np.array((kp_xy.value, ki_xy.value, kd_xy.value)), dt=0.01,
    ))

    target_pos = np.array([0.1, 0.3, 1.0])
    target_yaw = 0.5
    TMAX = 100*5
    states = np.zeros((sim.state.shape[0], TMAX))
    controls = np.zeros((4, TMAX))

    for i in range(TMAX):
        thrust, torque = sim.ctrl.compute(target_pos, target_yaw, sim.state)
        states[:, i] = sim.step(thrust, torque)
        controls[:,i] = [thrust, *torque]

    # print("final state:", sim.state)
    viewer.plot(np.arange(0, TMAX * sim.dt, sim.dt), states)
    viewer.figs[2]
    return TMAX, controls, sim, viewer


@app.cell
def _(TMAX, controls, np, plt, sim):
    plt.plot(np.arange(0, TMAX * sim.dt, sim.dt), controls[3,:])
    return


@app.cell
def _(marimo):
    kp_alt = marimo.ui.slider(-10, 10, 0.1, value=1.8, label="Altitude P(log)")
    ki_alt = marimo.ui.slider(-10, 10, 0.1, value=1.4, label="Altitude I(log)")
    kd_alt = marimo.ui.slider(-10, 10, 0.1, value=1.6, label="Altitude D(log)")
    kp_xy = marimo.ui.slider(-10, 10, 0.1, value=1.3, label="XY Pos P(log)")
    ki_xy = marimo.ui.slider(-10, 10, 0.1, value=0.6, label="XY Pos I(log)")
    kd_xy = marimo.ui.slider(-10, 10, 0.1, value=1.1, label="XY Pos D(log)")
    kp_att = marimo.ui.slider(-10, 10, 0.1, value=-5, label="Att P(log)")
    ki_att = marimo.ui.slider(-10, 10, 0.1, value=0, label="Att I(log)")
    kd_att = marimo.ui.slider(-10, 10, 0.1, value=1, label="Att D(log)")

    return kd_alt, kd_att, kd_xy, ki_alt, ki_att, ki_xy, kp_alt, kp_att, kp_xy


@app.cell
def _(viewer):
    viewer.figs[0]
    return


@app.cell
def _(viewer):
    viewer.figs[1]
    return


@app.cell
def _(viewer):
    viewer.figs[2]
    return


@app.cell
def _(viewer):
    viewer.figs[3]
    return


if __name__ == "__main__":
    app.run()
