# you won't have file history because this was part of a private repo, sorry ;)
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

matplotlib.rcParams["text.usetex"] = True       # latex is required for nice plots, if you don't want it, comment lines 9 and 10
matplotlib.rcParams["font.family"] = "serif"


class States3DPlotter:
    states: list[str] = [
        r"$p_x[m]$",
        r"$p_y[m]$",
        r"$p_z[m]$",
        r"$v_x[m/s]$",
        r"$v_y[m/s]$",
        r"$v_z[m/s]$",
        r"$\phi[rad]$",
        r"$\theta[rad]$",
        r"$\psi[rad]$",
        r"$w_x[rad/s]$",
        r"$w_y[rad/s]$",
        r"$w_z[rad/s]$",
    ]
    states_cat: list[str] = [
        r"Position",
        r"Velocity",
        r"Orientation",
        r"Angular Velocity",
    ]

    def __init__(self, name: str | None = None, rot_mat: bool = False) -> None:
        self.figs: list[plt.Figure] = [plt.figure(figsize=(18, 10)) for _ in range(4)]
        self.plots: list[plt.Axes] = []
        self.name = name
        if name is None:
            self.name = str(int(time.time() * 100) / 100)
        for index_fig, fig in enumerate(self.figs):
            fig.tight_layout()
            fig.suptitle(self.name)
            self.plots.append(fig.add_subplot(3, 1, 1))
            self.plots[-1].sharex(self.plots[0])
            self.plots[-1].set_ylabel(self.states[index_fig * 3])
            self.plots[-1].tick_params(
                axis="x", which="both", bottom=False, top=False, labelbottom=False
            )
            self.plots[-1].set_title(self.states_cat[index_fig])
            self.plots.append(fig.add_subplot(3, 1, 2))
            self.plots[-1].sharex(self.plots[0])
            self.plots[-1].set_ylabel(self.states[index_fig * 3 + 1])
            self.plots[-1].tick_params(
                axis="x", which="both", bottom=False, top=False, labelbottom=False
            )
            self.plots.append(fig.add_subplot(3, 1, 3))
            self.plots[-1].sharex(self.plots[0])
            self.plots[-1].set_ylabel(self.states[index_fig * 3 + 2])
            fig.align_ylabels()
            fig.subplots_adjust(wspace=0, hspace=0)
        if rot_mat:
            self.states += [
                r"$R_{11}$",
                r"$R_{12}$",
                r"$R_{13}$",
                r"$R_{21}$",
                r"$R_{22}$",
                r"$R_{23}$",
                r"$R_{31}$",
                r"$R_{32}$",
                r"$R_{33}$",
            ]
            self.states_cat += ["Rotation Matrix"]
            self.figs.append(plt.figure(figsize=(18, 10)))
            fig = self.figs[-1]
            fig.tight_layout()
            fig.suptitle(self.name)
            self.plots.append(fig.add_subplot(9, 1, 1))
            self.plots[-1].set_ylabel(self.states[12])
            self.plots[-1].tick_params(
                axis="x", which="both", bottom=False, top=False, labelbottom=False
            )
            self.plots[-1].set_title(self.states_cat[4])
            for index in range(1, 8):
                self.plots.append(fig.add_subplot(9, 1, index + 1))
                self.plots[-1].set_ylabel(self.states[index + 12])
                self.plots[-1].tick_params(
                    axis="x", which="both", bottom=False, top=False, labelbottom=False
                )
            self.plots.append(fig.add_subplot(9, 1, 9))
            self.plots[-1].set_ylabel(self.states[20])
            fig.align_ylabels()
            fig.subplots_adjust(wspace=0, hspace=0)


    def plot(
        self,
        times: npt.NDArray[np.float64],
        states: npt.NDArray[np.float64],
        *args,
        **kwargs,
    ):
        # if len(states.shape)==2:
        for index, state in enumerate(states):
            if np.all(np.isnan(state)) or times.shape[0]!= state.shape[0]:
                print("skipped one!")
                continue # to remove legend
            self.plots[index].plot(times, state, *args, **kwargs)
            self.plots[index].legend()

    def save(self):
        """
        Save the plots to the specified directory.
        """
        print("Saving figures", end="")
        for index, fig in enumerate(self.figs):
            fig.savefig(
                f"{self.name}_{self.states_cat[index]}.pdf", bbox_inches="tight"
            )
            print(".", end="")
        print(" done.")
