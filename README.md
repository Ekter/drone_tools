# drone-tools
Various tools for drone in python. Includes online cascade PID tuner using marimo and pretty drone state visualiser.

## Getting started

I don't think the project if ready for packaging yet, so you must clone the repository for now.

I advise creating a virtual environment and then installing the requirements from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Having latex allows for nice plots, but it is not required. If you don't have/want it, comment lines 9 and 10 of `visualize.py`.

## Usage

You can just import the tools from outside the package using relative imports.

If you want to use the cascade PID tuner, launch it with marimo:

```bash
marimo edit drn_sim_ma.py
```

You can change the PID gains using the sliders and see the behaviour of the drone in the simulation.
