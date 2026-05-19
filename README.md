# Wavefront Propagation

> Notebook-based wavefront propagation and dynamical diffraction simulations for polychromatic X-ray DFXM — updated scripts from the DTU Multiscale Imaging group's forward-simulation project.

## What this is

This repository bundles two computational modules used in the simulation pipeline for Dark-Field X-ray Microscopy with polychromatic beams. The first module (`forward_simulation_polychromatic`) implements a three-stage DFXM forward simulation: numerical integration of the incident wavefield, free-space propagation to the detector plane, and (partially) detector response modelling. The second module (`dynamical_diffraction`) computes rocking curves and diffraction patterns from plate-like crystals using the Takagi-Taupin equations — solved exactly via Fourier decomposition for perfect crystals, and via finite differences for strained crystals. Together they cover the wave-optics corrections that sit beyond the geometrical-optics model in `Geometrical_Optics_master`.

## Stack

- **Language:** Jupyter Notebook (96.7%), Python (3.3%)
- **Key libraries:** NumPy, SciPy; standard scientific Python stack
- **Theoretical reference:** [arXiv:1703.04100](https://arxiv.org/abs/1703.04100) — dynamical X-ray diffraction in coherent imaging (uses the Takagi-Taupin framework).
- **License:** not specified (research code — check before reuse)

## How to run

Research code — not packaged. Clone and open notebooks interactively:

```bash
git clone https://github.com/borgi-s/wavefront_propagation.git
cd wavefront_propagation
pip install numpy scipy jupyter
jupyter lab
```

Each sub-module has its own `setup.py` and expects a `parameters.ini` configuration file — see the sub-directory READMEs for the required fields before running.

## Background

Developed during PhD research at DTU Physics as part of the Multiscale Imaging group's forward-simulation work for polychromatic DFXM. This repository is an updated fork of [Multiscale-imaging/forward_simulation_polychromatic](https://github.com/Multiscale-imaging/forward_simulation_polychromatic), incorporating fixes and extensions made during the PhD. Latest tagged release: v1.001 (September 2023).
