# Poisson-Boltzmann Solver Interface
Interactive interface and visualization of the numerical solution to the generalized Poisson-Boltzmann equation (PBE) based on the [bokeh](https://bokeh.pydata.org/en/latest/) package. The parameters for the PBE such as concentrations, surface interactions, dielectric response profiles and presence of impurities can be chosen interactively (see below for a screenshot). The interfacial profiles can be drawn directly into the plots. 

![screen](/interface.jpg)

## Prerequisites
Needs the [bokeh](https://bokeh.pydata.org/en/latest/) python package and the [Numerical Poisson-Boltzman Solver](https://github.com/woldeaman/numerical_PBE_solver) to be installed.

## Usage
An interactive session can be started from the directory of the repository by typing
```sh
bokeh serve web_server.py --show
```
