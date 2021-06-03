# pneumoinfer

[![DOI](https://zenodo.org/badge/243589029.svg)](https://zenodo.org/badge/latestdoi/243589029)

This is a class structure and notebook implementation for inferring and simulating a type of multi-state models with a counting memory which records previous state occupations. The inference is currently written for pneumococcus-like models but may be generalised to others in the future.

The nifty thing that `pneumoinfer` does is it implements an approximate ODE description of the system (which would otherwise have to be explicitly simulated) and hence there is a radical reduction in time required to compute the likelihood of the model with respect to a given data set in comparison to other methods. The details behind this mathematical trickery are given in the theory documentation (which also serves as a convenient code use tutorial), which can be found in the `/notebooks` directory and/or can be read [in nbviewer here](https://nbviewer.jupyter.org/github/umbralcalc/pneumoinfer/blob/master/notebooks/theory-docs.ipynb).

