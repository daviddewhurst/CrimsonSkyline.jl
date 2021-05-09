# CrimsonSkyline.jl

This is a simple trace-based universal probabilistic programming language embedded in Julia. 

+ Inference is programmable. For example, Metropolis-Hastings is implemented simply by incrementally modifying
    traces with user-defined proposal kernels.
    However, there are also user-friendly default inference routines built-in. 
+ It includes a library of composable effects that change the interpretation of a program's stochastic compute graph.

CrimsonSkyline.jl currently supports three families of sampling-based inference algorithms:

+ Importance sampling
    + Likelihood weighting
    + Generic user-defined proposal
+ Metropolis-Hastings
    + Independent prior proposal
    + Generic user-defined proposal
+ Nested sampling

We might implement more inference algorithms soon.

## Examples

See the `examples` directory:
    + `basic.jl`: Bayesian linear regression and serving a posterior predictive model
    + `clustering.jl`: open-universe clustering model where the number of clusters is *a priori* unbounded
    + `time_series.jl`: basic time series inference and model comparison
    + `forecast.jl`: time series inference, posterior predictive, and generation of online forecasts using effects.

#### Other information
+ CrimsonSkyline.jl is released under the GNU GPL v3.
+ Copyright David Rushing Dewhurst and Charles River Analytics Inc., 2021 - present.
+ Development repository is at [https://gitlab.com/daviddewhurst/CrimsonSkyline.jl](https://gitlab.com/daviddewhurst/CrimsonSkyline.jl)
+ Mirrored at [https://github.com/daviddewhurst/CrimsonSkyline.jl](https://github.com/daviddewhurst/CrimsonSkyline.jl)