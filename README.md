# CrimsonSkyline.jl

This is a simple trace-based probabilistic programming language embedded in Julia. 

+ It implements both a higher-order PPL and first-order PPL. A trace can be compiled into an intermediate
    representation or into a factor graph.
+ Inference is programmable. For example, Metropolis-Hastings is implemented simply by incrementally modifying
    traces with user-defined proposal kernels.
+ It includes a library of composable effect handlers that change the interpretation of a program's stochastic compute graph.
+ It does not have many package dependencies.

CrimsonSkyline.jl currently supports two families of inference algorithms:

+ Importance sampling
    + Likelihood weighting
+ Metropolis-Hastings
    + Independent prior proposal
    + Generic user-defined proposal

We might implement more inference algorithms soon.

#### License, copyright, and other info
+ CrimsonSkyline.jl is released under the GNU GPL v3.
+ Copyright David Rushing Dewhurst and Charles River Analytics Inc., 2021 - present.
+ Development repository is at [gitlab.com/daviddewhurst/CrimsonSkyline.jl](gitlab.com/daviddewhurst/CrimsonSkyline.jl)
+ Mirrored at [github.com/daviddewhurst/CrimsonSkyline.jl](github.com/daviddewhurst/CrimsonSkyline.jl)