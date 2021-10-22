# CrimsonSkyline.jl

This is a trace-based universal probabilistic programming language embedded in Julia. 

Nice-to-haves:
+ Inference is programmable. For example, Metropolis-Hastings is implemented simply by incrementally modifying
    traces with user-defined proposal kernels.
    However, there are also user-friendly default inference routines built-in. 
+ It includes a library of composable effects that change the interpretation of a program's stochastic compute graph.

Why-wouldn't-you-haves:
+ As a Julia library should, it actually supports multiple dispatch. (C.f. other universal PPLs embedded in Julia that do not.)

Pet-peeve-haves:
+ It doesn't maintain any global state, e.g., there is no global parameter store. All inference is controlled locally by both the (user-defined) inference program and nodes themselves.

## Inference algorithms
CrimsonSkyline.jl currently supports three families of sampling-based inference algorithms:

+ Importance sampling
    + Likelihood weighting
    + Generic user-defined proposal
+ Metropolis-Hastings
    + Independent prior proposal
    + Generic user-defined proposal
+ [Nested sampling](https://projecteuclid.org/journals/bayesian-analysis/volume-1/issue-4/Nested-sampling-for-general-Bayesian-computation/10.1214/06-BA127.full)

We might implement more inference algorithms sometime.

## Examples

See the `examples` directory:

+ `changepoint.jl`: changepoint detection in British coal mining disaster data
+ `clustering.jl`: open-universe clustering model where the number of clusters is *a priori* unbounded. Demonstrates a true open universe problem without the artifical upper bound introduced in e.g., a typical variational inference approach.
+ `coin_flip.jl`: classic "how biased is this coin" problem
+ `effects.jl`: usage of effect functionals that change program interpretation
+ `forecast.jl`: time series inference, posterior predictive, and generation of online forecasts using effects
+ `program_node.jl`: compilation of a probabilistic program into a sample node
+ `regression.jl`: Bayesian linear regression and serving a posterior predictive model
+ `speedup.jl`: Illustrates ways to speed up inference, including using [plates](https://en.wikipedia.org/wiki/Plate_notation) and typed traces.
+ `time_series.jl`: basic time series inference and model comparison

Here is perhaps "the" canonical example. Suppose we have a coin of unknown fairness 
`bias ~ Beta(alpha, beta)`. Arbitrarily we'll say we believe `alpha = beta = 3.0`. 
We flip the coin a bunch of times and observe the following array of heads / tails:
`data = [true, true, false, false, true, true, true, true]` (true maps to heads here).
We introduce a simple stochastic function to model this process.

```
function coin_model(t::Trace, data::Vector{Bool})
    bias = sample(t, "bias", Beta(3.0, 3.0))
    coin = Bernoulli(bias)
    for (i, d) in enumerate(data)
        observe(t, "flip $i", coin, d)
    end
end
```

Though much more tuning and customizable inference is possible, we can get some quick
and dirty (and hopefully accurate!) inference results by calling `mh` (for Metropolis-Hastings) on `coin_model`:

```
inference_results = mh(coin_model; params = (data,))
```
The analytical posterior mean `"bias"` is roughly `0.6429` -- see `examples/coin_flip.jl`, or any introductory probability course, for more.
Computing `mean(inference_results, "bias")` should return a value that is very close to this number (again, run `examples/coin_flip.jl` to see this in action).

#### Other information
CrimsonSkyline.jl is released under the MIT license. Copyright David Rushing Dewhurst and Charles River Analytics Inc., 2021 - present. The development repository is at [https://gitlab.com/daviddewhurst/CrimsonSkyline.jl](https://gitlab.com/daviddewhurst/CrimsonSkyline.jl), and it is mirrored at [https://github.com/daviddewhurst/CrimsonSkyline.jl](https://github.com/daviddewhurst/CrimsonSkyline.jl) for Julia packaging purposes.

`CrimsonSkyline.jl` has been tested under MacOS High Sierra and Manjaro 21.0 (Ornara); it *should* work on any reasonably modern 64-bit Linux distribution.
It has *not* been tested on any Windows distribution. 

#### Support
We are grateful for financial support from DARPA. 
This material is based upon work supported by the NIWC Atlantic under Contract No. N6523620C8005. 
Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the NIWC Atlantic.
