# CrimsonSkyline.jl

## trace
To a large extent, `Node`s have local control over the behavior of inference algorithms via their 
`interpretation`. There are a variety of `Interpretation`s. The type hierarchy is flat:
```
abstract type Interpretation end
struct Nonstandard <: Interpretation end
struct Standard <: Interpretation end
struct Replayed <: Interpretation end
struct Conditioned <: Interpretation end 
struct Blocked <: Interpretation end
struct Deterministic <: Interpretation end
struct Proposed <: Interpretation end
```
```@autodocs
Modules = [CrimsonSkyline]
Pages = ["trace.jl"]
```

## effects
A library of functions that change the interpretation of some or all nodes in a trace.
```@autodocs
Modules = [CrimsonSkyline]
Pages = ["effects.jl"]
```

## graph
```@autodocs
Modules = [CrimsonSkyline]
Pages = ["graph.jl"]
```

## cpt
Conditional probability tables (CPT) are a work in progress. Currently there are no implemented inference algorithms that operate on CPTs. 
```@autodocs
Modules = [CrimsonSkyline]
Pages = ["cpt.jl"]
```

## importance
Importance sampling algorithms and utilities. 
Currently the following importance sampling algorithms are implemented:

+ Likelihood weighting

```@autodocs
Modules = [CrimsonSkyline]
Pages = ["importance.jl"]
```

## metropolis
Metropolis algorithm and utilities. Currently the following algorithms are implemented for both 
FOPPL and HOPPL programs:

+ Independent prior proposal
+ Arbitrary single- or multi-site proposal

Here are two examples of inference using the arbitrary MH step interface. Consider the following generative
function:
```
function normal_model(t :: Trace, data :: Vector{Float64})
    loc = sample(t, :loc, Normal(0.0, 10.0))
    scale = sample(t, :scale, LogNormal())
    for i in 1:length(data)
        observe(t, (:obs, i), Normal(loc, scale), data[i])
    end
end
```
To learn an approximate posterior for `:loc` and `:scale`, we will introduce two proposal kernels:
```
loc_proposal(old_t :: Trace, new_t :: Trace, data) = propose(new_t, :loc, Normal(old_t[:loc].value, 0.25))
scale_proposal(old_t :: Trace, new_t :: Trace, data) = propose(new_t, :scale, truncated(Normal(old_t[:scale].value, 0.25), 0.0, Inf))
```
Note that while `loc_proposal` is symmetric, `scale_proposal` is not. To perform inference, we pass these kernels to `mh_step` in 
a loop after first drawing a random trace:
```
t = trace()
normal_model(t, data)
for i in 1:niter
    t = mh_step(t, normal_model, loc_proposal; params = (data,))
    t = mh_step(t, normal_model, scale_proposal; params = (data,))
end
```
In this case, inference was fairly successful:
```
[ Info: True loc = 4.0
[ Info: True std = 1.0
[ Info: inferred E[loc] = 4.022688081613082
[ Info: inferred E[scale] = 0.9696559373495869
[ Info: approximate p(x) = sum_z p(x|z) = -138.63530736205144
```
As a less trivial (but still contrived!) inference example, we can infer the posterior distribution of a latent discrete random variable
in an open-universe model: 
```
function random_sum_model(t :: Trace, data)
    n = sample(t, :n, Geometric(0.1))
    loc = 0.0
    for i in 1:(n + 1)
        loc += sample(t, (:loc, i), Normal())
    end
    obs = Array{Float64, 1}()
    for j in 1:length(data)
        o = observe(t, (:data, j), Normal(loc, 1.0), data[j])
        push!(obs, o)
    end
    obs
end
```
(N.B.: we write the model is this form for pedagogic reasons; there is a far more efficient way to express the latent structure 
of this model, namely ``n \sim \text{Geometric}(0.1)``, ``\text{loc} \sim \text{Normal}(0, n)``.)
We are interested in learning the posterior distribution of `:n`. We introduce two proposal distributions, one for the latent discrete rv
and another generic proposal for the location increments:
```
function random_n_proposal(old_trace, new_trace, params...)
    old_n = float(old_trace[:n].value)
    if old_n > 0
        propose(new_trace, :n, Poisson(old_n))
    else
        propose(new_trace, :n, Poisson(1.0))
    end
end

gen_loc_proposal(old_trace, new_trace, ix, params...) = propose(new_trace, (:loc, ix), Normal(old_trace[(:loc, ix)].value, 0.25))
```
We again conduct inference by simply applying proposals within a loop. This time, the number of location increment proposals we need to 
construct is dependent on the sampled values of the latent random variable. We can either create these proposals on the fly as they're 
needed or create what is nearly guaranteed to be enough of them before any inference is performed, e.g., 
`loc_proposals = [(o, n, params...) -> gen_loc_proposal(o, n, i, params...) for i in 1:100]`.
Now that we have what we need, we can conduct inference:
```
t = trace()
random_sum_model(t, data)
...
for i in 1:niter
    t = mh_step(t, random_sum_model, random_n_proposal; params=(data,))
    for j in 1:(t[:n].value + 1)
        t = mh_step(t, random_sum_model, loc_proposals[j]; params=(data,))
    end
    push!(ns, t[:n].value)
end
```
Our inference results look promising:
```
[ Info: True :n = 9
[ Info: Posterior E[:n] = 7.581604598850287
```
For more examples, check back soon.

```@autodocs
Modules = [CrimsonSkyline]
Pages = ["metropolis.jl"]
```

## fusion

Methods to perform semi-analytical simplification and inference. 

### fusion/distributions
A collection of distributions that aren't in `Distributions.jl`. These distributions are useful in their own right, but are 
also arise when fusing pairs or triples of known distributions.

```@autodocs
Modules = [CrimsonSkyline]
Pages = ["fusion/distributions.jl"]
```

## Index
```@index
```