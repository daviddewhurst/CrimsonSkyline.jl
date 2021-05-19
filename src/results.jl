abstract type InferenceType end 
abstract type SamplingResults{I<:InferenceType} end

### base types ###
@doc raw"""
    struct NonparametricSamplingResults{I} <: SamplingResults{I}
        interpretation :: I
        log_weights :: Array{Float64, 1}
        return_values :: Array{Any, 1}
        traces :: Array{Trace, 1}
    end

Wrapper for results of sampling. Implements the following methods 
from `Base`: `getindex`, `length`, `keys`.
Intepretation of log weights is dependent on `I`.
"""
struct NonparametricSamplingResults{I} <: SamplingResults{I}
    interpretation :: I
    log_weights :: Array{Float64, 1}
    return_values :: Array{Any, 1}
    traces :: Array{Trace, 1}
end
function Base.keys(r :: NonparametricSamplingResults{I}) where I <: InferenceType
    k = Set{Any}()
    for t in r.traces
        for a in keys(t)
            union!(k, (a,))
        end
    end
    k
end
Base.getindex(r :: NonparametricSamplingResults{I}, k) where I <: InferenceType = [t.trace[k].value for t in r.traces]
Base.getindex(r :: SamplingResults{I}) where I <: InferenceType = r.return_values
Base.length(r :: SamplingResults{I}) where I <: InferenceType = length(r.log_weights)

@doc raw"""
    function addresses(r::SamplingResults{I}) where I <: InferenceType

Get all addresses associated with the `SamplingResults` object,
``A = \bigcup_{t\in \text{traces}}\mathcal A_t``
"""
function addresses(r::SamplingResults{I}) where I <: InferenceType
    a = Set()
    for t in r.traces
        union!(a, collect(keys(t)))
    end
    a
end
function get_first_node(r::NonparametricSamplingResults{I}, a) where I<:InferenceType
    for t in r.traces
        if a in keys(t.trace)
            return t[a]
        end
    end
end

StatsBase.mean(s :: NonparametricSamplingResults{I}, k, n) where I<:InferenceType = StatsBase.mean(sample(s, k, n))
StatsBase.mean(s :: NonparametricSamplingResults{I}, k) where I<:InferenceType = StatsBase.mean(s, k, length(s.log_weights))
StatsBase.std(s :: NonparametricSamplingResults{I}, k, n) where I<:InferenceType = StatsBase.std(sample(s, k, n))
StatsBase.std(s :: NonparametricSamplingResults{I}, k) where I<:InferenceType = StatsBase.std(s, k, length(s.log_weights))

@doc raw"""
    struct ParametricSamplingResults{I} <: SamplingResults{I}
        interpretation :: I
        log_weights :: Array{Float64, 1}
        return_values :: Array{Any, 1}
        traces :: Array{Trace, 1}
        distributions :: Dict
    end

`distributions` maps from addresses to distributions, ``a \mapsto \pi^{(a)}_\psi(z)``, where 
``\pi^{(a)}_\psi(z)`` solves

```math
\max_\psi E_{z \sim p(z|x)}[\log \pi^{(a)}_\psi(z)].
```

The distributions are not used to generate values but only to score sampled values; values are 
still sampled from the posterior traces.
Right now, the parametric approximation is very simple: values with support over the 
negative orthant of  ``\mathbb R^D`` are approximated by (multivariate) normal distributions, while 
values with support over only the positive orthant of ``\mathbb{R}^D`` are approximated by 
(multivariate) lognormal distributions. This behavior is expected to change in the future.
"""
struct ParametricSamplingResults{I} <: SamplingResults{I}
    interpretation :: I
    log_weights :: Array{Float64, 1}
    return_values :: Array{Any, 1}
    traces :: Array{Trace, 1}
    distributions :: Dict
end

@doc raw"""
    function to_parametric(r::NonparametricSamplingResults{I}) where I<:InferenceType 
        
Converts a nonparametric sampling results object into one that additionally contains
a mapping from addresses to distributions. 
"""
function to_parametric(r::NonparametricSamplingResults{I}) where I<:InferenceType
    a_set = addresses(r)
    distributions = Dict()
    for a in a_set
        rep_node = get_first_node(r, a)
        if !rep_node.observed
            parametric_dist = parametric_posterior(rep_node.address, rep_node.dist, r)
            distributions[a] = parametric_dist
        end
    end
    ParametricSamplingResults{I}(r.interpretation, r.log_weights, r.return_values, r.traces, distributions)
end
Base.getindex(r :: ParametricSamplingResults{I}, k) where I <: InferenceType = r.distributions[k]
getsampled(r :: ParametricSamplingResults{I}, k) where I <: InferenceType = [t.trace[k].value for t in r.traces]

@doc raw"""
    function Distributions.logpdf(r :: A, v) where A <: AbstractArray

Interprets an array of objects as a delta distribution over those objects. If `v` is in the support set, 
returns ``-log |r|``. Otherwise, returns ``-\infty``. 
"""
function Distributions.logpdf(r :: A, v) where A <: AbstractArray
    if v in r
        -log(length(r))
    else 
        -Inf
    end
end

@doc raw"""
    function sample(t :: Trace, a, r :: NonparametricSamplingResults{I}; pa = ()) where I <: InferenceType

Treat a marginal site of a `SamplingResults` as a distribution, sampling from it into a trace.
"""
function sample(t :: Trace, a, r :: NonparametricSamplingResults{I}) where I <: InferenceType
    s = sample(r, a, 1)  # relies on inference-specific implementations
    n = node(s, a, r[a], false, EMPIRICAL)
    t[a] = n
    s
end

@doc raw"""
    function sample(t :: Trace, a, r::ParametricSamplingResults{I}; pa = ()) where I <: InferenceType

Treat a marginal site of a `SamplingResults` as a distribution, sampling from it into a trace.
"""
function sample(t::Trace, a, r::ParametricSamplingResults{I}) where I <:InferenceType
    s = sample(r, a, 1)  # relies on inference-specific implementations
    n = node(s, a, r.distributions[a], false, EMPIRICAL)
    t[a] = n
    s
end

function sample(t :: Trace, a, d, i :: Empirical; pa = ())
    connect_pa_ch!(t, pa, a)
    t[a].value
end

@doc raw"""
    function aic(r :: SamplingResults{I}) where I <: InferenceType

Computes an empirical estimate of the Akaike Information Criterion from a `SamplingResults`.
The formula is 
    
```math
\text{AIC}(r)/2 = \min_{t \in \text{traces}(r)}|\text{params}(t)| - \hat\ell(t),
```
    
where ``|\text{params}(t)|`` is the number of non-observed and non-deterministic sample nodes and
``\hat\ell(t)`` is the empirical maximum likelihood.
"""
function aic(r :: SamplingResults{I}) where I <: InferenceType
    min_aic = Inf
    for t in r.traces
        a = aic(t)
        if a < min_aic
            min_aic = a
        end
    end
    min_aic
end

function parametric_posterior(address, dist, result :: SamplingResults)
    sd = size(dist)
    lsd = length(sd)
    lsd > 1 && error("parametric posterior only supported for rank <= 1 tensors")
    etype = eltype(dist)
    if lsd == 0 && etype === Float64
        univariate_continuous_parametric_posterior(address, dist, result)
    elseif lsd == 0 && etype === Int64
        univariate_discrete_parametric_posterior(address, dist, result)
    elseif lsd == 1 && etype === Float64
        multivariate_continuous_parametric_posterior(address, dist, result)
    end
end 

univariate_continuous_parametric_posterior(address, dist, result) = insupport(dist, -1.0) ? fit_mle(Normal, result[address]) : fit_mle(LogNormal, result[address])
univariate_discrete_parametric_posterior(address, dist, result) = insupport(dist, -1) ? fit_mle(DiscreteUniform, result[address]) : fit_mle(Poisson, result[address])
multivariate_continuous_parametric_posterior(address, dist, result) = insupport(dist, -1.0 .* ones(size(dist))) ? fit_mle(MvNormal, hcat(result[address]...)) : MvLogNormal(fit_mle(MvNormal, log.(hcat(result[address]...))))

@doc raw"""
    function reflate!(r::ParametricSamplingResults{I}, mapping::Dict) where I<:InferenceType

Reflates the scale parameters of the `distributions` of `r`. The inferred scale parameters 
approximate the true uncertainty of the posterior distribution under the critical assumption
that the data generating process (DGP) remains constant. When the DGP is nonstationary,
it is necessary to intervene and increase the value of the scale parameters to properly 
account for uncertainty. `mapping` is a dict of `address => value` where `value` is either
a float or a positive definite matrix depending on if the distribution associated with `address`
is scalar- or vector-valued. 
"""
function reflate!(r::ParametricSamplingResults{I}, mapping::Dict) where I<:InferenceType
    for (a, value) in mapping
        d = r.distributions[a]
        p = params(d)
        r.distributions[a] = typeof(d)(p[1], value)
    end
end

export NonparametricSamplingResults, sample, aic
export to_parametric, getsampled, reflate!