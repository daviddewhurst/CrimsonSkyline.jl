abstract type InferenceType end 
abstract type Results{I<:InferenceType} end
abstract type SamplingResults{I} <: Results{I} end

struct BareResults{I} <: Results{I}
    interpretation :: I
    values :: DefaultDict{Any, Vector{Any}}
end
export BareResults

struct StructuredResults{I} <: SamplingResults{I}
    interpretation :: I
    values :: Dict{String, Vector{Any}}
    log_weights :: Vector{Float64}
end
Base.keys(sr::StructuredResults) = keys(sr.values)
Base.getindex(sr::StructuredResults, k) = Base.getindex(sr.values, k)
export StructuredResults

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
function Base.getindex(r :: NonparametricSamplingResults{I}, k) where I <: InferenceType
    results = []
    for t in r.traces
        if k in keys(t)
            push!(results, t.trace[k].value)
        end
    end
    convert(Vector{typeof(results[1])}, results)  # assuming address type stability
end
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

function parametric_posterior(address, dist, result :: SamplingResults)
    sd = size(dist)
    lsd = length(sd)
    lsd > 1 && error("parametric posterior supported for only rank <= 1 tensors")
    etype = eltype(dist)
    if lsd == 0 && etype === Float64
        univariate_continuous_parametric_posterior(address, dist, result)
    elseif lsd == 0 && etype === Int64
        univariate_discrete_parametric_posterior(address, dist, result)
    elseif lsd == 1 && etype === Float64
        multivariate_continuous_parametric_posterior(address, dist, result)
    else
        error("Parametric posterior supported for only univariate and continuous multivariate distributions")
    end
end 

parametric_posterior(address, dist::D, result::SamplingResults) where D <: Categorical = fit_mle(Categorical, length(dist.p), results[address])
parametric_posterior(address, dist::D, results::SamplingResults) where D <: Beta = fit(Beta, results[address])
parametric_posterior(address, dist::D, results::SamplingResults) where D <: Dirichlet = fit_mle(Dirichlet, hcat(results[address]...); maxiter=5000, tol=1.0e-12)
parametric_posterior(address, dist::D, results::SamplingResults) where D <: Binomial = fit_mle(Binomial, dist.n, results[address])

univariate_continuous_parametric_posterior(address, dist, result) = insupport(dist, -1.0) ? fit(Normal, result[address]) : fit(LogNormal, result[address])
univariate_discrete_parametric_posterior(address, dist, result) = insupport(dist, -1) ? fit(DiscreteUniform, result[address]) : fit(Poisson, result[address])
function multivariate_continuous_parametric_posterior(address, dist, result)
    if insupport(dist, -1.0 .* ones(size(dist)))
        mat = hcat(result[address]...)
        m = vec(mean(mat, dims=2))
        s = vec(std(mat, dims=2))
        MvNormal(m, s)
    else 
        mat = log.(hcat(result[address]...))
        m = vec(mean(mat, dims=2))
        s = vec(std(mat, dims=2))
        MvLogNormal(MvNormal(m, s))
    end
end

function logprob(r::Union{NonparametricSamplingResults{I},ParametricSamplingResults{I}}, address::T) where {I<:InferenceType, T}
    denom = 0
    lps = 0.0
    for trace in r.traces
        for (a, node) in trace.trace
            if a == address
                denom += 1
                lps += node.logprob_sum
            end
        end
    end
    lps / denom
end
function logprob(r::Union{NonparametricSamplingResults{I},ParametricSamplingResults{I}}, addresses::Vector{T}) where {I<:InferenceType, T}
    denom = 0
    lps = 0.0
    for trace in r.traces
        for (a, node) in trace.trace
            if a in addresses
                denom += 1
                lps += node.logprob_sum
            end
        end
    end
    lps / denom
end
export logprob

export NonparametricSamplingResults, ParametricSamplingResults, sample, aic
export to_parametric, getsampled, addresses, get_first_node