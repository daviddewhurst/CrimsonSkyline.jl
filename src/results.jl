abstract type InferenceType end 

### base types ###
@doc raw"""
    struct SamplingResults{I<:InferenceType}
        interpretation :: I
        log_weights :: Array{Float64, 1}
        return_values :: Array{Any, 1}
        traces :: Array{Trace, 1}
    end

Wrapper for results of sampling. Implements the following methods 
from `Base`: `getindex`, `length`, `keys`.
Intepretation of log weights is dependent on `I`.
"""
struct SamplingResults{I<:InferenceType}
    interpretation :: I
    log_weights :: Array{Float64, 1}
    return_values :: Array{Any, 1}
    traces :: Array{Trace, 1}
end
function Base.keys(r :: SamplingResults{I}) where I <: InferenceType
    k = Set{Any}()
    for t in r.traces
        for a in keys(t)
            union!(k, (a,))
        end
    end
    k
end
Base.getindex(r :: SamplingResults{I}, k) where I <: InferenceType = [t.trace[k].value for t in r.traces]
Base.getindex(r :: SamplingResults{I}) where I <: InferenceType = r.return_values
Base.length(r :: SamplingResults{I}) where I <: InferenceType = length(r.log_weights)

StatsBase.mean(s :: SamplingResults{I}, k, n) where I<:InferenceType = StatsBase.mean(sample(s, k, n))
StatsBase.mean(s :: SamplingResults{I}, k) where I<:InferenceType = StatsBase.mean(s, k, length(s.log_weights))
StatsBase.std(s :: SamplingResults{I}, k, n) where I<:InferenceType = StatsBase.std(sample(s, k, n))
StatsBase.std(s :: SamplingResults{I}, k) where I<:InferenceType = StatsBase.std(s, k, length(s.log_weights))

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
    function sample(t :: Trace, a, r :: SamplingResults{I}; pa = ()) where I <: InferenceType

**EXPERIMENTAL**: treat a marginal site of a `SamplingResults` as a distribution, sampling from it into a trace
using nonstandard interpretation.
"""
function sample(t :: Trace, a, r :: SamplingResults{I}) where I <: InferenceType
    s = sample(r, a, 1)  # relies on inference-specific implementations
    n = node(s, a, r[a], false, EMPIRICAL)
    t[a] = n
    s
end

function sample(t :: Trace, a, d, i :: Empirical; pa = ())
    connect_pa_ch!(t, pa, a)
    t[a].value
end

export SamplingResults, sample