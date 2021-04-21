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
from `Base`: `getindex`. The log weights are unnormalized. 
"""
struct SamplingResults{I<:InferenceType}
    interpretation :: I
    log_weights :: Array{Float64, 1}
    return_values :: Array{Any, 1}
    traces :: Array{Trace, 1}
end
Base.getindex(r :: SamplingResults{I}, k) where I <: InferenceType = [t.trace[k].value for t in r.traces]
Base.getindex(r :: SamplingResults{I}) where I <: InferenceType = r.return_values

StatsBase.mean(s :: SamplingResults{I}, k, n) where I<:InferenceType = StatsBase.mean(sample(s, k, n))
StatsBase.mean(s :: SamplingResults{I}, k) where I<:InferenceType = StatsBase.mean(s, k, length(s.log_weights))
StatsBase.std(s :: SamplingResults{I}, k, n) where I<:InferenceType = StatsBase.std(sample(s, k, n))
StatsBase.std(s :: SamplingResults{I}, k) where I<:InferenceType = StatsBase.std(s, k, length(s.log_weights))

export SamplingResults