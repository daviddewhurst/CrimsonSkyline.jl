abstract type CSDistribution{T} end
abstract type Delta{T} <: CSDistribution{T} end

struct HardDelta{T<:Function} <: Delta{T}
    fact::T
end
struct SoftDelta{T<:Function} <: Delta{T}
    fact::T
    negative_evidence::Float64
end

Distributions.rand(d::Delta) = d.fact()
Distributions.logpdf(d::HardDelta{T}, x) where T = d.fact(x) ? 0.0 : -Inf
Distributions.logpdf(d::SoftDelta{T}, x) where T = d.fact(x) ? log(1.0 - exp(d.negative_evidence)) : d.negative_evidence

export HardDelta, SoftDelta, logprob