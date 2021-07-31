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

export HardDelta, SoftDelta

abstract type Program{T} <: CSDistribution{T} end
struct TypedSF{T} <: CSDistribution{T}
    sf::T
    input_type::DataType
    output_type::DataType
end
struct CompiledSF{T, S} <: CSDistribution{T}
    p::TypedSF{T}
    input::S
    addresses::Array{String}
    CompiledSF(p::TypedSF{T}, input::S, addresses::Array{String}) where {T,S} = input isa p.input_type ? new{T,S}(p, input, addresses) : error("Wrong input type")
end
compile(p::TypedSF{T}, input::S, addresses::Array{String}) where {T,S} = CompiledSF{T,S}(p, input, addresses)

(p::TypedSF)(x) = x isa p.input_type ? (y = p.sf(trace(), x); y isa p.output_type ? y : error("Wrong output type")) : error("Wrong input type")
(p::TypedSF)(x...) = x isa p.input_type ? (y = p.sf(trace(), x); y isa p.output_type ? y : error("Wrong output type")) : error("Wrong input type")
