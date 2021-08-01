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
    input_type::Union{DataType,Union}
    output_type::Union{DataType,Union}
end
struct CompiledSF{T, S} <: CSDistribution{T}
    p::TypedSF{T}
    input::S
    addresses::Array{String}
    input_is_evidence::Bool
    function CompiledSF{T,S}(p::TypedSF{T}, input::S, addresses::Vector{String}) where {T,S}
        !(input isa p.input_type) && error("Wrong input type")
        input_usage = guess_input_is_evidence(p.sf, input)
        new{T,S}(p, input, addresses, input_usage)
    end
end
compile(p::TypedSF{T}, input::S, addresses::Vector{String}) where {T,S} = CompiledSF{T,S}(p, input, addresses)

(p::TypedSF)(x) = x isa p.input_type ? (y = p.sf(trace(), x); y isa p.output_type ? y : error("Wrong output type")) : error("Wrong input type")
(p::TypedSF)(x...) = x isa p.input_type ? (y = p.sf(trace(), x); y isa p.output_type ? y : error("Wrong output type")) : error("Wrong input type")

function guess_input_is_evidence(f, i) :: Bool
    t = trace()
    f(t, i)
    for (address, node) in t.trace
        node.observed ? (return true) : nothing
    end
    false
end

function select_csf_sample(csf::CompiledSF, samples)
    if length(csf.addresses) == 1
        return samples[csf.addresses[1]][end]
    else
        return Dict(
            a => samples[a][end] for a in csf.addresses
        )
    end
end

function sample(csf::CompiledSF)
    if csf.input_is_evidence
        mh(csf.p.sf; params = (csf.input,))
    else
        forward_sampling(csf.p.sf; params = (csf.input,))
    end
end

function Distributions.rand(csf::CompiledSF)
    samples = sample(csf)
    select_csf_sample(csf, samples)
end

function node(value, address :: A, dist :: CompiledSF, is_obs :: Bool, i :: Interpretation) where A
    T = typeof(value)
    samples = sample(dist)
    lp = logprob(samples, dist.addresses)
    ParametricNode{A, CompiledSF, T}(address, dist, value, lp, lp, is_obs, Array{Node, 1}(), Array{Node, 1}(), i, i)
end

export TypedSF, CompiledSF, compile