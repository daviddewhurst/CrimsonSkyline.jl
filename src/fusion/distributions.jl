@doc raw"""
    abstract type CSDistribution{D} end

Abstract type from which all CrimsonSkyline distributions subtype.
All subtypes implement the following methods from `Distributions.jl`: 
`rand` and `logpdf`. Unless otherwise specified, these subtypes do *not* 
implement `logpdf.`, which may change in the future.
"""
abstract type CSDistribution{D} end

@doc raw"""
    struct Lomax{D} <: CSDistribution{D}
        α :: D
        θ :: D 
        γ :: Gamma{D}
    end

A Lomax distribution, which is a power-law distribution supported on ``[0, \infty)``.
It has the pdf 

``p(x | \alpha,\ \theta) = -(\alpha + 1) [ \log(\alpha \theta) + \log(1 + x \theta) ].``

Sampling from this distribution is accomplished via a gamma - exponential mixture. 
A value ``g \sim \text{Gamma}(\alpha,\ \theta)`` is drawn, and then 
the value ``e \sim \text{Exponential}(g)`` is returned. 
"""
struct Lomax{D} <: CSDistribution{D}
    α :: D
    θ :: D 
    γ :: Gamma{D}
end

@doc raw"""
    function Lomax(α :: D, θ :: D) where D <: Real

Outer constructor for Lomax struct.
"""
function Lomax(α :: D, θ :: D) where D <: Real
    γ = Gamma{D}(α, θ)
    Lomax{D}(α, θ, γ)
end

export Lomax, lomax

function Distributions.rand(d :: Lomax{D}) where D <: Real
    g = Distributions.rand(d.γ)
    Distributions.rand(Exponential(g))
end

function Distributions.logpdf(d :: Lomax{D}, x) where D <: Real
    # note that θ = 1 / λ in the typical definition of lomax pdf
    -1.0 * (d.α + 1.0) * (log(d.α * d.θ) + log(1.0 + x * d.θ))
end

export rand, logpdf