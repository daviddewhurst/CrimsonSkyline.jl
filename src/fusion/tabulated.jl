function beta_binomial_betabinomial(beta_dist, binomial_dist)
    BetaBinomial(binom_dist.n, beta_dist.α, beta_dist.β)
end

function gamma_poisson_negativebinomial(gamma_dist, poisson_dist)
    r = shape(gamma_dist)
    theta = scale(gamma_dist)
    p = theta / (theta + 1.0)
    NegativeBinomial(r, p)
end

function gamma_exponential_lomax(gamma_dist, exponential_dist)
    k = shape(gamma_dist)
    theta = scale(gamma_dist)
    Lomax(k, theta)
end

__pairs__ = Dict(
    (Beta{Float64}, Binomial{Float64}) => beta_binomial_betabinomial,
    (Gamma{Float64}, Poisson{Float64}) => gamma_poisson_negativebinomial,
    (Gamma{Float64}, Exponential{Float64}) => gamma_exponential_lomax
)

function fuse_pair(a, b)
    p = (typeof(a), typeof(b))
    if p in keys(__pairs__)
        __pairs__[p](a, b)
    else
        nothing
    end
end