import Pkg
Pkg.activate("..")

using CrimsonSkyline
using Distributions: Normal, LogNormal, MvNormal, truncated
using Logging
using StatsBase: mean, std
using Random: seed!

seed!(2021)

const Maybe{T} = Union{T, Nothing}

function eye(d :: Int)
    mat = zeros(Float64, d, d)
    for i in 1:d
        mat[i, i] = 1.0
    end
    mat
end

function linear_model!(t :: Trace, out :: Vector{Float64}, X :: Matrix{Float64}, y :: Maybe{Vector{Float64}})
    size_X = size(X)
    D = size_X[1]
    N = size_X[2]
    β = sample(t, :β, MvNormal(zeros(D), eye(D)))
    scale = sample(t, :scale, LogNormal(0.0, 1.0))
    loc = X' * β
    if y === nothing
        for n in 1:N
            out[n] = sample(t, (:y, n), Normal(loc[n], scale))
        end
    else
        for n in 1:N
            out[n] = observe(t, (:y, n), Normal(loc[n], scale), y[n])
        end
    end
end

beta_proposal(old_t :: Trace, new_t :: Trace, _, _, _) = propose(new_t, :β, MvNormal(old_t[:β].value, 0.5 .* eye(length(old_t[:β].value))))
scale_proposal(old_t :: Trace, new_t :: Trace, _, _, _) = propose(new_t, :scale, truncated(Normal(old_t[:scale].value, 0.25), 0.0, Inf))

function main()
    @info "Bayesian linear regression"
    D = 10
    N = 200
    X = randn(D, N)
    out = Vector{Float64}(undef, N)
    @info "Using model to generate dataset"
    t = trace()
    linear_model!(t, out, X, nothing)
    y = [t[(:y, n)].value for n in 1:N]
    β_true = t[:β].value
    @info "True regression coefficients: $β_true"

    @info "Conducting inference using user-defined Metropolis kernels"
    @time results = mh(
        linear_model!,
        [beta_proposal, scale_proposal];
        params = (out, X, y),
        burn = 1000,
        thin = 50,
        num_iterations = 5000,
        inverse_verbosity = 500
    )
    β_mean = mean(results, :β)
    β_std = std(results, :β)
    @info "Posterior mean regression coefficients = $β_mean"
    @info "Posterior std regression coefficients = $β_std"
end

main()