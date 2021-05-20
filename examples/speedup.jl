import Pkg
Pkg.activate("..")

using Distributions: Beta, Bernoulli, Normal, LogNormal, truncated
using CrimsonSkyline
using Random: seed!
using Logging
using StatsBase: mean

seed!(2021)

function coin_model(t::Trace, data::Vector{Bool})
    bias = sample(t, "bias", Beta(3.0, 3.0))
    coin = Bernoulli(bias)
    for (i, d) in enumerate(data)
        observe(t, "flip $i", coin, d)
    end
end

function plated_coin_model(t::Trace, data::Vector{Bool})
    bias = sample(t, "bias", Beta(3.0, 3.0))
    plate(t, observe, "flips", Bernoulli(bias), data)
end

function normal_model(t::Trace, data::Vector{Float64})
    loc = sample(t, "loc", Normal())
    scale = sample(t, "scale", LogNormal())
    for (i, d) in enumerate(data)
        observe(t, "obs $i", Normal(loc, scale), d)
    end
end

function plated_normal_model(t::Trace, data::Vector{Float64})
    loc = sample(t, "loc", Normal())
    scale = sample(t, "scale", LogNormal())
    plate(t, observe, "obs", Normal(loc, scale), data)
end

loc_proposal(t0, t1, params...) = propose(t1, "loc", Normal(t0["loc"].value, 0.25))
scale_proposal(t0, t1, params...) = propose(t1, "scale", truncated(Normal(t0["scale"].value, 0.25), 0.01, Inf))

function slow()
    @warn "\n~~~Coin model: not using plate~~~"
    data = [true, true, false, false, true, true, true, true]
    _ = mh(coin_model; params = (data,))  # to compile
    @time results = mh(coin_model; params = (data,))
    @info "Mean posterior bias estimate: $(mean(results, "bias"))\n"

    @warn "\n~~~Normal model, not using plate or typed trace"
    seed!(2021)
    data = randn(10) .* 2.0 .+ 2.0
    kernels = [loc_proposal, scale_proposal]
    _ = mh(normal_model, kernels; params = (data,), burn=1000, thin=50, num_iterations=1, inverse_verbosity=Inf)  # to compile
    @time results = mh(normal_model, kernels; params = (data,), burn=1000, thin=50, num_iterations=10000, inverse_verbosity=Inf)
end

function fast()
    @warn "\n~~~Coin model: using plate~~~\n"
    data = [true, true, false, false, true, true, true, true]
    _ = mh(plated_coin_model; params = (data,))  # to compile
    @time results = mh(plated_coin_model; params = (data,))
    @info "Mean posterior bias estimate: $(mean(results, "bias"))\n"

    @warn "\n~~~Normal model, using typed trace~~~"
    seed!(2021)
    data = randn(10) .* 2.0 .+ 2.0
    kernels = [loc_proposal, scale_proposal]
    types = (String, Float64)
    _ = mh(normal_model, kernels, types; params = (data,), burn=1000, thin=50, num_iterations=1, inverse_verbosity=Inf)  # to compile
    @time results = mh(normal_model, kernels, types; params = (data,), burn=1000, thin=50, num_iterations=10000, inverse_verbosity=Inf)

    @warn "\n~~~Normal model, using plate~~~"
    seed!(2021)
    data = randn(10) .* 2.0 .+ 2.0
    kernels = [loc_proposal, scale_proposal]
    _ = mh(plated_normal_model, kernels; params = (data,), burn=1000, thin=50, num_iterations=1, inverse_verbosity=Inf)  # to compile
    @time results = mh(plated_normal_model, kernels; params = (data,), burn=1000, thin=50, num_iterations=10000, inverse_verbosity=Inf)
end

function main()
    slow()
    fast()
end

main()