import Pkg
Pkg.activate("..")

using Distributions: Normal, LogNormal, truncated
using CrimsonSkyline
using Random: seed!
using Logging
using StatsBase: mean, std
using PrettyPrint: pprintln

seed!(2021)
const Maybe{T} = Union{T, Nothing}

# version for inference -- no return value, so lower memory reqd

function random_walk(t :: Trace, data :: Array{Float64, 2})
    loc = sample(t, :loc, Normal())
    scale = sample(t, :scale, LogNormal())
    shape = size(data)
    for n in 1:shape[1]
        x_n = sample(t, (:ic, n), Normal())
        for t_ix in 1:shape[2]
            x_n = observe(t, (:data, n, t_ix), Normal(loc + x_n, scale), data[n, t_ix])
        end
    end
end

# version for prediction and forecasting that returns sampled paths

function random_walk(t :: Trace, shape :: Tuple{Int64, Int64})
    loc = sample(t, :loc, Normal())
    scale = sample(t, :scale, LogNormal())
    out = Vector{Vector{Float64}}()
    for n in 1:shape[1]
        path = Float64[]
        x_n = sample(t, (:ic, n), Normal())
        push!(path, x_n)
        for t_ix in 1:shape[2]
            x_n = sample(t, (:data, n, t_ix), Normal(loc + x_n, scale))
            push!(path, x_n)
        end
        push!(out, path)
    end
    out
end

function ic_proposal_factory(data :: Array{Float64, 2})
    proposals = []
    n = size(data, 1)
    for k in 1:n
        push!(proposals, (t0, t1, data) -> propose(t1, (:ic, k), Normal(t0[(:ic, k)].value, 1.0)))
    end
    proposals
end
loc_proposal(t0, t1, data) = propose(t1, :loc, Normal(t0[:loc].value, 1.0))
scale_proposal(t0, t1, data) = propose(t1, :scale, truncated(Normal(t0[:scale].value, 1.0), 0.0, Inf))

function fake_data()
    @info "Generating fake data"
    n = 2
    t = 20
    t_pred = 10

    loc = 1.0
    scale = 1.5
    noise = randn(n, t + t_pred)
    data = cumsum(loc .+ scale .* noise, dims=2)
    test_data = data[:, t:end]
    data = data[:, 1:t]
    ic_proposals = ic_proposal_factory(data)
    proposals = append!([loc_proposal, scale_proposal], ic_proposals)
    
    @info "True loc = $loc, true scale = $scale"
    @time results = mh(random_walk, proposals; params=(data,), burn=250, thin=20, num_iterations=3000)
    @info "Estimated loc = ($(mean(results, :loc)) ± $(2.0 * std(results, :loc)))"
    @info "Estimated scale = ($(mean(results, :scale)) ± $(2.0 * std(results, :scale)))"

    n_forecast = 10
    posterior_model = update(random_walk, results)
    evidence = Dict((:ic, k) => data[k, end] for k in 1:n)
    forecast_model = condition(posterior_model, evidence)
    forecast = [forecast_model(trace(), (n, t_pred))[end][end] for _ in 1:n_forecast]

end

fake_data()