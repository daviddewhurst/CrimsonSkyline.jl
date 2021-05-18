function normal_model(t :: Trace, data :: Vector{Float64})
    loc = sample(t, :loc, Normal(0.0, 10.0))
    scale = sample(t, :scale, LogNormal())
    obs = Vector{Float64}(undef, length(data))
    for i in 1:length(data)
        obs[i] = observe(t, (:obs, i), Normal(loc, scale), data[i])
    end
    obs
end

@testset "parametric results 1" begin
    data = randn(2) .* 2.0 .+ 3.0
    results = mh(normal_model; params = (data,))
    nonparametric_updated = update(normal_model, results)
    r = nonparametric_updated(trace(), data)
    @info "Return value from nonparametric updated model: $r"

    p_results = to_parametric(results)
    parametric_updated = update(normal_model, p_results)
    r = parametric_updated(trace(), data)
    @info "Return value from parametric updated model: $r"
end

function modular_normal_model(t :: Trace, data :: Vector{Float64}, dists::Dict)
    loc = sample(t, "loc", dists["loc"])
    scale = sample(t, "scale", dists["scale"])
    obs = Vector{Float64}(undef, length(data))
    for i in 1:length(data)
        obs[i] = observe(t, (:obs, i), Normal(loc, scale), data[i])
    end
    obs
end

@testset "parametric results 2" begin
    loc = 2.0
    scale = 1.5
    n = 3
    @info "On step 0: n datapoints = $n, loc = $loc and scale = $scale"
    data = randn(n) .* scale .+ loc
    dists = Dict("loc" => Normal(0.0, 10.0), "scale" => LogNormal())

    results = mh(modular_normal_model; params = (data, dists))
    results = to_parametric(results)
    @info "On step 0, distributions are:\nloc = $(results.distributions["loc"])\nscale = $(results.distributions["scale"])\n"
    reflate!(results, Dict("loc" => 10.0, "scale" => 2.0))

    niter = 50
    for i in 1:niter
        n = Int(ceil(exp(randn() * 2))) + 5
        loc += randn()
        scale *= exp(randn() * 0.1)
        @info "On step $i: n datapoints = $n, loc = $loc and scale = $scale"
        data = randn(n) .* scale .+ loc
        dists = results.distributions

        results = mh(modular_normal_model; params = (data, dists))
        results = to_parametric(results)
        @info "On step $i, distributions are:\nloc = $(results.distributions["loc"])\nscale = $(results.distributions["scale"])\n"
        reflate!(results, Dict("loc" => 10.0, "scale" => 2.0))
    end

end