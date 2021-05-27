function eye(D)
    mat = zeros(D, D)
    for i in 1:D
        mat[i, i] = 1.0
    end
    mat
end

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
    plate(t, observe, :obs, Normal(loc, scale), data)
end

@testset "parametric results 2" begin
    loc = 2.0
    scale = 1.5
    n = 3
    @info "On step 0: n datapoints = $n, loc = $loc and scale = $scale"
    data = randn(n) .* scale .+ loc
    dists = Dict("loc" => Normal(0.0, 10.0), "scale" => LogNormal())

    @time results = mh(modular_normal_model; params = (data, dists))
    results = to_parametric(results)
    @info "On step 0, distributions are:\nloc = $(results.distributions["loc"])\nscale = $(results.distributions["scale"])\n"
    reflate!(results, Dict("loc" => 10.0, "scale" => 2.0))

    niter = 25
    for i in 1:niter
        n = Int(ceil(exp(randn() * 2))) + 5
        loc += randn()
        scale *= exp(randn() * 0.1)
        @info "On step $i: n datapoints = $n, loc = $loc and scale = $scale"
        data = randn(n) .* scale .+ loc
        dists = results.distributions

        @time results = mh(modular_normal_model; params = (data, dists))
        results = to_parametric(results)
        @info "On step $i, distributions are:\nloc = $(results.distributions["loc"])\nscale = $(results.distributions["scale"])\n"
        reflate!(results, Dict("loc" => 10.0, "scale" => 2.0))
    end
    
end

function modular_mv_normal_model(t :: Trace, X :: Matrix{Float64}, y::Vector{T}, dists::Dict) where T
    beta = sample(t, "beta", dists["beta"])
    loc = X' * beta
    scale = sample(t, "scale", dists["scale"])
    obs = Vector{Float64}(undef, length(y))
    for (i, yi) in enumerate(y)
        obs[i] = observe(t, (:obs, i), Normal(loc[i], scale), yi)
    end
    obs
end

@testset "parametric results 3" begin
    d = 10
    n = 5
    X = randn(d, n)
    beta = randn(d)
    sigma = 0.25
    y = X' * beta .+ randn(n) .* sigma
    dists = Dict("beta" => MvNormal(d, 1.0), "scale"=>LogNormal())
    @time results = mh(modular_mv_normal_model; params = (X, y, dists))
    results = to_parametric(results)
    @info "Distributions are:\nbeta = $(results.distributions["beta"])\nscale = $(results.distributions["scale"])\n"
    reflate!(results, Dict("beta" => PDMat(2.0 .* eye(d)), "scale" => 2.0))
    @info "After reflation, distributions are:\nbeta = $(results.distributions["beta"])\nscale = $(results.distributions["scale"])\n"
end

@testset "parametric results 4: merging updates" begin
    loc = 2.0
    scale = 1.5
    n = 3
    @info "n datapoints = $n, loc = $loc and scale = $scale"
    data = randn(n) .* scale .+ loc
    @info "First dataset: $data"
    data2 = randn(n) .* scale .+ (loc + 1.0)
    @info "Second dataset: $data2"
    dists = Dict("loc" => Normal(0.0, 10.0), "scale" => LogNormal())

    # doing this twice, simulating two remote workers
    # first time
    @time results = mh(modular_normal_model; params = (data, dists))
    results = to_parametric(results)
    @info "From location 1, distributions are:\nloc = $(results.distributions["loc"])\nscale = $(results.distributions["scale"])\n"
    reflate!(results, Dict("loc" => 10.0, "scale" => 2.0))

    # second time
    @time results2 = mh(modular_normal_model; params = (data2, dists))
    results2 = to_parametric(results2)
    @info "From location 2, distributions are:\nloc = $(results2.distributions["loc"])\nscale = $(results2.distributions["scale"])\n"
    reflate!(results2, Dict("loc" => 10.0, "scale" => 2.0))

    # now, combine results -- this would happen at e.g. a master location
    # with results collected in a consumer queue
    relative_weights = [1.0, 1.0]
    collected_results = [results, results2]
    @time new_dists = combine(collected_results, relative_weights)
    @info "New dists: $new_dists"

    # using the updated prior for future predictions, reserving, etc
    # the new dists could be distributed to each remote location
    data3 = randn(5) .* scale .+ -1.5
    @info "Third dataset: $data3"
    @time results_3 = mh(modular_normal_model; params = (data3, new_dists))
    results_3 = to_parametric(results_3)
    @info "Using combined priors and after inference, updated distributions are:\nloc = $(results_3.distributions["loc"])\nscale = $(results_3.distributions["scale"])\n"
end

@testset "distribution representation 1" begin
    loc = 2.0
    scale = 1.5
    n = 3
    @info "n datapoints = $n, loc = $loc and scale = $scale"
    data = randn(n) .* scale .+ loc
    @info "First dataset: $data"
    dists = Dict("loc" => Normal(0.0, 10.0), "scale" => LogNormal())
    @time results = mh(modular_normal_model; params = (data, dists))
    results = to_parametric(results)
    @info "From location 1, distributions are:\nloc = $(results.distributions["loc"])\nscale = $(results.distributions["scale"])\n"
    compressed = to_json(results)
    @info "Compressed results = $compressed"
    @test typeof(compressed) == String

    # now run inference with loaded results as prior
    built = from_json(compressed)
    @info "Built parametric results: $built"
    reflate!(built, Dict("loc" => 10.0, "scale" => 2.0))
    data2 = randn(5) .* scale .+ -1.5
    @info "Second dataset: $data2"
    @time results_2 = mh(modular_normal_model; params = (data2, built.distributions))
    results_2 = to_parametric(results_2)
    @info "Using priors loaded from json and after inference, updated distributions are:\nloc = $(results_2.distributions["loc"])\nscale = $(results_2.distributions["scale"])\n"
end

@testset "distribution representation 2" begin
    d = 10
    n = 5
    X = randn(d, n)
    beta = randn(d)
    sigma = 0.25
    y = X' * beta .+ randn(n) .* sigma
    dists = Dict("beta" => MvNormal(d, 1.0), "scale"=>LogNormal())
    @time results = mh(modular_mv_normal_model; params = (X, y, dists))
    results = to_parametric(results)
    compressed = to_json(results)
    @info "Compressed results = $compressed"
    @test typeof(compressed) == String
    built = from_json(compressed)
    @info "Built parametric results: $built"
end

function beta_binomial(t, n, data, dists)
    b = sample(t, "b", dists["b"])
    plate(t, sample, "data", Binomial(n, b), data)
end

function branching_structure(t, data, dists)
    b = sample(t, "b", dists["b"])
    if b > 0.5
        loc = sample(t, "loc", dists["loc"])
        scale = sample(t, "scale", dists["scale"])
        plate(t, sample, "data", Normal(loc, scale), data)
    else
        lambda = sample(t, "rate", dists["rate"])
        data = convert(Vector{Int64}, data)
        plate(t, sample, "data", Poisson(lambda), data)
    end
end

@testset "distribution representation 3" begin
    @warn "Testing beta-binomial parametric posterior"
    data = [1, 3, 1, 5, 6, 3, 2, 0]
    n = 7
    dists = Dict("b" => Beta(2.0, 3.0))
    @time results = mh(beta_binomial; params = (n, data, dists))
    results = to_parametric(results)
    @info results.distributions

    @warn "Testing branching structure model"
    data = [10.0, 20.0, 10.0, 30.0, 50.0, 20.0]
    dists = Dict(
        "b" => Beta(1.0, 1.0), 
        "loc" => Normal(mean(data), std(data)),
        "scale" => LogNormal(1.0, 1.0),
        "rate" => Gamma(2.0, 2.0)
    )
    @time results = likelihood_weighting(branching_structure, data, dists; nsamples=10)
    @info results
    results = to_parametric(results)
    @info results.distributions
end