function normal_model(t :: Trace, data :: Vector{Float64})
    loc = sample(t, :loc, Normal(0.0, 10.0))
    scale = sample(t, :scale, LogNormal())
    plate(t, observe, :obs, Normal(loc, scale), data)
end

function cat_model(t, dim, data)
    probs = sample(t, "probs", Dirichlet(ones(dim)))
    plate(t, observe, "data", Categorical(probs), data)
end

@testset "aic 1" begin
    data = [1, 2, 1, 3, 4, 3, 3, 3, 4, 2]
    dim = 4
    results = mh(cat_model; params = (dim, data), burn=1000, thin=100, num_iterations=21000)
    @time cat_aic = aic(results)
    @info "Computed AIC of cat model: $cat_aic"
    @info "Cat model has $(numparams(results.traces[1])) parameters"
end

@testset "hpds 1" begin
    data = randn(10) .* 2.0 .+ 4.0
    results = mh(normal_model; params = (data,), burn=1000, thin=100, num_iterations=21000)
    @time intervals = hpdi(results, 0.95, [:loc, :scale])
    @info "HPDI: $intervals"
    normal_aic = aic(results)
    @info "Computed AIC of normal model: $normal_aic"
    @info "Normal model has $(numparams(results.traces[1])) parameters"
end