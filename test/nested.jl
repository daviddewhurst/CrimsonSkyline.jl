include("../src/util.jl")

function model(t :: Trace, data)
    loc = sample(t, "loc", Normal(0.0, 10.0))
    scale = sample(t, "scale", Gamma(3.0, 3.0))
    for (i, d) in enumerate(data)
        observe(t, "data $i", Normal(loc, scale), d)
    end
end

@testset "nested sampling 1" begin
    true_loc = 8.0
    true_scale = 2.0
    data = rand(4) .* true_scale .+ true_loc
    @info "True loc = $true_loc"
    @info "True scale = $true_scale"

    for n in [3, 4, 8, 15, 25, 50]
        @info "Nested sampling inference with $n points"
        @time results = nested(model; params = (data,), num_points = n)
        mean_loc = mean(results, "loc")
        mean_std = mean(results, "scale")
        @info "Posterior mean loc = $mean_loc"
        @info "Posterior mean scale = $mean_std"
        log_Z = logsumexp(results.log_weights)
        @info "log p(x) ≈ $log_Z"
    end
end