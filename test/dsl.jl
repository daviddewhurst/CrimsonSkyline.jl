@model function simple_model(x::Vector{Float64})
    loc = sample(Normal())
    log_scale = sample(Normal())
    data = plate(observe, Normal(loc, exp(log_scale)), x)
end

@testset "model expansion 1" begin
    results = inference(
        simple_model, LikelihoodWeighting();
        params = (3.0 .+ randn(10),), inference_params = Dict("num_iterations" => 1000)
    )
    post_loc = mean(results, "loc")
    @info "Posterior location estimate = $post_loc"
end