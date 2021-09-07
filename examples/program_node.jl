import Pkg
Pkg.activate("..")

using CrimsonSkyline
using Distributions: Normal, LogNormal, Poisson
using StatsBase: mean

function log_rate_model(t::Trace, ::Tuple)
    loc = sample(t, "loc", Normal())
    scale = sample(t, "scale", LogNormal())
    sample(t, "log_rate", Normal(loc, scale))
end

function count_model(t::Trace, log_rate::CompiledSF)
    log_rate_value = sample(t, "log_rate", log_rate)
    sample(t, "count", Poisson(exp(log_rate_value)))
end

function main()
    compiled_log_rate_model = compile(
        TypedSF(log_rate_model, Tuple, Float64),
        (),
        ["log_rate"]
    )
    conditioned_count_model = condition(
        count_model,
        Dict("count" => 30)
    )
    results = inference(
        conditioned_count_model, Metropolis();
        params = (compiled_log_rate_model,),
        inference_params = Dict("num_iterations" => 55000, "burn" => 5000, "thin" => 500)
    )
    @info "Posterior rate: $(exp(mean(results, "log_rate")))"

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end