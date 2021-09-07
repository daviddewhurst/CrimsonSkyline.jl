import Pkg
Pkg.activate("..")

using CrimsonSkyline
using Distributions: Normal, LogNormal, Poisson

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
    t = trace()
    count_model(t, compiled_log_rate_model)
    @info "Compiled log rate model, recorded log rate node: $t"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end