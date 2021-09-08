import Pkg
Pkg.activate("..")

using CrimsonSkyline
using Distributions: Beta, Geometric, Binomial, truncated
using StatsBase: mean, std

function bayes_net(t::Trace, data::Int64)
    prob = sample(t, "prob", Beta(2.0, 2.0))
    n = observe(t, "n", truncated(Geometric(prob), 1, Inf), data)
    sample(t, "count", Binomial(n, prob))
end

function main()
    data = 7
    kernels = make_kernels(bayes_net; params = (data,))
    results = inference(
        bayes_net, Metropolis();
        params = (data,),
        inference_params = Dict(
            "kernels" => kernels,
            "num_iterations" => 55000,
            "burn" => 10000,
            "thin" => 500,
            "inverse_verbosity" => 5000
        )
    )
    prob_stats = map(f -> f(results, "prob"), [mean, std])
    count_stats = map(f -> f(results, "count"), [mean, std])
    @info "Posterior stats: prob (mean, std) = $prob_stats, count (mean, std) = $count_stats"

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end