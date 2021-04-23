import Pkg
Pkg.activate("..")

using CrimsonSkyline
using Distributions: Normal, Gamma
using Logging
using StatsBase: mean, std
using Random: seed!

seed!(2021)

function random_walk(t :: Trace, data :: Array{Float64, 2})
    loc = sample(t, :loc, Normal())
    scale = sample(t, :scale, Gamma(3.0, 5.0))
    shape = size(data)
    for n in 1:shape[1]
        x_n = sample(t, (:ic, n), Normal())
        for t_ix in 1:shape[2]
            x_n = observe(t, (:data, n, t_ix), Normal(loc + x_n, scale), data[n, t_ix])
        end
    end
end

function main()
    n = 3
    t = 20
    loc = 2.0
    scale = 1.5
    noise = rand(n, t)
    data = cumsum(loc .+ scale .* noise, dims=2)

    @info "True loc = $loc, true scale = $scale"
    results = mh(random_walk; params=(data,), burn=1000, thin=50, num_iterations=50000)
    @info "Estimated 95% CI loc = ($(mean(results, :loc)) ± $(2.0 * std(results, :loc)))"
    @info "Estimated 95% CI scale = ($(mean(results, :scale)) ± $(2.0 * std(results, :scale)))"
end

main()