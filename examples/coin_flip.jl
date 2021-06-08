import Pkg
Pkg.activate("..")

using Distributions: Beta, Bernoulli
using CrimsonSkyline
using Random: seed!
using Logging
using StatsBase: mean

seed!(2021)

function coin_model(t::Trace, data::Vector{Bool})
    bias = sample(t, "bias", Beta(3.0, 3.0))
    coin = Bernoulli(bias)
    for (i, d) in enumerate(data)
        observe(t, "flip $i", coin, d)
    end
end

function main()
    @warn "~~~Coin flip problem~~~"
    data = [true, true, false, false, true, true, true, true]
    @info "Demonstrating trace structure"
    t = trace()
    coin_model(t, data[1:2])
    @info "Trace with data subset: $t"
    @info "Running default mh inference (no user tuning required)"
    @time inference_results = mh(coin_model; params = (data,))
    # posterior parameters:
    # alpha' = alpha + sum x_i = 3.0 + 6 = 9.0
    # beta' = beta + n - sum x_i = 3.0 + 8 - 6 = 5.0
    # posterior = Beta(alpha', beta') = Beta(9.0, 5.0)
    # posterior mean = alpha' / (alpha' + beta') = 0.6429
    true_posterior_mean = 0.6429
    @info "True posterior bias: $true_posterior_mean"
    @info "Mean posterior bias estimate: $(mean(inference_results, "bias"))"
    plot_marginal(inference_results, "bias", "plots/coin_flip", "bias.png")
end

main()