import Pkg
Pkg.activate("..")

using Distributions: Exponential, DiscreteUniform, Poisson, Normal, DiscreteNonParametric, truncated
using CrimsonSkyline
using Random: seed!
using Logging
using StatsBase: mean

seed!(2021)

# source: http://people.reed.edu/~jones/141/Coal.html
const coal_mining_data = [
    4, 5, 4, 1, 0, 4, 3, 4, 0, 6, 3, 3, 
    4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5, 2,
    2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0, 1, 0, 1, 1, 0, 0, 
    3, 1, 0, 3, 2, 2, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 
    0, 1, 1, 0, 2, 3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 0, 1, 
    4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1
]

function changepoint_model(t, data)
    rate_1 = sample(t, "rate_1", Exponential(5.0))
    rate_2 = sample(t, "rate_2", Exponential(5.0))
    changepoint = sample(t, "changepoint", DiscreteUniform(2, length(data) - 1))
    for (i, d) in enumerate(data)
        i < changepoint ? observe(t, "data $i", Poisson(rate_1), d) : observe(t, "data $i", Poisson(rate_2), d)
    end
end

rate_1_proposal(t0, t1, params...) = propose(t1, "rate_1", truncated(Normal(t0["rate_1"].value, 0.5), 0.01, Inf))
rate_2_proposal(t0, t1, params...) = propose(t1, "rate_2", truncated(Normal(t0["rate_2"].value, 0.5), 0.01, Inf))
function cp_proposal(t0, t1, data)
    ld = length(data) - 1
    pt = t0["changepoint"].value
    if pt == 2
        propose(t1, "changepoint", DiscreteNonParametric([2, 3], [0.5, 0.5]))
    elseif pt == ld
        propose(t1, "changepoint", DiscreteNonParametric([ld - 1, ld], [0.5, 0.5]))
    else
        propose(t1, "changepoint", DiscreteNonParametric([pt - 1, pt, pt + 1], [1/3.0, 1/3.0, 1/3.0]))
    end
end

function main()
    @time results = mh(
        changepoint_model, [rate_1_proposal, rate_2_proposal, cp_proposal];
        params = (coal_mining_data,), burn = 1000, thin = 100, num_iterations = 11000, inverse_verbosity = 1000
    )
    ests = hpdi(results, 0.9, ["rate_1", "rate_2", "changepoint"])
    @info "90% posterior HPDIs: $ests"
    plot_marginal(results, "changepoint", "plots/changepoint", "changepoint.png")
end

main()