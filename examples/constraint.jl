import Pkg
Pkg.activate("..")

using CrimsonSkyline
using Distributions: Beta, Geometric
using Logging
using StatsBase: mean, std
using Random: seed!
using Plots

seed!(2021)

num_constraint(x, num, bound) = (x > num - bound) && (x < num + bound)

function constrained(t::Trace, evidence::Vector{Int64}, bound::Int64)
    prob = sample(t, "prob", Beta(2.0, 2.0))
    num = sample(t, "num", Geometric(prob))
    plate(t, observe, "constraint", SoftDelta(x -> num_constraint(x, num, bound), -2.0), evidence)
end

function unconstrained(t::Trace, evidence::Vector{Int64})
    prob = sample(t, "prob", Beta(2.0, 2.0))
    plate(t, observe, "num", Geometric(prob), evidence)
end

function main()
    data = [1, 5, 2, 9, 3, 2, 1, 3, 2, 1, 1, 4, 3, 2]
    bound = 3
    @time constrained_results = mh(constrained; params = (data, bound))
    @time unconstrained_results = mh(unconstrained; params = (data,))
    aic_constrained = aic(constrained_results)
    aic_unconstrained = aic(unconstrained_results)
    @info "AIC of constrained model = $aic_constrained"
    @info "AIC of unconstrained model = $aic_unconstrained"
    hpdi_con = hpdi(constrained_results, 0.5, ["prob"])
    hpdi_uncon = hpdi(unconstrained_results, 0.5, ["prob"])
    @info "50% HPDI of prob under constrained model: $hpdi_con"
    @info "50% HPDI of prob under unconstrained model: $hpdi_uncon"
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
