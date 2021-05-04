using Test
using Logging
using Distributions: Normal, Poisson, Gamma, LogNormal, Bernoulli, Geometric, MvNormal, truncated
using StatsBase: mean, std
using PrettyPrint: pprintln
using Random

using CrimsonSkyline

Random.seed!(2021)

include("basic.jl")
include("condition_and_observe.jl")
include("importance_sampling.jl")
include("graph_and_cpt.jl")
include("metropolis.jl")
include("nested.jl")