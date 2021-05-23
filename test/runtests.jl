using Test
using Logging
using Distributions: Normal, Poisson, Gamma, LogNormal, Bernoulli, Geometric, MvNormal, truncated, Dirichlet, Categorical, logpdf
using StatsBase: mean, std
using PrettyPrint: pprintln
using Random
using PDMats

using CrimsonSkyline

Random.seed!(2021)

include("basic.jl")
include("condition_and_observe.jl")
include("io.jl")
include("forward.jl")
include("importance_sampling.jl")
include("metropolis.jl")
include("nested.jl")
include("results.jl")
include("typed_trace.jl")