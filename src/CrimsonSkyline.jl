"""
Author: David Rushing Dewhurst

Copyright: David Rushing Dewhurst and Charles River Analytics Inc., 2021 - present

Released under the MIT license.
"""

module CrimsonSkyline

using DataFrames
using DataStructures
using Distributions
using PrettyPrint: pprintln
using StatsBase
using Logging
using JuliaDB
using JSON
using Plots
using PDMats
using SQLite

include("modeling/util.jl")

include("modeling/trace.jl")
include("modeling/distributions.jl")

include("representation/results.jl")

include("modeling/field.jl")

include("representation/statistics.jl")

include("modeling/effects.jl")

include("inference/forward.jl")
include("inference/rejection.jl")
include("inference/importance.jl")
include("inference/metropolis.jl")
include("inference/nested.jl")

include("representation/io.jl")
include("representation/plot.jl")

include("inference/kernel.jl")
include("inference/inference.jl")
include("representation/db.jl")

end # module
