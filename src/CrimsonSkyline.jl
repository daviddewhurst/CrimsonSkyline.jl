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

include("util.jl")

include("trace.jl")
include("distributions.jl")

include("results.jl")

include("field.jl")

include("statistics.jl")

include("effects.jl")

include("forward.jl")
include("rejection.jl")
include("importance.jl")
include("metropolis.jl")
include("nested.jl")

include("io.jl")
include("plot.jl")

include("db.jl")

end # module
