"""
Author: David Rushing Dewhurst

Copyright: David Rushing Dewhurst and Charles River Analytics Inc., 2021 - present

Released under the GNU GPL v3 license.
"""

module CrimsonSkyline

using DataStructures
using Distributions
using PrettyPrint: pprintln
using StatsBase
using Logging
using JuliaDB
using JSON
using PDMats

include("util.jl")
include("trace.jl")

include("results.jl")
include("statistics.jl")

include("effects.jl")

include("forward.jl")
include("rejection.jl")
include("importance.jl")
include("metropolis.jl")
include("nested.jl")

include("io.jl")

end # module
