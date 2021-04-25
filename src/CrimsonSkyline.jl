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

include("util.jl")
include("fusion/distributions.jl")
include("fusion/tabulated.jl")

include("cpt.jl")
include("trace.jl")

include("graph.jl")

include("results.jl")
include("effects.jl")
include("importance.jl")
include("metropolis.jl")
include("vi.jl")

include("io.jl")

end # module
