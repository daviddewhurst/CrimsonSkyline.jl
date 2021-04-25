using Zygote

struct Parameter <: Interpretation end
const PARAMETER = Parameter()

@doc raw"""
    function node(value, address :: A) where A

Outer constructor for `Node` used as a parameter placeholder.
"""
function node(value, address :: A) where A
    T = typeof(value)
    Node{A, Parameter, T, Float64}(address, PARAMETER, value, 0.0, 0.0, false, Array{Node, 1}(), Array{Node, 1}(), PARAMETER, PARAMETER)
end

function sample(t :: Trace, a, v, i :: Parameter; pa = ())
    n = node(v, a)
    t[a] = n
    connect_pa_ch!(t, pa, a)
    v
end

parameter(t :: Trace, a, v; pa = ()) = sample(t, a, v, PARAMETER; pa = pa)

export Parameter, parameter