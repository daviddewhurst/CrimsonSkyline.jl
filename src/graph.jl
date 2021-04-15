@doc raw"""
    function node_info(t :: Trace, a)

Returns an `OrderedDict` containing all information about the node `t[a]` required
for graph-based inference algorithms.

Keys include:
+ `"address"`: the address of the node in the trace
+ `"dist"`: the probability distribution associated with the address
+ `"observed"`: whether or not the value associated with the node was observed
+ `"interpretation"`: the interpretation of the node in the trace
+ `"data"` (optional): if the value associated with the node was observed, the value associated
    with the node. 
+ `"pa"`: the parents of the node in the trace. Note that it is necessary to pass parent addresses
    to one of the various sample methods in order to build a nontrivial `GraphIR`.
"""
function node_info(t :: Trace, a)
    n = t[a]
    info = OrderedDict(
        "address" => n.address,
        "dist" => n.dist,
        "observed" => n.observed,
        "interpretation" => n.interpretation
    )
    if n.observed
        info["data"] = n.value
    end
    ch = [that_n.address for that_n in n.ch]
    pa = [that_n.address for that_n in n.pa]
    info["pa"] = pa
    (info, ch)
end

@doc raw"""
    struct GraphIR 
        info :: OrderedDict{Any, OrderedDict}
        graph :: OrderedDict
    end

An intermediate representation of a single trace as a directed acyclic graph (DAG) that 
stores node information in `info` and DAG structure in `graph`. The keys and values of `graph`
are addresses, while `info` is a mapping from addresses to results from calls to `node_info`. 
"""
struct GraphIR 
    info :: OrderedDict{Any, OrderedDict}
    graph :: OrderedDict
end

@doc raw"""
    function graph_ir(t :: Trace)

Outer constructor for `GraphIR`
"""
function graph_ir(t :: Trace)
    g = OrderedDict()
    info = OrderedDict()
    for n in values(t)
        (node, children) = node_info(t, n.address)
        info[node["address"]] = node
        g[node["address"]] = children
    end
    GraphIR(info, g)
end

@doc raw"""
    struct Factor 
        info :: OrderedDict{Any, OrderedDict}
        node_to_factor :: OrderedDict
        factor_to_node :: OrderedDict
    end

A representation of a factor graph. 
"""
struct Factor 
    info :: OrderedDict{Any, OrderedDict}
    node_to_factor :: OrderedDict
    factor_to_node :: OrderedDict
end

@doc raw"""
    function factor(g :: GraphIR)

Outer constructor for a factor graph from an intermediate DAG representation. 
"""
function factor(g :: GraphIR)
    collected_g = reverse(collect(g.info))
    factors = OrderedDict()
    nodes = OrderedDict(addr => Set() for addr in keys(g.info))
    i = 0
    for (addr, node) in collected_g
        types = [typeof(p) for p in node["pa"]]
        push!(types, typeof(node["address"]))
        f = Set{Union{types...}}()
        union!(f, (node["address"],))
        union!(f, [p for p in node["pa"]])
        factors[i] = f
        i += 1
    end
    for (f, node_set) in factors
        for n in node_set
            union!(nodes[n], (f,))
        end
    end
    Factor(g.info, nodes, factors)
end
@doc raw"""
    factor(t :: Trace)

Outer constructor for a factor graph from a trace. 
"""
factor(t :: Trace) = factor(graph_ir(t))


export node_info, GraphIR, graph_ir, Factor, factor