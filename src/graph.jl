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
    if n.observed || n.interpretation == CONDITIONED
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

function get_pa_values(d, sampled :: OrderedDict)
    pa = d["pa"]
    l_pa = length(pa)
    local values
    if l_pa == 0
        values = nothing
    elseif l_pa == 1
        values = get(sampled, pa[1], nothing)
    else  # parsed from left to right
        values = [get(sampled, pa[i], nothing) for i in 1:l_pa]
    end
    values
end

function get_dist(d, sampled :: OrderedDict)
    values = get_pa_values(d, sampled)
    values === nothing ? d["dist"] : typeof(d["dist"])(values...)
end

function sample(d :: D, sampled :: OrderedDict, i :: I) where {D <: AbstractDict, I <: Interpretation}
    dist = get_dist(d, sampled)
    value = rand(dist)
    lp = logpdf(dist, value)
    (value, lp)
end

function sample(d :: D, sampled :: OrderedDict, i :: Union{Standard,Conditioned}) where {D <: AbstractDict}
    dist = get_dist(d, sampled)
    lp = logpdf(dist, d["data"])
    (d["data"], lp)
end

function sample(d :: D, sampled :: OrderedDict, i :: Deterministic) where {D <: AbstractDict}
    values = get_pa_values(d, sampled)
    local value
    if values === nothing
        value = d["dist"]()
    else
        value = d["dist"](values...)
    end
    (value, 0.0)  # deterministic transsform doesn't affect log prob
end

sample(d :: D, sampled :: OrderedDict, i :: Input) where {D <: AbstractDict} = (d["data"], 0.0)

@doc raw"""
    function sample(g :: GraphIR)

Sample from the Bayes net implicitly defined by `g`. Returns a tuple `(s, lp)`, 
where `s` is an `OrderedDict` of `address => value` and `lp` is an `OrderedDict` of
`address => log probability`.
"""
function sample(g :: GraphIR)
    sampled = OrderedDict()
    log_probs = OrderedDict()
    for address in keys(g.graph)
        node = g.info[address]
        (this_s, this_lp) = sample(node, sampled, node["interpretation"])
        sampled[address] = this_s
        log_probs[address] = this_lp
    end
    (sampled, log_probs)
end

function leaf_vars(g :: GraphIR)
    v = []
    for (a, n) in g.info
        if n["interpretation"] != INPUT && length(n["pa"]) == 0
            push!(v, a)
        end
    end
    v
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
export sample