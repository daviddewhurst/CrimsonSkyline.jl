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

struct GraphIR 
    info :: OrderedDict{Any, OrderedDict}
    graph :: OrderedDict
end

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

struct Factor 
    info :: OrderedDict{Any, OrderedDict}
    node_to_factor :: OrderedDict
    factor_to_node :: OrderedDict
end

function factor(g :: GraphIR)
    collected_g = reverse(collect(g.info))
    factors = OrderedDict()
    nodes = OrderedDict(addr => Set() for addr in keys(g.info))
    i = 0
    for (addr, node) in collected_g
        f = Set([node["address"]])
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
factor(t :: Trace) = factor(graph_ir(t))


export node_info, GraphIR, graph_ir, Factor, factor