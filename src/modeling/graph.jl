abstract type PGM{G} end

struct NodeInfo{A,D,I}
    address::A
    dist::D
    interpretation::I
end
abstract type DirectedPGM{G} <: PGM{G} end
struct TraceConvertedPGM{G<:SimpleDiGraph} <: DirectedPGM{G}
    graph::G
    info::Dict
    mapping::OrderedDict
end

function pgm(t::Trace)
    node_card = 0
    graph_dict = Dict()
    node_info = Dict()
    mapping = OrderedDict()
    for (address, node) in t.trace
        graph_dict[address] = node.ch
        node_info[address] = NodeInfo(address, node.dist, node.interpretation)
        node_card += 1
        mapping[address] = node_card
    end
    graph = SimpleDiGraph(node_card)
    for (pa, arr_of_ch) in graph_dict
        for ch in arr_of_ch
            add_edge!(graph, mapping[pa], mapping[ch.address])
        end
    end
    TraceConvertedPGM(graph, node_info, mapping)
end

function pgm(f; params = ())
    t = trace()
    f(t, params...)
    pgm(t)
end

struct DiscretePGM{G<:SimpleDiGraph} <: DirectedPGM{G}
    graph::G
    info::Dict
    mapping::OrderedDict
end

is_multivariate(d) = (typeof(d) <: ContinuousMultivariateDistribution) || (typeof(d) <: DiscreteMultivariateDistribution)
is_discrete(d) = (typeof(d) <: DiscreteUnivariateDistribution) || (typeof(d) <: DiscreteMultivariateDistribution)
function is_bounded(d)
    ex = extrema(d)
    !(ex[1] == -Inf || ex[2] == Inf)
end

function to_table(n, values, probs)
    probs = probs ./ sum(probs)
    table = DataFrame()
    table[Symbol(n.address)] = values
    table[:prob] = probs
    table
end

function discretize(n, quantiles, d::D) where D <: DiscreteUnivariateDistribution
    if typeof(d) <: Truncated
        values = collect(support(d))
        probs = pdf.(d, values)
    else
        try
            values = collect(support(d))
            probs = pdf.(d, values)
        catch e
            if isa(e, Inexacterror)
                values = quantile.(d, quantiles)
                unique!(values)
                probs = pdf.(d, values)
            else
                throw(e)
            end
        end
    end
    to_table(n, values, probs)
end

function discretize(n, quantiles, d::D) where D <: ContinuousUnivariateDistribution
    values = quantile.(n.dist, quantiles)
    probs = pdf.(n.dist, values)
    to_table(n, values, probs)
end

function discretize(n, quantiles::Vector{Float64})
    is_multivariate(n.dist) && error("Can only discretize univariate distributions.")
    discretize(n, quantiles, n.dist)
end

function parent_values(tables, parents)
    pv = OrderedDict()
    for p in parents
        pv[p] = unique(tables[p][Symbol(p)])
    end
    pv
end

### for both -- product of all levels from parents with levels from self
### for each parent combo, dispach to underlying "no parent" implementation with modified dist (parent params)

function discretize(tables, n, quantiles, parents, d::D) where D <: DiscreteUnivariateDistribution
    pv = parent_values(tables, parents)
    parameters = product(values(pv)...)
    
end

function discretize(tables, n, quantiles, parents, d::D) where D <: ContinuousUnivariateDistribution
    pv = parent_values(tables, parents)
end

###
###

function discretize(tables::Dict, n, quantiles::Vector{Float64}, parents)
    is_multivariate(n.dist) && error("Can only discretize univariate distributions.")
    discretize(tables, n, quantiles, parents, n.dist)
end

function discretize(g::TraceConvertedPGM, quantiles::Vector{Float64})
    tables = Dict()
    for address in keys(g.mapping)
        parents = inneighbors(g.graph, g.mapping[address])
        table = length(parents) == 0 ? discretize(g.info[address], quantiles) : discretize(tables, g.info[address], quantiles, parents)
        tables[addres] = table
    end
end

discretize(g::TraceConvertedPGM) = discretize(g, collect(0.1:0.1:0.9))
discretize(f; params = ()) = discretize(pgm(f; params = params))

abstract type UndirectedPGM{G} <: PGM{G} end
struct FactorGraph{G<:SimpleGraph} <: UndirectedPGM{G}
    graph::G
    factors::Dict
    mapping::OrderedDict
end

export pgm, discretize