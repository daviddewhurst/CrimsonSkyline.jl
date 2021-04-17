function to_table(t :: Trace)
    pk = 1:length(t.trace)
    v = collect(values(t))
    address = [n.address for n in v]
    dist = [n.dist for n in v]
    value = [n.value for n in v]
    logprob = [n.logprob for n in v]
    logprob_sum = [n.logprob_sum for n in v]
    observed = [n.observed for n in v]
    interpretation = [n.interpretation for n in v]
    last_interpretation = [n.last_interpretation for n in v]
    table(
        (
            pk=pk,
            address=address,
            dist=dist,
            value=value,
            logprob=logprob,
            logprob_sum=logprob_sum,
            observed=observed,
            interpretation=interpretation,
            last_interpretation=last_interpretation
        );
        pkey=:pk
    )
end


function save(t :: Trace, f)
    trace_table = to_table(t)
    JuliaDB.save(trace_table, f)
    f
end

function load(f)
    trace_table = JuliaDB.load(f)
    t = trace()
    for ix in 1:length(trace_table)
        row = trace_table[ix]
        address = row.address
        t[address] = node(
            row.value,
            row.address,
            row.dist,
            row.observed,
            row.interpretation
        )
    end
    logprob!(t)
    t
end

export save, load, to_table