function trace_info(t::Trace)
    v = collect(values(t))
    address = [n.address for n in v]
    dist = [n.dist for n in v]
    value = [n.value for n in v]
    logprob = [n.logprob for n in v]
    logprob_sum = [n.logprob_sum for n in v]
    observed = [n.observed for n in v]
    interpretation = [n.interpretation for n in v]
    last_interpretation = [n.last_interpretation for n in v]
    (
        address,
        dist,
        value,
        logprob,
        logprob_sum,
        observed,
        interpretation,
        last_interpretation
    )
end

@doc raw"""
    function to_table(t :: Trace)

Turns a trace into a juliadb table. Does not store parent / child relationships.
"""
function to_table(t :: Trace)
    pk = 1:length(t.trace)
    info = trace_info(t)
    table(
        (
            pk=pk,
            address=info[1],
            dist=info[2],
            value=info[3],
            logprob=info[4],
            logprob_sum=info[5],
            observed=info[6],
            interpretation=info[7],
            last_interpretation=info[8]
        );
        pkey=:pk
    )
end

@doc raw"""
    function save(t :: Trace, f)

Saves a trace to disk at the filepath `f`.
"""
function save(t :: Trace, f)
    if endswith(f, ".jdb")
        trace_table = to_table(t)
        JuliaDB.save(trace_table, f)
    else
        error(
            "File must end with .jdb for saving in JuliaDB format."
        )
    end
    f
end

function save(r :: SamplingResults, f)
    mkpath(f)
    pk = 1:length(r.traces)
    r_table = table(
        (pk = pk,
        log_weight = r.log_weights,
        return_value = r.return_values,
        trace = r.traces);
        pkey=:pk
    )
    JuliaDB.save(r_table, joinpath(f, "results.jdb"))
    open(joinpath(f, "metadata.txt"), "w") do file
        write(file, "Interpretation: $(string(r.interpretation))")
    end
end

@doc raw"""
    function load(f) :: Trace
        
Loads a serialized juliadb table from file `f` and converts it into a trace.
"""
function load_jdb(f) :: Trace
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

function load_csm(f) :: SamplingResults
    interpretation = open(joinpath(f, "metadata.txt"), "r") do f
        line = readline(f)
        interpretation_string = split(line, ": ")[end]
        eval(Meta.parse(interpretation_string))
    end
    results = SamplingResults{typeof(interpretation)}(interpretation, Float64[], [], Trace[])
    results_table = JuliaDB.load(joinpath(f, "results.jdb"))
    for ix in 1:length(results_table)
        row = results_table[ix]
        push!(results.log_weights, row.log_weight)
        push!(results.return_values, row.return_value)
        push!(results.traces, row.trace)
    end
    results
end

function load(f)
    if endswith(f, ".jdb")
        load_jdb(f)
    elseif endswith(f, ".csm")
        load_csm(f)
    else
        error(
            "Can load only .jdb (JuliaDB) files."
        )
    end
end

export save, load, to_table, to_dataframe