function io_model(t::Trace, dim::Int64, data::Int64)
    z = sample(t, "z", Dirichlet(ones(dim)))
    observe(t, "x", Categorical(z), data)
end

@testset "io 1" begin
    t = trace()
    dim = 10
    data = 7
    io_model(t, dim, data)
    testpath = joinpath(@__DIR__, "TESTIO")
    t_table = to_table(t)
    @info "Created JuliaDB table: $t_table"

    mkpath(testpath)
    db_file = joinpath(testpath, "test.jdb")
    save(t, db_file)
    loaded_trace = load(db_file)
    @info "z node: $(loaded_trace["z"])"
    @info "x node: $(loaded_trace["x"])"
    @test loaded_trace["x"].address == "x"
    @test typeof(loaded_trace["x"].dist) == Categorical{Float64, Vector{Float64}}

    # test save / load of results
    results = mh(io_model; params = (dim, data), burn = 0, thin = 1, num_iterations=10)
    results_file = joinpath(testpath, "io_model.csm")
    save(results, results_file)
    loaded_results = load(joinpath(testpath, "io_model.csm"))

    # use saved results for prediction
    updated_model = update(io_model, loaded_results)
    new_t, _ = updated_model(trace(), dim, data)
    @info "Trace created using saved results: $new_t"
    @test new_t["z"].interpretation == EMPIRICAL
    rm(testpath, recursive=true)
end