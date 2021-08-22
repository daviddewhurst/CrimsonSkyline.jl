ss_cu_model(t) = sample(t, "a", Normal())
ss_du_model(t) = sample(t, "a", Poisson(3.0))
ss_cm_model(t) = sample(t, "a", MvNormal(3, 1.0))

function ms_model(t)
    log_rate = sample(t, "log_rate", Normal(2.0, 1.0))
    sample(t, "data", Poisson(exp(log_rate)))
end

@testset "single site" begin 
    @testset "unrestricted continuous kernel" begin
        t0 = trace()
        ss_cu_model(t0)
        k = make_kernel(ss_cu_model, t0["a"])

        t1 = trace()
        k(t0, t1)
        @info "Original trace: $t0"
        @info "New trace: $t1"
        @test mean(t1["a"].dist) == t0["a"].value
    end

    @testset "discrete univariate kernel" begin
        t0 = trace()
        ss_du_model(t0)
        k = make_kernel(ss_du_model, t0["a"])
        
        t1 = trace()
        k(t0, t1)
        @info "Original trace: $t0"
        @info "New trace: $t1"
        @test length(t1["a"].dist.support) == 3

        cond = condition(ss_du_model, Dict("a" => 0))
        t2 = trace()
        cond(t2)
        t3 = trace()
        k(t2, t3)
        @info "Original trace: $t2"
        @info "New trace: $t3"
        @test t3["a"].dist.support == [0, 1]
    end

    @testset "continuous multivariate kernel" begin
        t0 = trace()
        ss_cm_model(t0)
        k = make_kernel(ss_cm_model, t0["a"])

        t1 = trace()
        k(t0, t1)
        @info "Original trace: $t0"
        @info "New trace: $t1"
        @test mean(t1["a"].dist) == t0["a"].value
    end

    @testset "generate kernels" begin
        kernels = make_kernels(ms_model)
        t0 = trace()
        ms_model(t0)
        t1 = trace()
        map(k -> k(t0, t1), kernels)
        @info "Original trace: $t0"
        @info "New trace: $t1"
    end

end