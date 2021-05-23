@testset "forward sampling 1" begin
    results = forward_sampling(plate_program!; params = (rand(10),), num_iterations=100)
    @test results.interpretation == Forward()
    log_rate = results[:log_rate]
    @info "Mean log rate = $(mean(log_rate))"
end