factor_ab(x) = logpdf(MvNormal(PDMat([1.0 0.5; 0.5 1.0])), x)

function factor_bc(x)
    (b, c) = x
    if c == 1
        log(0.25) + logpdf(Normal(0.5, 0.5), b)
    elseif c == -1
        log(0.75) + logpdf(Normal(-0.5, 2.0), b)
    else
        0.0
    end
end

# this is a really bad proposal kernel
function factor_abc_proposal(x::Dict)
    new = Dict()
    new["a"] = randn() * 0.2 + x["a"]
    new["b"] = randn() * 0.2 + x["b"]
    new["c"] = rand([-1, 1])
    new
end
function factor_abc_proposal(x_prime::Dict, x::Dict)
    lp = 0.0
    lp += logpdf(Normal(x["a"], 0.2), x_prime["a"])
    lp += logpdf(Normal(x["b"], 0.2), x_prime["b"])
    lp += logpdf(DiscreteNonParametric([-1, 1], [0.5, 0.5]), x_prime["c"])
    lp
end

@testset "make field and manually step" begin

    factors = Dict(["a", "b"] => factor_ab, ["b", "c"] => factor_bc)
    evidence = Dict("b" => 3.0)
    field = RandomField(factors, evidence)

    initial_values = Dict("a" => 0.0, "b" => 0.0, "c" => 1)
    log_prob_initial = field(initial_values)
    @info "With initial values $initial_values, log_prob = $log_prob_initial"

    (new_val, new_log_prob) = mh_step(field, factor_abc_proposal, initial_values, log_prob_initial)
    @info "New val = $new_val with log_prob = $new_log_prob"
end