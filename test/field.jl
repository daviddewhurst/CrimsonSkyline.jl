factor_ab(x) = logpdf(MvNormal(PDMat([1.0 0.5; 0.5 1.0])), x)
factor_bc(x) = logpdf(MvNormal([2.0, 2.0], PDMat([2.0 -1.0; -1.0 2.0])), x)

struct FactorProposal
    addresses :: Vector{String}
    last :: Vector{String}
    std :: Float64
end
FactorProposal(addresses) = FactorProposal(addresses, [addresses[1]], 0.5)
function (fp::FactorProposal)(x::Dict)
    new = deepcopy(x)
    address = rand(fp.addresses)
    new[address] = randn() * fp.std + new[address]
    fp.last[1] = address
    new
end
function (fp::FactorProposal)(x_prime::Dict, x::Dict)
    address = fp.last[1]
    logpdf(Normal(x[address], fp.std), x_prime[address]) - log(length(fp.addresses))
end

#=
@testset "make field and sample" begin
    factors = Dict(["a", "b"] => factor_ab, ["b", "c"] => factor_bc)
    evidence = Dict("b" => 3.0)
    #evidence = Dict()
    field = RandomField(factors, evidence)
    addresses = ["a", "c"]
    proposal = FactorProposal(addresses)
    initial_values = Dict("a" => 0.0, "b" => 3.0, "c" => 0.0)
    samples = mh(field, [proposal], initial_values; burn=1000, thin=100, num_iterations=11000)
    for address in ["a", "b", "c"]
        plot_marginal(samples, address, "out", "factor-marginal-$address.png")
    end
end
=#

@testset "using generative field" begin
    factors = Dict(["a", "b"] => factor_ab, ["b", "c"] => factor_bc)
    field = RandomField(factors)
    addresses = ["a", "b", "c"]
    proposal = FactorProposal(addresses)
    init = Dict("a" => 0.0, "b" => 0.0, "c" => 0.0)
    gf = GenerativeField(field, proposal, init)

    # generative fields can be used in generic stochastic functions
    t = trace()
    sample(t, "field", gf)
    @info t

    # note that we can pass upstream samped values as evidence
    t = trace()
    upstream_value = sample(t, "a", Normal())
    field = RandomField(factors, Dict("a" => upstream_value))
    proposal = FactorProposal(["b", "c"])
    init = Dict("a" => 0.0, "b" => 0.0, "c" => 0.0)
    gf = GenerativeField(field, proposal, init)
    sample(t, "field", gf)
    @info t
end

