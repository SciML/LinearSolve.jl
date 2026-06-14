using TrimTest

function (@main)(argv::Vector{String})::Cint
    x = parse(Float64, argv[1])
    sol = TrimTest.TestLUFactorization.solve_linear(x)
    println(Core.stdout, sum(sol.u))
    return 0
end
