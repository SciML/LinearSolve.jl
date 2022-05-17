using LinearSolve, LinearSolvePardiso, SparseArrays

A1 = sparse([ 1. 0 -2  3
                0  5  1  2
            -2  1  4 -7
                3  2 -7  5 ])
b1 = rand(4)
prob1 = LinearProblem(A1, b1)

lambda = 3
e = ones(n)
e2 = ones(n-1)
A2 = spdiagm(-1 => im*e2, 0 => lambda*e, 1 => -im*e2)
b2 = rand(n) + im * zeros(n)

prob2 = LinearProblem(A2, b2)

for alg in (
            PardisoJL(),
            MKLPardisoFactorize(),
            MKLPardisoIterate(),
            )

    u = solve(prob1, alg; cache_kwargs...).u
    @test A1 * u ≈ b1

    u = solve(prob2, alg; cache_kwargs...).u
    @test eltype(u) <: Complex
    @test_broken A2 * u ≈ b2
end