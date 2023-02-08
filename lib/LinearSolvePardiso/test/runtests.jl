using LinearSolve, LinearSolvePardiso, SparseArrays, Random

A1 = sparse([
    1.0 0 -2 3
    0 5 1 2
    -2 1 4 -7
    3 2 -7 5
])
b1 = rand(4)
prob1 = LinearProblem(A1, b1)

lambda = 3
n = 4
e = ones(n)
e2 = ones(n - 1)
A2 = spdiagm(-1 => im * e2, 0 => lambda * e, 1 => -im * e2)
b2 = rand(n) + im * zeros(n)
cache_kwargs = (; verbose = true, abstol = 1e-8, reltol = 1e-8, maxiter = 30)

prob2 = LinearProblem(A2, b2)

for alg in (PardisoJL(), MKLPardisoFactorize(), MKLPardisoIterate())
    u = solve(prob1, alg; cache_kwargs...).u
    @test A1 * u ≈ b1

    u = solve(prob2, alg; cache_kwargs...).u
    @test eltype(u) <: Complex
    @test_broken A2 * u ≈ b2
end

Random.seed!(10)
A = sprand(n, n, 0.8);
A2 = 2.0 .* A;
b1 = rand(n);
b2 = rand(n);
prob = LinearProblem(copy(A), copy(b1))

prob = LinearProblem(copy(A), copy(b1))
linsolve = init(prob, UMFPACKFactorization())
sol11 = solve(linsolve)
linsolve = LinearSolve.set_b(sol11.cache, copy(b2))
sol12 = solve(linsolve)
linsolve = LinearSolve.set_A(sol12.cache, copy(A2))
sol13 = solve(linsolve)

linsolve = init(prob, MKLPardisoFactorize())
sol31 = solve(linsolve)
linsolve = LinearSolve.set_b(sol31.cache, copy(b2))
sol32 = solve(linsolve)
linsolve = LinearSolve.set_A(sol32.cache, copy(A2))
sol33 = solve(linsolve)

@test sol11.u ≈ sol31.u
@test sol12.u ≈ sol32.u
@test sol13.u ≈ sol33.u
