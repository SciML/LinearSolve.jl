using LinearSolve, Test
using SparseArrays, LinearAlgebra

m, n = 13, 3

A = rand(m, n)
b = rand(m)
prob = LinearProblem(A, b)
res = A \ b
@test solve(prob).u ≈ res
@test solve(prob, QRFactorization()) ≈ res
@test solve(prob, KrylovJL_LSMR()) ≈ res

A = sprand(m, n, 0.5)
b = rand(m)
prob = LinearProblem(A, b)
res = A \ b
@test solve(prob).u ≈ res
@test solve(prob, QRFactorization()) ≈ res
@test solve(prob, KrylovJL_LSMR()) ≈ res

A = sprand(n, m, 0.5)
b = rand(n)
prob = LinearProblem(A, b)
res = Matrix(A) \ b
@test solve(prob, KrylovJL_CRAIGMR()) ≈ res

A = sprandn(1000, 100, 0.1)
b = randn(1001)
prob = LinearProblem(A, view(b, 1:1000))
linsolve = init(prob, QRFactorization())
solve(linsolve)

A = randn(1000, 100)
b = randn(1000)
@test isapprox(solve(LinearProblem(A, b)).u, Symmetric(A' * A) \ (A' * b))
solve(LinearProblem(A, b)).u;
solve(LinearProblem(A, b), (LinearSolve.NormalCholeskyFactorization())).u;
solve(LinearProblem(A, b), (LinearSolve.NormalBunchKaufmanFactorization())).u;
solve(LinearProblem(A, b), assumptions = (OperatorAssumptions(false; condition = OperatorCondition.WellConditioned))).u;

A = sprandn(5000, 100, 0.1)
b = randn(5000)
@test isapprox(solve(LinearProblem(A, b)).u, ldlt(A' * A) \ (A' * b))
solve(LinearProblem(A, b)).u;
solve(LinearProblem(A, b), (LinearSolve.NormalCholeskyFactorization())).u;
solve(LinearProblem(A, b), assumptions = (OperatorAssumptions(false; condition = OperatorCondition.WellConditioned))).u;