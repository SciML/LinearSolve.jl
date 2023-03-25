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