using LinearSolve, RecursiveFactorization, LinearAlgebra, SparseArrays, Test
using JET

A = rand(4, 4)
b = rand(4)
prob = LinearProblem(A, b)
JET.@test_opt init(prob)
@inferred init(prob)
@inferred solve(prob)
JET.@test_opt solve(prob)
JET.@test_opt solve(prob, LUFactorization())
JET.@test_opt solve(prob, GenericLUFactorization())
JET.@test_opt solve(prob, QRFactorization())
JET.@test_opt solve(prob, DiagonalFactorization())
#JET.@test_opt solve(prob, SVDFactorization())
#JET.@test_opt solve(prob, KrylovJL_GMRES())

prob = LinearProblem(sparse(A), b)
#JET.@test_opt solve(prob, UMFPACKFactorization())
#JET.@test_opt solve(prob, KLUFactorization())
#JET.@test_opt solve(prob, SparspakFactorization())
#JET.@test_opt solve(prob)