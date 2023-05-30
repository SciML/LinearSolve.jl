using LinearSolve, LinearAlgebra, SparseArrays, Test
@test LinearSolve.defaultalg(nothing, zeros(3)).alg === DefaultAlgorithmChoice.GenericLUFactorization
@test LinearSolve.defaultalg(nothing, zeros(50)).alg === DefaultAlgorithmChoice.RFLUFactorization
@test LinearSolve.defaultalg(nothing, zeros(600)).alg === DefaultAlgorithmChoice.LUFactorization
@test LinearSolve.defaultalg(LinearAlgebra.Diagonal(zeros(5)), zeros(5)).alg === DefaultAlgorithmChoice.DiagonalFactorization

@test LinearSolve.defaultalg(nothing, zeros(5),
                             LinearSolve.OperatorAssumptions(false)).alg === DefaultAlgorithmChoice.QRFactorization

@test LinearSolve.defaultalg(sprand(1000, 1000, 0.01), zeros(1000)).alg === DefaultAlgorithmChoice.KLUFactorization
@test LinearSolve.defaultalg(sprand(11000, 11000, 0.001), zeros(11000)).alg === DefaultAlgorithmChoice.UMFPACKFactorization

# Test inference 
A = rand(4, 4)
b = rand(4)
prob = LinearProblem(A, b)
@inferred solve(prob)
@inferred init(prob, nothing)