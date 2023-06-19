using LinearSolve, LinearAlgebra, SparseArrays, Test, JET
@test LinearSolve.defaultalg(nothing, zeros(3)).alg ===
      LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization
prob = LinearProblem(rand(3, 3), rand(3))
solve(prob)

@test LinearSolve.defaultalg(nothing, zeros(50)).alg ===
      LinearSolve.DefaultAlgorithmChoice.RFLUFactorization
prob = LinearProblem(rand(50, 50), rand(50))
solve(prob)

@test LinearSolve.defaultalg(nothing, zeros(600)).alg ===
      LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization
prob = LinearProblem(rand(600, 600), rand(600))
solve(prob)

@test LinearSolve.defaultalg(LinearAlgebra.Diagonal(zeros(5)), zeros(5)).alg ===
      LinearSolve.DefaultAlgorithmChoice.DiagonalFactorization

@test LinearSolve.defaultalg(nothing, zeros(5),
    LinearSolve.OperatorAssumptions(false)).alg ===
      LinearSolve.DefaultAlgorithmChoice.QRFactorization

@test LinearSolve.defaultalg(sprand(1000, 1000, 0.5), zeros(1000)).alg ===
      LinearSolve.DefaultAlgorithmChoice.KLUFactorization
prob = LinearProblem(sprand(1000, 1000, 0.5), zeros(1000))
solve(prob)

@test LinearSolve.defaultalg(sprand(11000, 11000, 0.001), zeros(11000)).alg ===
      LinearSolve.DefaultAlgorithmChoice.UMFPACKFactorization
prob = LinearProblem(sprand(11000, 11000, 0.5), zeros(11000))
solve(prob)

@static if VERSION >= v"v1.9-"
    # Test inference 
    A = rand(4, 4)
    b = rand(4)
    prob = LinearProblem(A, b)
    JET.@test_opt init(prob, nothing)
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
    @inferred solve(prob)
    @inferred init(prob, nothing)
end
