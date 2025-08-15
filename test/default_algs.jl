using LinearSolve, RecursiveFactorization, LinearAlgebra, SparseArrays, Test

@test LinearSolve.defaultalg(nothing, zeros(3)).alg === LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization
prob = LinearProblem(rand(3, 3), rand(3))
solve(prob)

if LinearSolve.appleaccelerate_isavailable()
    @test LinearSolve.defaultalg(nothing, zeros(50)).alg ===
          LinearSolve.DefaultAlgorithmChoice.AppleAccelerateLUFactorization
else
    @test LinearSolve.defaultalg(nothing, zeros(50)).alg ===
          LinearSolve.DefaultAlgorithmChoice.RFLUFactorization
end
prob = LinearProblem(rand(50, 50), rand(50))
solve(prob)

if LinearSolve.usemkl
    @test LinearSolve.defaultalg(nothing, zeros(600)).alg ===
          LinearSolve.DefaultAlgorithmChoice.MKLLUFactorization
elseif LinearSolve.appleaccelerate_isavailable()
    @test LinearSolve.defaultalg(nothing, zeros(600)).alg ===
          LinearSolve.DefaultAlgorithmChoice.AppleAccelerateLUFactorization
else
    @test LinearSolve.defaultalg(nothing, zeros(600)).alg ===
          LinearSolve.DefaultAlgorithmChoice.LUFactorization
end

prob = LinearProblem(rand(600, 600), rand(600))
solve(prob)

@test LinearSolve.defaultalg(LinearAlgebra.Diagonal(zeros(5)), zeros(5)).alg ===
      LinearSolve.DefaultAlgorithmChoice.DiagonalFactorization

@test LinearSolve.defaultalg(nothing, zeros(5),
    LinearSolve.OperatorAssumptions(false)).alg ===
      LinearSolve.DefaultAlgorithmChoice.QRFactorization

A = spzeros(2, 2)
# test that solving a singular problem doesn't error
prob = LinearProblem(A, ones(2))
@test solve(prob, UMFPACKFactorization()).retcode == ReturnCode.Infeasible
@test solve(prob, KLUFactorization()).retcode == ReturnCode.Infeasible

@test LinearSolve.defaultalg(sprand(10^4, 10^4, 1e-5) + I, zeros(1000)).alg ===
      LinearSolve.DefaultAlgorithmChoice.KLUFactorization
prob = LinearProblem(sprand(1000, 1000, 0.5), zeros(1000))
solve(prob)

@test LinearSolve.defaultalg(sprand(11000, 11000, 0.001), zeros(11000)).alg ===
      LinearSolve.DefaultAlgorithmChoice.UMFPACKFactorization
prob = LinearProblem(sprand(11000, 11000, 0.5), zeros(11000))
solve(prob)

# Test inference
A = rand(4, 4)
b = rand(4)
prob = LinearProblem(A, b)
prob = LinearProblem(sparse(A), b)
@inferred solve(prob)
@inferred init(prob, nothing)

prob = LinearProblem(big.(rand(10, 10)), big.(zeros(10)))
solve(prob)

## Operator defaults
## https://github.com/SciML/LinearSolve.jl/issues/414

m, n = 2, 2
A = rand(m, n)
b = rand(m)
x = rand(n)
f = (w, v, u, p, t) -> mul!(w, A, v)
fadj = (w, v, u, p, t) -> mul!(w, A', v)
funcop = FunctionOperator(f, x, b; op_adjoint = fadj)
prob = LinearProblem(funcop, b)
sol1 = solve(prob)
sol2 = solve(prob, LinearSolve.KrylovJL_GMRES())
@test sol1.u == sol2.u

m, n = 3, 2
A = rand(m, n)
b = rand(m)
x = rand(n)
f = (w, v, u, p, t) -> mul!(w, A, v)
fadj = (w, v, u, p, t) -> mul!(w, A', v)
funcop = FunctionOperator(f, x, b; op_adjoint = fadj)
prob = LinearProblem(funcop, b)
sol1 = solve(prob)
sol2 = solve(prob, LinearSolve.KrylovJL_LSMR())
@test sol1.u == sol2.u

m, n = 2, 3
A = rand(m, n)
b = rand(m)
x = rand(n)
f = (w, v, u, p, t) -> mul!(w, A, v)
fadj = (w, v, u, p, t) -> mul!(w, A', v)
funcop = FunctionOperator(f, x, b; op_adjoint = fadj)
prob = LinearProblem(funcop, b)
sol1 = solve(prob)
sol2 = solve(prob, LinearSolve.KrylovJL_CRAIGMR())
@test sol1.u == sol2.u

# Default for Underdetermined problem but the size is a long rectangle
A = [2.0 1.0
     0.0 0.0
     0.0 0.0]
b = [1.0, 0.0, 0.0]
prob = LinearProblem(A, b)
sol = solve(prob)

@test !SciMLBase.successful_retcode(sol.retcode)

## Show that we cannot select a default alg once by checking the rank, since it might change
## later in the cache
## Common occurrence for iterative nonlinear solvers using linear solve
A = [2.0 1.0
     1.0 1.0
     0.0 0.0]
b = [1.0, 1.0, 0.0]
prob = LinearProblem(A, b)

cache = init(prob)
sol = solve!(cache)

@test sol.u ≈ [0.0, 1.0]

cache.A = [2.0 1.0
           0.0 0.0
           0.0 0.0]

sol = solve!(cache)

@test !SciMLBase.successful_retcode(sol.retcode)

## Non-square Sparse Defaults 
# https://github.com/SciML/NonlinearSolve.jl/issues/599
A = SparseMatrixCSC{Float64, Int64}([1.0 0.0
                                     1.0 1.0])
b = ones(2)
A2 = hcat(A, A)
prob = LinearProblem(A, b)
@test SciMLBase.successful_retcode(solve(prob))

prob2 = LinearProblem(A2, b)
@test SciMLBase.successful_retcode(solve(prob2))

A = SparseMatrixCSC{Float64, Int32}([1.0 0.0
                                     1.0 1.0])
b = ones(2)
A2 = hcat(A, A)
prob = LinearProblem(A, b)
@test_broken SciMLBase.successful_retcode(solve(prob))

prob2 = LinearProblem(A2, b)
@test SciMLBase.successful_retcode(solve(prob2))

# Column-Pivoted QR fallback on failed LU
A = [1.0 0 0 0
     0 1 0 0
     0 0 1 0
     0 0 0 0]
b = rand(4)
prob = LinearProblem(A, b)
sol = solve(prob,
    LinearSolve.DefaultLinearSolver(
        LinearSolve.DefaultAlgorithmChoice.LUFactorization; safetyfallback = false))
@test sol.retcode === ReturnCode.Failure
@test sol.u == zeros(4)

sol = solve(prob)
@test sol.u ≈ svd(A)\b

