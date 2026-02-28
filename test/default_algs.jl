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

@test LinearSolve.defaultalg(
    nothing, zeros(5),
    LinearSolve.OperatorAssumptions(false)
).alg ===
    LinearSolve.DefaultAlgorithmChoice.QRFactorization

A = spzeros(2, 2)
# test that solving a singular problem doesn't error
prob = LinearProblem(A, ones(2))
@test solve(prob, UMFPACKFactorization()).retcode == ReturnCode.Infeasible
@test solve(prob, KLUFactorization()).retcode == ReturnCode.Infeasible

@test LinearSolve.defaultalg(sprand(10^4, 10^4, 1.0e-5) + I, zeros(1000)).alg ===
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
A = [
    2.0 1.0
    0.0 0.0
    0.0 0.0
]
b = [1.0, 0.0, 0.0]
prob = LinearProblem(A, b)
sol = solve(prob)

@test !SciMLBase.successful_retcode(sol.retcode)

## Show that we cannot select a default alg once by checking the rank, since it might change
## later in the cache
## Common occurrence for iterative nonlinear solvers using linear solve
A = [
    2.0 1.0
    1.0 1.0
    0.0 0.0
]
b = [1.0, 1.0, 0.0]
prob = LinearProblem(A, b)

cache = init(prob)
sol = solve!(cache)

@test sol.u ≈ [0.0, 1.0]

cache.A = [
    2.0 1.0
    0.0 0.0
    0.0 0.0
]

sol = solve!(cache)

@test !SciMLBase.successful_retcode(sol.retcode)

## Non-square Sparse Defaults
# https://github.com/SciML/NonlinearSolve.jl/issues/599
A = SparseMatrixCSC{Float64, Int64}(
    [
        1.0 0.0
        1.0 1.0
    ]
)
b = ones(2)
A2 = hcat(A, A)
prob = LinearProblem(A, b)
@test SciMLBase.successful_retcode(solve(prob))

prob2 = LinearProblem(A2, b)
@test SciMLBase.successful_retcode(solve(prob2))

A = SparseMatrixCSC{Float64, Int32}(
    [
        1.0 0.0
        1.0 1.0
    ]
)
b = ones(2)
A2 = hcat(A, A)
prob = LinearProblem(A, b)
@test_broken SciMLBase.successful_retcode(solve(prob))

prob2 = LinearProblem(A2, b)
@test SciMLBase.successful_retcode(solve(prob2))

# Column-Pivoted QR fallback on failed LU
A = [
    1.0 0 0 0
    0 1 0 0
    0 0 1 0
    0 0 0 0
]
b = rand(4)
prob = LinearProblem(A, b)
sol = solve(
    prob,
    LinearSolve.DefaultLinearSolver(
        LinearSolve.DefaultAlgorithmChoice.LUFactorization; safetyfallback = false
    )
)
@test sol.retcode === ReturnCode.Failure
@test sol.u == zeros(4)

sol = solve(prob)
@test sol.u ≈ svd(A) \ b

# Test that QR fallback restores cache.A from prob.A after in-place LU corruption
# (Regression test for https://github.com/SciML/LinearSolve.jl/issues/887)
# In-place LU (MKL, AppleAccelerate, Generic, RF) corrupts cache.A via getrf!.
# The fallback must restore cache.A from the backup before running QR.
# Use a non-trivial singular matrix (not already in LU-factored form) so that
# in-place LU actually changes cache.A.
A_singular = Float64[
    0.48 0.77 0.35 0.12
    0.91 0.24 0.68 0.55
    0.63 0.42 0.19 0.87
    0.0 0.0 0.0 0.0
]
b_singular = Float64[0.31, 0.72, 0.56, 0.14]

sol_qr = solve(
    LinearProblem(copy(A_singular), copy(b_singular)), QRFactorization(ColumnNorm())
)

# Test through GenericLUFactorization explicitly (in-place via generic_lufact!)
sol_generic = solve(
    LinearProblem(copy(A_singular), copy(b_singular)),
    LinearSolve.DefaultLinearSolver(
        LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization
    )
)
@test sol_generic.retcode === ReturnCode.Success
@test sol_generic.u ≈ sol_qr.u

# Test through default solver (selects GenericLUFactorization for size ≤ 10)
prob_singular = LinearProblem(copy(A_singular), copy(b_singular))
sol_default = solve(prob_singular)
@test sol_default.retcode === ReturnCode.Success
@test sol_default.u ≈ sol_qr.u

# Verify prob.A is not modified (serves as zero-cost backup)
prob_backup = LinearProblem(copy(A_singular), copy(b_singular))
cache_backup = init(prob_backup)
@test !(cache_backup.A === cache_backup.cacheval.A_backup) # not aliased
@test cache_backup.cacheval.A_backup === prob_backup.A      # references prob.A
solve!(cache_backup)
@test prob_backup.A ≈ A_singular  # prob.A unchanged

# Regression test: A_backup must be updated from cache.A before each LU solve.
# When a caller (e.g. NonlinearSolve) reuses the cache with a different matrix via
# copyto!(cache.A, new_J), the QR fallback must use the current matrix, not the
# stale initial one.
prob_reuse = LinearProblem(copy(A_singular), copy(b_singular))
cache_reuse = init(prob_reuse)
sol1 = solve!(cache_reuse)
@test sol1.retcode === ReturnCode.Success

# Now update cache.A with a different singular matrix (same size) and re-solve.
# This simulates NonlinearSolve updating the Jacobian between Newton steps.
A_singular2 = Float64[
    0.0 0.0 0.0 0.0
    0.0 2.0 0.0 0.0
    0.0 0.0 3.0 0.0
    0.0 0.0 0.0 0.0
]
b_singular2 = Float64[0.0, 4.0, 9.0, 0.0]
copyto!(cache_reuse.A, A_singular2)
cache_reuse.A = cache_reuse.A  # trigger setproperty! to sync A_backup
copyto!(cache_reuse.b, b_singular2)
sol2 = solve!(cache_reuse)
@test sol2.retcode === ReturnCode.Success
sol_qr2 = solve(
    LinearProblem(copy(A_singular2), copy(b_singular2)), QRFactorization(ColumnNorm())
)
@test sol2.u ≈ sol_qr2.u

# Regression test for https://github.com/SciML/LinearSolve.jl/issues/890
# WOperator with init_cacheval overload that unwraps A.J (as OrdinaryDiffEqDifferentiation does)
using SciMLOperators: WOperator, MatrixOperator
function LinearSolve.init_cacheval(
        alg::LinearSolve.DefaultLinearSolver, A::WOperator, b, u,
        Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{Bool, LinearSolve.LinearVerbosity},
        assumptions::LinearSolve.OperatorAssumptions
    )
    return LinearSolve.init_cacheval(
        alg, A.J, b, u, Pl, Pr,
        maxiters, abstol, reltol, verbose, assumptions
    )
end

A_w = sparse([-1.0 0.5; 0.3 -1.0])
J_w = MatrixOperator(A_w)
M_w = MatrixOperator(sparse([1.0 0.0; 0.0 1.0]))
W = WOperator{true}(M_w, 1.0, J_w, [1.0, 0.5])
prob_w = LinearProblem(W, [1.0, 2.0])
sol_w = solve(prob_w)
@test sol_w.retcode === ReturnCode.Success

# Post-solve residual check: near-singular matrix where LU "succeeds" but gives garbage
# LU doesn't flag near-zero pivots, only exact zeros. The residual check catches this.
using Random
Random.seed!(42)
n = 20
U = qr(randn(n, n)).Q * Diagonal(vcat(1e-15, ones(n - 1)))  # one near-zero singular value
A_nearsing = Matrix(U * Diagonal(ones(n)) * U')
# Make it clearly near-singular but with no exact zero pivot
A_nearsing .+= 1e-16 * randn(n, n)
b_nearsing = randn(n)

# Without safetyfallback, LU result is returned as-is (may be garbage)
sol_nosafe = solve(
    LinearProblem(copy(A_nearsing), copy(b_nearsing)),
    LinearSolve.DefaultLinearSolver(
        LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization; safetyfallback = false
    )
)
# With safetyfallback (default), residual check should trigger QR fallback
sol_safe = solve(
    LinearProblem(copy(A_nearsing), copy(b_nearsing)),
    LinearSolve.DefaultLinearSolver(
        LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization
    )
)
sol_qr_ref = solve(
    LinearProblem(copy(A_nearsing), copy(b_nearsing)),
    QRFactorization(ColumnNorm())
)
# The safe fallback should give a much better solution than raw LU
resid_nosafe = norm(A_nearsing * sol_nosafe.u - b_nearsing)
resid_safe = norm(A_nearsing * sol_safe.u - b_nearsing)
resid_qr = norm(A_nearsing * sol_qr_ref.u - b_nearsing)
@test resid_safe <= 10 * resid_qr || resid_safe < 1e-10  # safe is close to QR quality
@test sol_safe.retcode === ReturnCode.Success

# Well-conditioned matrix: residual check should NOT trigger fallback
A_wellcond = Float64[4 1; 1 3]
b_wellcond = Float64[1.0, 2.0]
sol_well = solve(LinearProblem(copy(A_wellcond), copy(b_wellcond)))
@test sol_well.retcode === ReturnCode.Success
@test sol_well.u ≈ A_wellcond \ b_wellcond

# safetyfallback=false skips residual check entirely
sol_no_fallback = solve(
    LinearProblem(copy(A_nearsing), copy(b_nearsing)),
    LinearSolve.DefaultLinearSolver(
        LinearSolve.DefaultAlgorithmChoice.LUFactorization; safetyfallback = false
    )
)
# Should return without fallback (no error, but possibly bad solution)
@test sol_no_fallback.retcode === ReturnCode.Success

# Individual LU algorithm residualsafety tests
# LUFactorization(residualsafety=true) on near-singular matrix → ReturnCode.Failure
sol_lu_rs = solve(
    LinearProblem(copy(A_nearsing), copy(b_nearsing)),
    LUFactorization(residualsafety = true)
)
@test sol_lu_rs.retcode === ReturnCode.Failure

# GenericLUFactorization(residualsafety=true) on near-singular matrix → ReturnCode.Failure
sol_glu_rs = solve(
    LinearProblem(copy(A_nearsing), copy(b_nearsing)),
    GenericLUFactorization(residualsafety = true)
)
@test sol_glu_rs.retcode === ReturnCode.Failure

# Default LUFactorization() on near-singular matrix → ReturnCode.Success (no check)
sol_lu_default = solve(
    LinearProblem(copy(A_nearsing), copy(b_nearsing)),
    LUFactorization()
)
@test sol_lu_default.retcode === ReturnCode.Success

# Well-conditioned matrix: residualsafety=true should still succeed
sol_lu_well_rs = solve(
    LinearProblem(copy(A_wellcond), copy(b_wellcond)),
    LUFactorization(residualsafety = true)
)
@test sol_lu_well_rs.retcode === ReturnCode.Success
@test sol_lu_well_rs.u ≈ A_wellcond \ b_wellcond
