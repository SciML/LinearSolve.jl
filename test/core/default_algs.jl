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
@test solve(prob, PureKLUFactorization()).retcode == ReturnCode.Infeasible
# Column-pivoted sparse QR is rank-revealing: it returns a least-squares
# solution (Success) rather than Infeasible on a singular system.
@test SciMLBase.successful_retcode(solve(prob, SparseColumnPivotedQRFactorization()).retcode)

# Non-square sparse QR mirrors the KLU/UMFPACK structure split: less-structured
# (small) systems use the pure-Julia column-pivoted QR; more-structured
# (large + dense) systems use SuiteSparse SPQR via `QRFactorization`.
@test LinearSolve.defaultalg(
    sprand(20, 10, 0.3), zeros(20), LinearSolve.OperatorAssumptions(false)
).alg === LinearSolve.DefaultAlgorithmChoice.SparseColumnPivotedQRFactorization
if Base.USE_GPL_LIBS
    @test LinearSolve.defaultalg(
        sprand(2000, 1500, 0.5) + sparse(1:1500, 1:1500, 1.0, 2000, 1500),
        zeros(2000), LinearSolve.OperatorAssumptions(false)
    ).alg === LinearSolve.DefaultAlgorithmChoice.QRFactorization
end
let Anq = sprand(40, 25, 0.2) + sparse(1:25, 1:25, 1.0, 40, 25), bnq = rand(40)
    solnq = solve(LinearProblem(Anq, bnq))
    @test SciMLBase.successful_retcode(solnq.retcode)
    # least-squares solution satisfies the normal equations
    @test norm(Anq' * (Anq * solnq.u) - Anq' * bnq) < 1.0e-8
end

# Square sparse singular: the default sparse LU (PureKLU) falls back to the
# column-pivoted sparse QR and succeeds.
let As = sparse([1.0 0.0; 0.0 0.0]), bs = [1.0, 0.0]
    sols = solve(LinearProblem(As, bs))
    @test SciMLBase.successful_retcode(sols.retcode)
    @test all(isfinite, sols.u)
end

# Singular matrix with an *explicit stored zero* pivot: KLU/PureKLU report
# `KLU_OK` but produce a non-finite solution. The finiteness guard must surface
# that as `Infeasible` rather than a silent `Success` with NaNs.
let A_ez = sparse([1, 2], [1, 2], [1.0, 0.0], 2, 2), b_ez = ones(2)
    klu_sol = solve(LinearProblem(A_ez, b_ez), KLUFactorization())
    @test klu_sol.retcode == ReturnCode.Infeasible
    @test !any(isnan, klu_sol.u)  # not a silent NaN Success
    @test solve(LinearProblem(A_ez, b_ez), PureKLUFactorization()).retcode ==
        ReturnCode.Infeasible
end

# Generic (non-BLAS) element types such as BigFloat use the pure-Julia PureKLU as
# the default sparse LU — no `using Sparspak` required. This is the path BVP
# solvers hit for BigFloat problems.
let n = 20
    A_bf = sprand(BigFloat, n, n, 0.3) + (n * one(BigFloat)) * I
    b_bf = rand(BigFloat, n)
    @test LinearSolve.defaultalg(A_bf, b_bf, LinearSolve.OperatorAssumptions(true)).alg ===
        LinearSolve.DefaultAlgorithmChoice.KLUFactorization
    sol_bf = solve(LinearProblem(A_bf, b_bf))
    @test SciMLBase.successful_retcode(sol_bf.retcode)
    @test eltype(sol_bf.u) === BigFloat
    @test norm(A_bf * sol_bf.u - b_bf) / norm(b_bf) < 1.0e-30
    # Re-solve with a same-pattern numeric update (mimics Newton iterations):
    # PureKLU must refactor rather than reuse a stale factorization.
    cache_bf = init(LinearProblem(A_bf, b_bf))
    solve!(cache_bf)
    A_bf2 = copy(A_bf)
    A_bf2.nzval .*= 3
    cache_bf.A = A_bf2
    sol_bf2 = solve!(cache_bf)
    @test norm(A_bf2 * sol_bf2.u - b_bf) / norm(b_bf) < 1.0e-30
end
# Square sparse BigFloat singular system falls back to the generic column-pivoted
# sparse QR and succeeds.
let As_bf = sparse(BigFloat.([1 2 3; 2 4 6; 1 1 1])), bs_bf = BigFloat.([1, 2, 3])
    sol = solve(LinearProblem(As_bf, bs_bf))
    @test SciMLBase.successful_retcode(sol.retcode)
    @test all(isfinite, sol.u)
end

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
# Previously broken on the Sparspak path (its init_cacheval requires the index
# integer type to match); the PureKLU default handles Int32-indexed sparse.
@test SciMLBase.successful_retcode(solve(prob))

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
U = qr(randn(n, n)).Q * Diagonal(vcat(1.0e-15, ones(n - 1)))  # one near-zero singular value
A_nearsing = Matrix(U * Diagonal(ones(n)) * U')
# Make it clearly near-singular but with no exact zero pivot
A_nearsing .+= 1.0e-16 * randn(n, n)
b_nearsing = randn(n)

# Without safetyfallback, LU result is returned as-is (may be garbage)
sol_nosafe = solve(
    LinearProblem(copy(A_nearsing), copy(b_nearsing)),
    LinearSolve.DefaultLinearSolver(
        LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization; safetyfallback = false
    )
)
# With residualsafety=true, residual check should trigger QR fallback
sol_safe = solve(
    LinearProblem(copy(A_nearsing), copy(b_nearsing)),
    LinearSolve.DefaultLinearSolver(
        LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization;
        residualsafety = true
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
@test resid_safe <= 10 * resid_qr || resid_safe < 1.0e-10  # safe is close to QR quality
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
# LUFactorization(residualsafety=true) on near-singular matrix → APosterioriSafetyFailure
sol_lu_rs = solve(
    LinearProblem(copy(A_nearsing), copy(b_nearsing)),
    LUFactorization(residualsafety = true)
)
@test sol_lu_rs.retcode === ReturnCode.APosterioriSafetyFailure

# GenericLUFactorization(residualsafety=true) on near-singular matrix → APosterioriSafetyFailure
sol_glu_rs = solve(
    LinearProblem(copy(A_nearsing), copy(b_nearsing)),
    GenericLUFactorization(residualsafety = true)
)
@test sol_glu_rs.retcode === ReturnCode.APosterioriSafetyFailure

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

# QR fallback reuse: after QR fallback, subsequent solves with the same matrix
# should use QR (not the corrupted in-place LU cache).
# Regression test for https://github.com/SciML/LinearSolve.jl/issues/911
# Simulates the ODE solver pattern: first solve triggers QR fallback,
# then subsequent stages reuse the factorization without updating A.
A_illcond = copy(A_nearsing)
b1 = randn(n)
b2 = randn(n)
prob_reuse_qr = LinearProblem(A_illcond, b1)
alg_reuse = LinearSolve.DefaultLinearSolver(
    LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization;
    residualsafety = true
)
cache_qr_reuse = init(prob_reuse_qr, alg_reuse)
sol_stage1 = solve!(cache_qr_reuse)
@test sol_stage1.retcode === ReturnCode.Success
@test cache_qr_reuse.cacheval.fell_back_to_qr
# Stage 2: only change b, not A (simulates Rosenbrock stage reuse)
cache_qr_reuse.b = b2
sol_stage2 = solve!(cache_qr_reuse)
@test sol_stage2.retcode === ReturnCode.Success
# Verify the solution matches a fresh QR solve (not corrupted by LU)
sol_qr_stage2_ref = solve(LinearProblem(copy(A_nearsing), copy(b2)), QRFactorization(ColumnNorm()))
@test sol_stage2.u ≈ sol_qr_stage2_ref.u
# After setting a new A, fell_back_to_qr should be reset
cache_qr_reuse.A = copy(A_nearsing)  # triggers setproperty! for :A
@test !cache_qr_reuse.cacheval.fell_back_to_qr

# === Sparse KLU fast-path heuristic ===
# Tunes the sparse default-alg dispatch so that "small enough" sparse problems
# always pick KLU regardless of density. The previous heuristic only used
# density × size, which routed e.g. a 199×199 sparse matrix at 2.5% density to
# UMFPACK even though KLU is algorithmically the right tool for that size.
let
    # Small sparse, high density → fast path (length(b) ≤ 1_000) picks KLU
    A_small_dense = sprand(100, 100, 0.5) + I
    @test LinearSolve.defaultalg(A_small_dense, rand(100), LinearSolve.OperatorAssumptions(true)).alg ===
        LinearSolve.DefaultAlgorithmChoice.KLUFactorization

    # Right at the fast-path boundary, length(b) = 1_000, still picks KLU
    A_boundary = sprand(1_000, 1_000, 0.5) + I
    @test LinearSolve.defaultalg(A_boundary, rand(1_000), LinearSolve.OperatorAssumptions(true)).alg ===
        LinearSolve.DefaultAlgorithmChoice.KLUFactorization

    # Just past the fast-path boundary on a dense matrix → UMFPACK
    A_past = sprand(1_001, 1_001, 0.5) + I
    @test LinearSolve.defaultalg(A_past, rand(1_001), LinearSolve.OperatorAssumptions(true)).alg ===
        LinearSolve.DefaultAlgorithmChoice.UMFPACKFactorization

    # Medium-size, very sparse (density < 2e-4) → density branch picks KLU
    n = 9_000
    A_diag = spdiagm(0 => ones(n))
    @test nnz(A_diag) / length(A_diag) < 2.0e-4
    @test LinearSolve.defaultalg(A_diag, rand(n), LinearSolve.OperatorAssumptions(true)).alg ===
        LinearSolve.DefaultAlgorithmChoice.KLUFactorization

    # Medium-size, dense sparse → UMFPACK
    A_med_dense = sprand(5_000, 5_000, 0.5) + I
    @test LinearSolve.defaultalg(A_med_dense, rand(5_000), LinearSolve.OperatorAssumptions(true)).alg ===
        LinearSolve.DefaultAlgorithmChoice.UMFPACKFactorization
end

# === Sparse LU → SPQR fallback ===
# Mirrors the dense LU → QR fallback: if the sparse LU (KLU, UMFPACK,
# Sparspak) reports failure or produces non-finite values, the default
# solver retries with sparse column-pivoted QR (SPQR).
let
    # 1) Structurally singular small sparse matrix: direct KLU returns
    # Infeasible; default solver falls back and returns Success.
    A_singular = spzeros(3, 3)
    b_singular = ones(3)
    @test solve(LinearProblem(A_singular, b_singular), KLUFactorization()).retcode ===
        ReturnCode.Infeasible
    sol = solve(LinearProblem(A_singular, b_singular))
    @test sol.retcode === ReturnCode.Success
    @test all(iszero, sol.u)  # least-squares solution of A = 0

    # 2) Rank-deficient (zero row) — least-squares solution exists.
    A_rd = sparse([1.0 2.0; 0.0 0.0])
    b_rd = [3.0, 0.0]
    sol_rd = solve(LinearProblem(A_rd, b_rd))
    @test sol_rd.retcode === ReturnCode.Success

    # 3) ComplexF64 singular sparse → fallback also fires.
    A_c = sparse(ComplexF64[0 0; 0 0])
    b_c = ComplexF64[1.0, 1.0]
    @test solve(LinearProblem(A_c, b_c)).retcode === ReturnCode.Success

    # 4) safetyfallback = false ⇒ no fallback; LU's Infeasible propagates.
    alg_nofallback = LinearSolve.DefaultLinearSolver(
        LinearSolve.DefaultAlgorithmChoice.KLUFactorization; safetyfallback = false
    )
    @test solve(LinearProblem(spzeros(3, 3), ones(3)), alg_nofallback).retcode ===
        ReturnCode.Infeasible

    # 5) Reuse path: after fallback, changing only b should reuse the cached QR.
    cache_sp = init(LinearProblem(spzeros(4, 4), ones(4)))
    sol1 = solve!(cache_sp)
    @test sol1.retcode === ReturnCode.Success
    @test cache_sp.cacheval.fell_back_to_qr
    cache_sp.b = -ones(4)
    sol2 = solve!(cache_sp)
    @test sol2.retcode === ReturnCode.Success
    # Setting a new A clears the QR-fallback flag
    cache_sp.A = spzeros(4, 4)
    @test !cache_sp.cacheval.fell_back_to_qr

    # 6) UMFPACK path (length(b) > fast-path boundary, dense enough).
    # Zero row makes A rank-deficient; b[1] ≠ 0 ⇒ least-squares residual is ≥ 1.
    n_u = 1_200
    A_u = sprand(n_u, n_u, 0.001) + 1.0 * I
    A_u = SparseMatrixCSC(A_u)
    A_u[1, :] .= 0
    A_u = sparse(A_u)
    @test LinearSolve.defaultalg(A_u, ones(n_u), LinearSolve.OperatorAssumptions(true)).alg ===
        LinearSolve.DefaultAlgorithmChoice.UMFPACKFactorization
    sol_u = solve(LinearProblem(A_u, ones(n_u)))
    @test sol_u.retcode === ReturnCode.Success
    # Sanity: a non-singular matrix should NOT fall back.
    A_nons = spdiagm(0 => ones(50), 1 => fill(-0.5, 49), -1 => fill(-0.5, 49))
    sol_nons = solve(LinearProblem(A_nons, rand(50)))
    @test sol_nons.retcode === ReturnCode.Success
    @test !init(LinearProblem(A_nons, rand(50))).cacheval.fell_back_to_qr
end

# Regression test for https://github.com/SciML/LinearSolve.jl/issues/991:
# `KLUFactorization(; reuse_symbolic = false)` on a singular matrix used to
# throw `LinearAlgebra.SingularException` because the `reuse_symbolic = false`
# branch in `solve!(::LinearCache, ::KLUFactorization)` called `KLU.klu(A)`
# without `check = false`. It now returns `ReturnCode.Infeasible` like the
# `reuse_symbolic = true` branch.
let
    Is = [1, 2, 4, 1, 5, 7, 9]
    Js = [4, 4, 4, 7, 8, 9, 10]
    Vs = [
        0.35209876935890616, 0.11497167473826142, 0.24468931805408345,
        0.056730539026381366, 0.9152256278300128, 0.46648361943562067,
        0.6891333467010995,
    ]
    A_991 = sparse(Is, Js, Vs, 10, 10)
    pr_991 = LinearProblem(A_991, rand(10))
    @test solve(pr_991, KLUFactorization()).retcode === ReturnCode.Infeasible
    @test solve(pr_991, KLUFactorization(; reuse_symbolic = false)).retcode ===
        ReturnCode.Infeasible
    # Default solver path: should fall back to SPQR and succeed.
    @test solve(pr_991).retcode === ReturnCode.Success
end
