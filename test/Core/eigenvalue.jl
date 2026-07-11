using LinearSolve
using LinearAlgebra
using Test
using Arpack
using ArnoldiMethod
using KrylovKit
using JacobiDavidson

A = Diagonal([1.0, 2.0, 3.0, 4.0])

prob = EigenvalueProblem(A)
sol = solve(prob)
@test sol.u == [4.0, 3.0, 2.0, 1.0]
@test sol.vectors * Diagonal(sol.u) ≈ A * sol.vectors
@test sol.retcode === ReturnCode.Success

prob_nev = EigenvalueProblem(A; num_eigenpairs = 2, eigentarget = EigenvalueTarget.SmallestMagnitude)
sol_nev = solve(prob_nev)
@test sol_nev.u == [1.0, 2.0]

prob_shift = EigenvalueProblem(A; num_eigenpairs = 2, shift = 2.2)
sol_shift = solve(prob_shift)
@test sol_shift.u == [2.0, 3.0]

B = Diagonal(fill(2.0, 4))
prob_gen = EigenvalueProblem(A, B; num_eigenpairs = 2, eigentarget = EigenvalueTarget.LargestRealPart)
sol_gen = solve(prob_gen)
@test sol_gen.u == [2.0, 1.5]

prob_uniform = EigenvalueProblem(A, 2I; num_eigenpairs = 2, eigentarget = EigenvalueTarget.LargestRealPart)
sol_uniform = solve(prob_uniform)
@test sol_uniform.u == [2.0, 1.5]

A_backend = Diagonal(1.0:8.0)
B_backend = Diagonal(fill(2.0, 8))

sol_arpack = solve(EigenvalueProblem(Matrix(A_backend); num_eigenpairs = 2), ArpackJL())
@test sol_arpack.u ≈ [8.0, 7.0]

sol_arpack_gen = solve(
    EigenvalueProblem(Matrix(A_backend), Matrix(B_backend); num_eigenpairs = 2), ArpackJL()
)
@test sol_arpack_gen.u ≈ [4.0, 3.5]

sol_arnoldi = solve(
    EigenvalueProblem(A_backend; num_eigenpairs = 2), LinearSolve.ArnoldiMethod()
)
@test sol_arnoldi.u ≈ [8.0, 7.0]

sol_arnoldi_default = solve(EigenvalueProblem(A_backend), LinearSolve.ArnoldiMethod())
@test length(sol_arnoldi_default.u) == 6

sol_arnoldi_shift = solve(
    EigenvalueProblem(A_backend; num_eigenpairs = 2, shift = 3.2), LinearSolve.ArnoldiMethod()
)
@test sol_arnoldi_shift.u ≈ [3.0, 4.0]

# ArnoldiMethod has no smallest-magnitude target of its own.
@test_throws ArgumentError solve(
    EigenvalueProblem(
        A_backend; num_eigenpairs = 2, eigentarget = EigenvalueTarget.SmallestMagnitude
    ),
    LinearSolve.ArnoldiMethod()
)

sol_krylovkit = solve(EigenvalueProblem(A_backend; num_eigenpairs = 2), KrylovKitEigen())
@test sol_krylovkit.u ≈ [8.0, 7.0]

sol_krylovkit_shift = solve(
    EigenvalueProblem(Matrix(A_backend); num_eigenpairs = 2, shift = 3.2), KrylovKitEigen()
)
@test sol_krylovkit_shift.u ≈ [3.0, 4.0]

sol_krylovkit_gen_shift = solve(
    EigenvalueProblem(Matrix(A_backend), Matrix(B_backend); num_eigenpairs = 2, shift = 1.6),
    KrylovKitEigen()
)
@test sol_krylovkit_gen_shift.u ≈ [1.5, 2.0]

# Jacobi-Davidson is a target/interior method: it finds the eigenvalues nearest
# the target shift. `shift` gives the interior target directly, and
# `SmallestMagnitude`/`SmallestRealPart` work with the implicit zero shift.
# (`LargestMagnitude`/`LargestRealPart` need a `shift` guess.)
A_jd = Matrix(Diagonal(Float64.(1:30)))

sol_jd = solve(EigenvalueProblem(A_jd; num_eigenpairs = 2, shift = 10.3), JacobiDavidsonJL())
@test sort(real(sol_jd.u)) ≈ [10.0, 11.0]
@test sol_jd.retcode === ReturnCode.Success
@test norm(A_jd * sol_jd.vectors - sol_jd.vectors * Diagonal(sol_jd.u)) < 1.0e-6

sol_jd_sm = solve(
    EigenvalueProblem(
        A_jd; num_eigenpairs = 1, eigentarget = EigenvalueTarget.SmallestMagnitude
    ),
    JacobiDavidsonJL()
)
@test real(sol_jd_sm.u[1]) ≈ 1.0

# Generalized problems are not supported by the JacobiDavidson backend (upstream
# `jdqz` is broken); it should error rather than silently misbehave.
@test_throws ErrorException solve(
    EigenvalueProblem(A_jd, Matrix(Diagonal(fill(2.0, 30))); num_eigenpairs = 1),
    JacobiDavidsonJL()
)

# `eigentarget` only accepts `EigenvalueTarget` values, not ARPACK-style symbols.
prob_enum = EigenvalueProblem(
    A; num_eigenpairs = 2, eigentarget = EigenvalueTarget.SmallestMagnitude
)
@test prob_enum.eigentarget === EigenvalueTarget.SmallestMagnitude
@test solve(prob_enum).u == [1.0, 2.0]

# An invalid `eigentarget` is rejected at construction with a clear type error.
@test_throws TypeError EigenvalueProblem(A; eigentarget = :LM)
@test_throws TypeError EigenvalueProblem(A; eigentarget = 5)

# The iterative algorithms are keyword-only: positional arguments error, and
# forwarded keywords reach the backend.
@test_throws MethodError ArpackJL(5)
@test ArpackJL(; maxiter = 500).kwargs.maxiter == 500
