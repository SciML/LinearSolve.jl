using LinearSolve
using LinearSolve: LinearVerbosity
using SciMLLogging: SciMLLogging, Verbosity
using Test

A = [1.0 0 0 0
     0 1 0 0
     0 0 1 0
     0 0 0 0]
b = rand(4)
prob = LinearProblem(A, b)

@test_logs (:warn,
    "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.") solve(
    prob,
    verbose = LinearVerbosity(default_lu_fallback = Verbosity.Warn()))

@test_logs (:warn,
    "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.") solve(
    prob, verbose = true)

@test_logs min_level=SciMLLogging.Logging.Warn solve(prob, verbose = false)

@test_logs (:info,
    "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.") solve(
    prob,
    verbose = LinearVerbosity(default_lu_fallback = Verbosity.Info()))