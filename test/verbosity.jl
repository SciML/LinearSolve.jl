using LinearSolve
using LinearSolve: LinearVerbosity
using SciMLVerbosity: Verbosity
using Logging
using Test

A = [1.0 0 0 0
     0 1 0 0
     0 0 1 0
     0 0 0 0]
b = rand(4)
prob = LinearProblem(A, b)

@test_logs (:warn, "Falling back to LU factorization") solve(prob, verbose = LinearVerbosity(default_lu_fallback = Verbosity.Warn()))

@test_logs (:info, "Falling back to LU factorization") solve(prob, verbose = LinearVerbosity(default_lu_fallback = Verbosity.Info()))

@test_logs min_level = Logging.Info solve(prob, verbose = LinearVerbosity(Verbosity.None()))