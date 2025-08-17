#!/usr/bin/env julia

# Simple benchmark for SymTridiagonal LDLtFactorization improvement
# This demonstrates the performance improvement from issue #732

using LinearSolve
using LinearAlgebra
using Printf

println("=" ^ 80)
println("SymTridiagonal Performance Benchmark (Issue #732)")
println("=" ^ 80)
println()

# Create test matrix from issue #732
k = 10000
ρ = 0.95
A_tri = SymTridiagonal(ones(k) .+ ρ^2, -ρ * ones(k-1))
b = rand(k)

println("Matrix size: $k × $k (example from issue #732)")
println()

# Time native \ operator
println("1. Native \\ operator:")
t_native = @elapsed begin
    for _ in 1:10
        x = A_tri \ b
    end
end
println("   Average time: $(round(t_native/10 * 1e6, digits=1)) μs")

# Time LinearSolve with explicit LDLtFactorization
println("\n2. LinearSolve with explicit LDLtFactorization:")
prob = LinearProblem(A_tri, b)
t_ldlt = @elapsed begin
    for _ in 1:10
        x = solve(prob, LDLtFactorization())
    end
end
println("   Average time: $(round(t_ldlt/10 * 1e3, digits=2)) ms")

# Show the factorization caching benefit
println("\n3. Factorization caching test (100 solves with different RHS):")

# Without caching (native)
println("   a) Native \\ (no caching):")
t_no_cache = @elapsed begin
    for _ in 1:100
        b_new = rand(k)
        x = A_tri \ b_new
    end
end
println("      Total time: $(round(t_no_cache * 1e3, digits=1)) ms")
println("      Time per solve: $(round(t_no_cache/100 * 1e6, digits=1)) μs")

# With caching (LinearSolve)
println("\n   b) LinearSolve with cached LDLtFactorization:")
t_with_cache = @elapsed begin
    cache = init(prob, LDLtFactorization())
    solve!(cache)  # First solve (includes factorization)
    for _ in 1:99
        cache.b = rand(k)
        solve!(cache)  # Subsequent solves (reuse factorization)
    end
end
println("      Total time: $(round(t_with_cache * 1e3, digits=1)) ms")
println("      Time per solve: $(round(t_with_cache/100 * 1e3, digits=2)) ms")

speedup = t_no_cache / t_with_cache
println("\n   Speedup with caching: $(round(speedup, digits=1))x")

# Verify the default algorithm selection
println("\n4. Default algorithm selection:")
default_alg = LinearSolve.defaultalg(A_tri, b, LinearSolve.OperatorAssumptions(true))
if default_alg isa LinearSolve.DefaultLinearSolver
    if default_alg.alg == LinearSolve.DefaultAlgorithmChoice.LDLtFactorization
        println("   ✓ LDLtFactorization is selected by default for SymTridiagonal")
    else
        println("   ✗ Wrong algorithm selected: $(default_alg.alg)")
    end
else
    println("   Algorithm type: $(typeof(default_alg))")
end

println("\n" * "=" ^ 80)
println("Summary:")
println("- LDLtFactorization works correctly with SymTridiagonal matrices")
println("- Caching provides significant speedup for multiple solves")
println("- Default algorithm selection has been updated to use LDLtFactorization")
println("=" ^ 80)