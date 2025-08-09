using Pkg
Pkg.activate("lib/LinearSolveAutotune")
Pkg.instantiate()

using LinearSolveAutotune
using LinearSolve
using Test

# Test 1: Check that get_package_versions works
println("Test 1: Checking package version collection...")
sys_info = LinearSolveAutotune.get_system_info()
if haskey(sys_info, "package_versions")
    println("✓ Package versions collected successfully:")
    versions = sys_info["package_versions"]
    for (pkg, ver) in versions
        println("  - $pkg: $ver")
    end
else
    println("✗ Package versions not found in system info")
end

# Test 2: Test correctness check with a small matrix
println("\nTest 2: Testing correctness check...")
using Random, LinearAlgebra
Random.seed!(123)

n = 10
A = rand(Float64, n, n)
b = rand(Float64, n)

# Get reference solution
prob_ref = LinearProblem(A, b)
ref_sol = solve(prob_ref, LinearSolve.LUFactorization())
println("Reference solution norm: ", norm(ref_sol.u))

# Test with a simple algorithm
prob_test = LinearProblem(A, b)
test_sol = solve(prob_test, LinearSolve.SimpleLUFactorization())
rel_error = norm(test_sol.u - ref_sol.u) / norm(ref_sol.u)
println("SimpleLUFactorization relative error: ", rel_error)

if rel_error < 1e-2
    println("✓ Correctness check would pass (error < 1e-2)")
else
    println("✗ Correctness check would fail (error >= 1e-2)")
end

println("\nTest 3: Running small benchmark with correctness checks...")
# Run a minimal benchmark to test the full integration
matrix_sizes = [5, 10]
algorithms = [LinearSolve.LUFactorization(), LinearSolve.SimpleLUFactorization()]
alg_names = ["LUFactorization", "SimpleLUFactorization"]
eltypes = [Float64]

results = LinearSolveAutotune.benchmark_algorithms(
    matrix_sizes, algorithms, alg_names, eltypes;
    samples = 2, seconds = 0.1, check_correctness = true, correctness_tol = 1e-2
)

println("\nBenchmark results with correctness checks:")
println(results)

println("\n✓ All tests completed successfully!")