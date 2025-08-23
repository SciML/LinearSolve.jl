using LinearSolve
using LinearAlgebra
using BenchmarkTools

println("Testing allocation improvements for 32Mixed precision methods...")

# Test size
n = 100
A = rand(Float64, n, n) + 5.0I  # Well-conditioned matrix
b = rand(Float64, n)

# Test MKL32MixedLUFactorization if available
if LinearSolve.usemkl
    println("\nTesting MKL32MixedLUFactorization:")
    prob = LinearProblem(A, b)
    
    # Warm up
    sol = solve(prob, MKL32MixedLUFactorization())
    
    # Test allocations on subsequent solves
    cache = init(prob, MKL32MixedLUFactorization())
    solve!(cache)  # First solve (factorization)
    
    # Change b and solve again - this should have minimal allocations
    cache.b .= rand(n)
    alloc_bytes = @allocated solve!(cache)
    println("  Allocations on second solve: $alloc_bytes bytes")
    
    # Benchmark
    println("  Benchmark results:")
    @btime solve!(cache) setup=(cache.b .= rand($n))
end

# Test OpenBLAS32MixedLUFactorization if available
if LinearSolve.useopenblas
    println("\nTesting OpenBLAS32MixedLUFactorization:")
    prob = LinearProblem(A, b)
    
    # Warm up
    sol = solve(prob, OpenBLAS32MixedLUFactorization())
    
    # Test allocations on subsequent solves
    cache = init(prob, OpenBLAS32MixedLUFactorization())
    solve!(cache)  # First solve (factorization)
    
    # Change b and solve again - this should have minimal allocations
    cache.b .= rand(n)
    alloc_bytes = @allocated solve!(cache)
    println("  Allocations on second solve: $alloc_bytes bytes")
    
    # Benchmark
    println("  Benchmark results:")
    @btime solve!(cache) setup=(cache.b .= rand($n))
end

# Test AppleAccelerate32MixedLUFactorization if available
if Sys.isapple() && LinearSolve.appleaccelerate_isavailable()
    println("\nTesting AppleAccelerate32MixedLUFactorization:")
    prob = LinearProblem(A, b)
    
    # Warm up
    sol = solve(prob, AppleAccelerate32MixedLUFactorization())
    
    # Test allocations on subsequent solves
    cache = init(prob, AppleAccelerate32MixedLUFactorization())
    solve!(cache)  # First solve (factorization)
    
    # Change b and solve again - this should have minimal allocations
    cache.b .= rand(n)
    alloc_bytes = @allocated solve!(cache)
    println("  Allocations on second solve: $alloc_bytes bytes")
    
    # Benchmark
    println("  Benchmark results:")
    @btime solve!(cache) setup=(cache.b .= rand($n))
end

# Test RF32MixedLUFactorization if RecursiveFactorization is available
try
    using RecursiveFactorization
    println("\nTesting RF32MixedLUFactorization:")
    prob = LinearProblem(A, b)
    
    # Warm up
    sol = solve(prob, RF32MixedLUFactorization())
    
    # Test allocations on subsequent solves
    cache = init(prob, RF32MixedLUFactorization())
    solve!(cache)  # First solve (factorization)
    
    # Change b and solve again - this should have minimal allocations
    cache.b .= rand(n)
    alloc_bytes = @allocated solve!(cache)
    println("  Allocations on second solve: $alloc_bytes bytes")
    
    # Benchmark
    println("  Benchmark results:")
    @btime solve!(cache) setup=(cache.b .= rand($n))
catch e
    println("\nRecursiveFactorization not available, skipping RF32MixedLUFactorization test")
end

println("\n✅ Allocation test complete!")
println("Note: Ideally, the allocation count on the second solve should be minimal (< 1KB)")
println("      as all temporary arrays should be pre-allocated in init_cacheval.")