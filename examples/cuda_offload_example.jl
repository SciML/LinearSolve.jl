"""
Example demonstrating the new CudaOffloadLUFactorization and CudaOffloadQRFactorization algorithms.

This example shows how to use the new GPU offloading algorithms for solving linear systems
with different numerical properties.
"""

using LinearSolve
using LinearAlgebra
using Random

# Set random seed for reproducibility
Random.seed!(123)

println("CUDA Offload Factorization Examples")
println("=" ^ 40)

# Create a well-conditioned problem
println("\n1. Well-conditioned problem (condition number ≈ 10)")
A_good = rand(100, 100)
A_good = A_good + 10I  # Make it well-conditioned
b_good = rand(100)
prob_good = LinearProblem(A_good, b_good)

println("   Matrix size: $(size(A_good))")
println("   Condition number: $(cond(A_good))")

# Try to use CUDA if available
try
    using CUDA
    
    # Solve with LU (faster for well-conditioned)
    println("\n   Solving with CudaOffloadLUFactorization...")
    sol_lu = solve(prob_good, CudaOffloadLUFactorization())
    println("   Solution norm: $(norm(sol_lu.u))")
    println("   Residual norm: $(norm(A_good * sol_lu.u - b_good))")
    
    # Solve with QR (more stable)
    println("\n   Solving with CudaOffloadQRFactorization...")
    sol_qr = solve(prob_good, CudaOffloadQRFactorization())
    println("   Solution norm: $(norm(sol_qr.u))")
    println("   Residual norm: $(norm(A_good * sol_qr.u - b_good))")
    
catch e
    println("\n   Note: CUDA.jl is not loaded. To use GPU offloading:")
    println("   1. Install CUDA.jl: using Pkg; Pkg.add(\"CUDA\")")
    println("   2. Add 'using CUDA' before running this example")
    println("   3. Ensure you have a CUDA-compatible NVIDIA GPU")
end

# Create an ill-conditioned problem
println("\n2. Ill-conditioned problem (condition number ≈ 1e6)")
A_bad = rand(50, 50)
# Make it ill-conditioned
U, S, V = svd(A_bad)
S[end] = S[1] / 1e6  # Create large condition number
A_bad = U * Diagonal(S) * V'
b_bad = rand(50)
prob_bad = LinearProblem(A_bad, b_bad)

println("   Matrix size: $(size(A_bad))")
println("   Condition number: $(cond(A_bad))")

try
    using CUDA
    
    # For ill-conditioned problems, QR is typically more stable
    println("\n   Solving with CudaOffloadQRFactorization (recommended for ill-conditioned)...")
    sol_qr_bad = solve(prob_bad, CudaOffloadQRFactorization())
    println("   Solution norm: $(norm(sol_qr_bad.u))")
    println("   Residual norm: $(norm(A_bad * sol_qr_bad.u - b_bad))")
    
    println("\n   Solving with CudaOffloadLUFactorization (may be less stable)...")
    sol_lu_bad = solve(prob_bad, CudaOffloadLUFactorization())
    println("   Solution norm: $(norm(sol_lu_bad.u))")
    println("   Residual norm: $(norm(A_bad * sol_lu_bad.u - b_bad))")
    
catch e
    println("\n   Skipping GPU tests (CUDA not available)")
end

# Demonstrate the deprecation warning
println("\n3. Testing deprecated CudaOffloadFactorization")
try
    using CUDA
    println("   Creating deprecated CudaOffloadFactorization...")
    alg = CudaOffloadFactorization()  # This will show a deprecation warning
    println("   The deprecated algorithm still works but shows a warning above")
catch e
    println("   Skipping deprecation test (CUDA not available)")
end

println("\n" * "=" ^ 40)
println("Summary:")
println("- Use CudaOffloadLUFactorization for well-conditioned problems (faster)")
println("- Use CudaOffloadQRFactorization for ill-conditioned problems (more stable)")
println("- The old CudaOffloadFactorization is deprecated")