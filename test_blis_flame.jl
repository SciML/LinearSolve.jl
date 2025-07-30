#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

# First, install and load the required JLL packages (since they're weak dependencies)
try
    Pkg.add(["blis_jll", "libflame_jll"])
catch e
    println("Note: JLL packages may already be installed: ", e)
end

using blis_jll, libflame_jll
println("BLIS path: ", blis_jll.blis)
println("libFLAME path: ", libflame_jll.libflame)

# Load LinearSolve and test the BLIS extension - this should trigger the extension loading
using LinearSolve

# Test basic functionality
A = rand(4, 4)
b = rand(4)
prob = LinearProblem(A, b)

println("Testing BLISLUFactorization with FLAME...")
try
    sol = solve(prob, LinearSolve.BLISLUFactorization())
    println("✓ BLISLUFactorization successful!")
    println("Solution norm: ", norm(sol.u))
    
    # Verify solution accuracy
    residual = norm(A * sol.u - b)
    println("Residual norm: ", residual)
    
    if residual < 1e-10
        println("✓ Solution is accurate!")
    else
        println("✗ Solution may not be accurate")
    end
    
catch err
    println("✗ Error occurred: ", err)
    
    # Let's try to get more detailed error information
    println("\nDiagnosing issue...")
    
    # Check if the extension is loaded
    if hasmethod(LinearSolve.BLISLUFactorization, ())
        println("✓ BLISLUFactorization is available")
    else
        println("✗ BLISLUFactorization is not available")
    end
    
    # Check if we can create an instance
    try
        alg = LinearSolve.BLISLUFactorization()
        println("✓ Can create BLISLUFactorization instance")
    catch e
        println("✗ Cannot create BLISLUFactorization instance: ", e)
    end
    
    rethrow(err)
end