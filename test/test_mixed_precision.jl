using Test
using LinearSolve
using LinearAlgebra
using Random

Random.seed!(123)

@testset "Mixed Precision LU Factorizations" begin
    n = 100
    A = rand(Float64, n, n)
    b = rand(Float64, n)
    
    # Make A better conditioned to avoid excessive precision loss
    A = A + 5.0 * I
    
    prob = LinearProblem(A, b)
    
    # Reference solution with full precision
    sol_ref = solve(prob, LUFactorization())
    
    @testset "MKL32MixedLUFactorization" begin
        if LinearSolve.usemkl
            sol_mixed = solve(prob, MKL32MixedLUFactorization())
            @test sol_mixed.retcode == ReturnCode.Success
            # Check that solution is reasonably close (allowing for reduced precision)
            @test norm(sol_mixed.u - sol_ref.u) / norm(sol_ref.u) < 1e-5
            # Verify it actually solves the system
            @test norm(A * sol_mixed.u - b) / norm(b) < 1e-5
        else
            @test_skip "MKL not available"
        end
    end
    
    @testset "AppleAccelerate32MixedLUFactorization" begin
        if Sys.isapple()
            sol_mixed = solve(prob, AppleAccelerate32MixedLUFactorization())
            @test sol_mixed.retcode == ReturnCode.Success
            # Check that solution is reasonably close (allowing for reduced precision)
            @test norm(sol_mixed.u - sol_ref.u) / norm(sol_ref.u) < 1e-5
            # Verify it actually solves the system
            @test norm(A * sol_mixed.u - b) / norm(b) < 1e-5
        else
            @test_skip "Apple Accelerate not available"
        end
    end
    
    @testset "Complex matrices" begin
        # Test with complex matrices
        A_complex = rand(ComplexF64, n, n) + 5.0 * I
        b_complex = rand(ComplexF64, n)
        prob_complex = LinearProblem(A_complex, b_complex)
        sol_ref_complex = solve(prob_complex, LUFactorization())
        
        if LinearSolve.usemkl
            sol_mixed = solve(prob_complex, MKL32MixedLUFactorization())
            @test sol_mixed.retcode == ReturnCode.Success
            @test norm(sol_mixed.u - sol_ref_complex.u) / norm(sol_ref_complex.u) < 1e-5
        end
        
        if Sys.isapple()
            sol_mixed = solve(prob_complex, AppleAccelerate32MixedLUFactorization())
            @test sol_mixed.retcode == ReturnCode.Success
            @test norm(sol_mixed.u - sol_ref_complex.u) / norm(sol_ref_complex.u) < 1e-5
        end
    end
end

# Note: CUDA and Metal tests would require those packages to be loaded
# and appropriate hardware to be available
@testset "GPU Mixed Precision (Mocked)" begin
    @test isdefined(LinearSolve, :CUDAOffload32MixedLUFactorization)
    @test isdefined(LinearSolve, :MetalOffload32MixedLUFactorization)
    
    # These would error without the appropriate packages loaded, which is expected
    @test_throws Exception CUDAOffload32MixedLUFactorization()
    @test_throws Exception MetalOffload32MixedLUFactorization()
end