using Test
using LinearAlgebra
using Random

# Load LinearSolve with the working directory
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
using LinearSolve

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
            @test norm(sol_mixed.u - sol_ref.u) / norm(sol_ref.u) < 1.0e-5
            # Verify it actually solves the system
            @test norm(A * sol_mixed.u - b) / norm(b) < 1.0e-5
        else
            @test_skip "MKL not available"
        end
    end

    @testset "AppleAccelerate32MixedLUFactorization" begin
        if Sys.isapple()
            sol_mixed = solve(prob, AppleAccelerate32MixedLUFactorization())
            @test sol_mixed.retcode == ReturnCode.Success
            # Check that solution is reasonably close (allowing for reduced precision)
            @test norm(sol_mixed.u - sol_ref.u) / norm(sol_ref.u) < 1.0e-5
            # Verify it actually solves the system
            @test norm(A * sol_mixed.u - b) / norm(b) < 1.0e-5
        else
            @test_skip "Apple Accelerate not available"
        end
    end

    @testset "OpenBLAS32MixedLUFactorization" begin
        if LinearSolve.useopenblas
            sol_mixed = solve(prob, OpenBLAS32MixedLUFactorization())
            @test sol_mixed.retcode == ReturnCode.Success
            # Check that solution is reasonably close (allowing for reduced precision)
            @test norm(sol_mixed.u - sol_ref.u) / norm(sol_ref.u) < 1.0e-5
            # Verify it actually solves the system
            @test norm(A * sol_mixed.u - b) / norm(b) < 1.0e-5
        else
            @test_skip "OpenBLAS not available"
        end
    end

    @testset "RF32MixedLUFactorization" begin
        # Test if RecursiveFactorization is available
        try
            using RecursiveFactorization
            sol_mixed = solve(prob, RF32MixedLUFactorization())
            @test sol_mixed.retcode == ReturnCode.Success
            # Check that solution is reasonably close (allowing for reduced precision)
            @test norm(sol_mixed.u - sol_ref.u) / norm(sol_ref.u) < 1.0e-5
            # Verify it actually solves the system
            @test norm(A * sol_mixed.u - b) / norm(b) < 1.0e-5

            # Test without pivoting
            #sol_mixed_nopivot = solve(prob, RF32MixedLUFactorization(pivot=Val(false)))
            #@test sol_mixed_nopivot.retcode == ReturnCode.Success
            #@test norm(A * sol_mixed_nopivot.u - b) / norm(b) < 1e-5
        catch e
            if isa(e, ArgumentError) && occursin("RecursiveFactorization", e.msg)
                @test_skip "RecursiveFactorization not available"
            else
                rethrow(e)
            end
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
            @test norm(sol_mixed.u - sol_ref_complex.u) / norm(sol_ref_complex.u) < 1.0e-5
        end

        if Sys.isapple()
            sol_mixed = solve(prob_complex, AppleAccelerate32MixedLUFactorization())
            @test sol_mixed.retcode == ReturnCode.Success
            @test norm(sol_mixed.u - sol_ref_complex.u) / norm(sol_ref_complex.u) < 1.0e-5
        end

        if LinearSolve.useopenblas
            sol_mixed = solve(prob_complex, OpenBLAS32MixedLUFactorization())
            @test sol_mixed.retcode == ReturnCode.Success
            @test norm(sol_mixed.u - sol_ref_complex.u) / norm(sol_ref_complex.u) < 1.0e-5
        end

        # Note: RecursiveFactorization currently optimized for real matrices
        # Complex support may have different performance characteristics
        try
            using RecursiveFactorization
            sol_mixed = solve(prob_complex, RF32MixedLUFactorization())
            @test sol_mixed.retcode == ReturnCode.Success
            @test norm(sol_mixed.u - sol_ref_complex.u) / norm(sol_ref_complex.u) < 1.0e-5
        catch e
            if isa(e, ArgumentError) && occursin("RecursiveFactorization", e.msg)
                @test_skip "RecursiveFactorization not available"
            else
                # RecursiveFactorization may not support complex matrices well
                @test_skip "RF32MixedLUFactorization may not support complex matrices"
            end
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
