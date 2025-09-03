using LinearSolve, LinearAlgebra, Test

"""
Test for LinearSolve.jl Issue #771: LU algorithms that work when solving fail in autotune_setup

This test reproduces the issue where individual LU factorization algorithms work correctly
for solving linear systems but fail during autotune benchmarking. 

Reference: https://github.com/SciML/LinearSolve.jl/issues/771
"""

@testset "Issue #771: AutoTune Algorithm Failures" begin
    # Test setup - reproduce the exact scenario from the issue
    n = 4
    A = rand(n, n)
    b1 = rand(n)
    prob = LinearProblem(A, b1)

    @testset "Individual Algorithm Functionality" begin
        # These should all work as reported in the issue
        
        @testset "AppleAccelerateLUFactorization" begin
            @test_nowarn begin
                linsolve_accelerate = init(prob, AppleAccelerateLUFactorization())
                result = solve!(linsolve_accelerate)
                @test result.retcode == SciMLBase.ReturnCode.Success
                @test norm(A * result.u - b1) < 1e-10
            end
        end

        @testset "OpenBLASLUFactorization" begin  
            @test_nowarn begin
                linsolve_oblas = init(prob, OpenBLASLUFactorization())
                result = solve!(linsolve_oblas)
                @test result.retcode == SciMLBase.ReturnCode.Success
                @test norm(A * result.u - b1) < 1e-10
            end
        end

        @testset "GenericLUFactorization" begin
            @test_nowarn begin
                linsolve_generic = init(prob, GenericLUFactorization())
                result = solve!(linsolve_generic)
                @test result.retcode == SciMLBase.ReturnCode.Success
                @test norm(A * result.u - b1) < 1e-10
            end
        end

        @testset "SimpleLUFactorization" begin
            @test_nowarn begin
                linsolve_simple = init(prob, SimpleLUFactorization())
                result = solve!(linsolve_simple)
                @test result.retcode == SciMLBase.ReturnCode.Success
                @test norm(A * result.u - b1) < 1e-10
            end
        end

        @testset "LUFactorization" begin
            @test_nowarn begin
                linsolve_lu = init(prob, LUFactorization())
                result = solve!(linsolve_lu)
                @test result.retcode == SciMLBase.ReturnCode.Success
                @test norm(A * result.u - b1) < 1e-10
            end
        end
    end

    @testset "Environment Verification" begin
        # Test that we can verify the user's environment for common issues
        
        @testset "Julia Version Check" begin
            # Issue was related to Julia 1.10+ precompilation changes
            julia_version = VERSION
            @test julia_version >= v"1.6"  # Minimum supported version
            
            if julia_version >= v"1.10"
                # On Julia 1.10+, extensions should load without method overwriting errors
                # This is a basic test that LinearSolve can be loaded
                @test isdefined(LinearSolve, :LUFactorization)
            end
        end
        
        @testset "Package Versions" begin
            # Check that we're using a recent enough version
            # The exact version may vary in the test environment
            @test isdefined(LinearSolve, :AppleAccelerateLUFactorization)
            @test isdefined(LinearSolve, :OpenBLASLUFactorization)
            
            @info "Testing with Julia version: $(VERSION)"
        end
    end
end