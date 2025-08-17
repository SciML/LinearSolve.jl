using Pkg
Pkg.activate(".")
using LinearSolve
using Test
using LinearAlgebra

# Test that the new algorithm choices are available in the enum
@testset "New Algorithm Choices" begin
    choices = Symbol.(instances(LinearSolve.DefaultAlgorithmChoice.T))
    println("Available choices: ", choices)
    @test :BLISLUFactorization in choices
    @test :CudaOffloadLUFactorization in choices
    @test :MetalLUFactorization in choices
end

# Test that availability checking functions exist
@testset "Availability Functions" begin
    # These should return false since the extensions aren't loaded
    @test LinearSolve.useblis() == false
    @test LinearSolve.usecuda() == false
    @test LinearSolve.usemetal() == false
    
    # Test that is_algorithm_available correctly reports availability
    @test LinearSolve.is_algorithm_available(LinearSolve.DefaultAlgorithmChoice.BLISLUFactorization) == false
    @test LinearSolve.is_algorithm_available(LinearSolve.DefaultAlgorithmChoice.CudaOffloadLUFactorization) == false
    @test LinearSolve.is_algorithm_available(LinearSolve.DefaultAlgorithmChoice.MetalLUFactorization) == false
end

# Test that the algorithms can be instantiated without extensions (with throwerror=false)
@testset "Algorithm Instantiation" begin
    # These should work with throwerror=false
    alg1 = LinearSolve.BLISLUFactorization(throwerror=false)
    @test alg1 isa LinearSolve.BLISLUFactorization
    
    alg2 = LinearSolve.CudaOffloadLUFactorization(throwerror=false)
    @test alg2 isa LinearSolve.CudaOffloadLUFactorization
    
    # Metal is only available on Apple platforms
    if Sys.isapple()
        alg3 = LinearSolve.MetalLUFactorization(throwerror=false)
        @test alg3 isa LinearSolve.MetalLUFactorization
    else
        # On non-Apple platforms, it should still not error with throwerror=false
        alg3 = LinearSolve.MetalLUFactorization(throwerror=false)
        @test alg3 isa LinearSolve.MetalLUFactorization
    end
    
    # These should throw errors with throwerror=true (default)
    @test_throws ErrorException LinearSolve.BLISLUFactorization()
    @test_throws ErrorException LinearSolve.CudaOffloadLUFactorization()
    
    # Metal error message depends on platform
    if Sys.isapple()
        @test_throws ErrorException LinearSolve.MetalLUFactorization()
    else
        # On non-Apple platforms, should error with platform message
        @test_throws ErrorException LinearSolve.MetalLUFactorization()
    end
end

# Test that preferences system recognizes the new algorithms
@testset "Preferences Support" begin
    # Test that the preference string mapping works
    alg = LinearSolve._string_to_algorithm_choice("BLISLUFactorization")
    @test alg === LinearSolve.DefaultAlgorithmChoice.BLISLUFactorization
    
    alg = LinearSolve._string_to_algorithm_choice("CudaOffloadLUFactorization")
    @test alg === LinearSolve.DefaultAlgorithmChoice.CudaOffloadLUFactorization
    
    alg = LinearSolve._string_to_algorithm_choice("MetalLUFactorization")
    @test alg === LinearSolve.DefaultAlgorithmChoice.MetalLUFactorization
end

# Test basic solve still works with DefaultLinearSolver
@testset "Default Solver Still Works" begin
    A = rand(10, 10)
    b = rand(10)
    prob = LinearProblem(A, b)
    
    # Should use default solver and work fine
    sol = solve(prob)
    @test sol.retcode == ReturnCode.Success
    @test norm(A * sol.u - b) < 1e-10
end

println("All tests passed!")