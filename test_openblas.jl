using LinearAlgebra
using LinearSolve
using Test

@testset "OpenBLASLUFactorization Tests" begin
    # Test with Float64
    @testset "Float64" begin
        A = rand(10, 10)
        b = rand(10)
        prob = LinearProblem(A, b)
        
        sol_openblas = solve(prob, OpenBLASLUFactorization())
        sol_default = solve(prob, LUFactorization())
        
        @test norm(A * sol_openblas.u - b) < 1e-10
        @test norm(sol_openblas.u - sol_default.u) < 1e-10
    end
    
    # Test with Float32
    @testset "Float32" begin
        A = rand(Float32, 10, 10)
        b = rand(Float32, 10)
        prob = LinearProblem(A, b)
        
        sol_openblas = solve(prob, OpenBLASLUFactorization())
        sol_default = solve(prob, LUFactorization())
        
        @test norm(A * sol_openblas.u - b) < 1e-5
        @test norm(sol_openblas.u - sol_default.u) < 1e-5
    end
    
    # Test with ComplexF64
    @testset "ComplexF64" begin
        A = rand(ComplexF64, 10, 10)
        b = rand(ComplexF64, 10)
        prob = LinearProblem(A, b)
        
        sol_openblas = solve(prob, OpenBLASLUFactorization())
        sol_default = solve(prob, LUFactorization())
        
        @test norm(A * sol_openblas.u - b) < 1e-10
        @test norm(sol_openblas.u - sol_default.u) < 1e-10
    end
    
    # Test with ComplexF32
    @testset "ComplexF32" begin
        A = rand(ComplexF32, 10, 10)
        b = rand(ComplexF32, 10)
        prob = LinearProblem(A, b)
        
        sol_openblas = solve(prob, OpenBLASLUFactorization())
        sol_default = solve(prob, LUFactorization())
        
        @test norm(A * sol_openblas.u - b) < 1e-5
        @test norm(sol_openblas.u - sol_default.u) < 1e-5
    end
end

println("All tests passed!")