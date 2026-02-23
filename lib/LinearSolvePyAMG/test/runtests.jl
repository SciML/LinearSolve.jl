using Test
using LinearAlgebra
using SparseArrays
using LinearSolve
using LinearSolvePyAMG
using SciMLBase: ReturnCode

# Helper: 1-D Poisson-like SPD tridiagonal system
function poisson1d(n)
    A = spdiagm(-1 => -ones(n - 1), 0 => 2ones(n), 1 => -ones(n - 1))
    b = rand(n)
    return Float64.(A), b
end

@testset "LinearSolvePyAMG" begin
    @testset "constructors" begin
        # Default constructor
        alg = PyAMGJL()
        @test alg.method === :RugeStuben
        @test alg.accel === nothing

        # Named method
        alg2 = PyAMGJL(method = :SmoothedAggregation)
        @test alg2.method === :SmoothedAggregation

        # Accelerated
        alg3 = PyAMGJL(accel = "cg")
        @test alg3.accel == "cg"

        # Convenience constructors
        @test PyAMGJL_RugeStuben().method === :RugeStuben
        @test PyAMGJL_SmoothedAggregation().method === :SmoothedAggregation

        # Bad method throws
        @test_throws ArgumentError PyAMGJL(method = :BadMethod)
    end

    @testset "RugeStuben – plain V-cycle" begin
        A, b = poisson1d(100)
        prob = LinearProblem(A, b)
        sol = solve(prob, PyAMGJL())

        @test sol.retcode == ReturnCode.Success
        @test norm(A * sol.u - b) / norm(b) < 1.0e-5
    end

    @testset "SmoothedAggregation – plain V-cycle" begin
        A, b = poisson1d(100)
        prob = LinearProblem(A, b)
        sol = solve(prob, PyAMGJL_SmoothedAggregation())

        @test sol.retcode == ReturnCode.Success
        @test norm(A * sol.u - b) / norm(b) < 1.0e-5
    end

    @testset "RugeStuben – CG acceleration" begin
        A, b = poisson1d(100)
        prob = LinearProblem(A, b)
        sol = solve(prob, PyAMGJL(accel = "cg"))

        @test sol.retcode == ReturnCode.Success
        @test norm(A * sol.u - b) / norm(b) < 1.0e-8
    end

    @testset "RugeStuben – GMRES acceleration" begin
        A, b = poisson1d(100)
        prob = LinearProblem(A, b)
        sol = solve(prob, PyAMGJL(accel = "gmres"))

        @test sol.retcode == ReturnCode.Success
        @test norm(A * sol.u - b) / norm(b) < 1.0e-6
    end

    @testset "re-solve with different b (same A)" begin
        A, b1 = poisson1d(80)
        b2 = rand(80)
        prob = LinearProblem(A, b1)
        cache = init(prob, PyAMGJL(accel = "cg"))

        sol1 = solve!(cache)
        @test norm(A * sol1.u - b1) / norm(b1) < 1.0e-8

        reinit!(cache; b = b2)
        sol2 = solve!(cache)
        @test norm(A * sol2.u - b2) / norm(b2) < 1.0e-8
    end

    @testset "reinit with new A rebuilds hierarchy" begin
        A1, b = poisson1d(60)
        # A2 is a scaled version (still SPD)
        A2 = 2.0 * A1
        prob = LinearProblem(A1, b)
        cache = init(prob, PyAMGJL())

        sol1 = solve!(cache)
        @test norm(A1 * sol1.u - b) / norm(b) < 1.0e-5

        reinit!(cache; A = A2, b = A2 * sol1.u)  # consistent rhs
        sol2 = solve!(cache)
        @test norm(A2 * sol2.u - cache.b) / norm(cache.b) < 1.0e-5
    end
end
