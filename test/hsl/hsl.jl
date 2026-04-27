using HSL
using LinearSolve
using LinearAlgebra
using Random
using SparseArrays
using Test

Random.seed!(1234)
const REQUIRE_FUNCTIONAL_HSL =
    lowercase(get(ENV, "LINEARSOLVE_HSL_REQUIRE_FUNCTIONAL", "false")) == "true"

if !HSL.LIBHSL_isfunctional()
    if REQUIRE_FUNCTIONAL_HSL
        error(
            "LINEARSOLVE_HSL_REQUIRE_FUNCTIONAL=true but HSL_jll is not functional in this environment."
        )
    else
        @info "Skipping LinearSolveHSL tests: HSL_jll is not functional in this environment"
        @test true
    end
else
    @testset "HSL MA57 wrapper" begin
        n = 40
        A = sprand(Float64, n, n, 0.12)
        A = A + A' + 40.0I
        b = rand(Float64, n)
        prob = LinearProblem(A, b)

        sol = solve(prob, HSLMA57Factorization())
        @test sol.retcode == ReturnCode.Success
        @test A * sol.u ≈ b rtol = 1.0e-9 atol = 1.0e-11

        cache = init(prob, HSLMA57Factorization())
        sol1 = solve!(cache)
        cache.b = rand(Float64, n)
        b2 = copy(cache.b)
        sol2 = solve!(cache)
        @test sol1.retcode == ReturnCode.Success
        @test sol2.retcode == ReturnCode.Success
        @test A * sol2.u ≈ b2 rtol = 1.0e-9 atol = 1.0e-11
    end

    @testset "HSL MA97 wrapper" begin
        n = 35
        A = sprand(Float64, n, n, 0.1)
        A = A + A' + 30.0I
        b = rand(Float64, n)
        prob = LinearProblem(A, b)

        sol = solve(prob, HSLMA97Factorization(matrix_type = :real_spd))
        @test sol.retcode == ReturnCode.Success
        @test A * sol.u ≈ b rtol = 1.0e-9 atol = 1.0e-11

        cache = init(prob, HSLMA97Factorization(matrix_type = :real_spd))
        sol1 = solve!(cache)
        cache.b = rand(Float64, n)
        b2 = copy(cache.b)
        sol2 = solve!(cache)
        @test sol1.retcode == ReturnCode.Success
        @test sol2.retcode == ReturnCode.Success
        @test A * sol2.u ≈ b2 rtol = 1.0e-9 atol = 1.0e-11
    end
end
