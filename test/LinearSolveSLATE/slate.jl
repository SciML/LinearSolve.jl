using LinearSolve, LinearAlgebra, Random, SciMLBase, Test
using SLATE_jll

# `test/Core/slate.jl` only checks that an unconfigured `SLATEFactorization` reports
# itself clearly. This group points the wrapper at the library SLATE_jll ships and
# actually solves with it, so the numerics are covered rather than skipped.
# SLATE_jll is Linux-only, so on any other platform there is nothing to run.
if !SLATE_jll.is_available()
    @info "SLATE_jll has no build for this platform; skipping the SLATE solver tests"
    @testset "SLATE unavailable" begin
        @test !LinearSolve.slate_isavailable()
    end
else
    ENV["SLATE_LAPACK_LIB"] = SLATE_jll.libslate_lapack_api

    @testset "SLATE against the library SLATE_jll ships" begin
        @test LinearSolve.slate_isavailable()

        Random.seed!(54)

        # The wrapper dispatches `slate_sgesv`/`dgesv`/`cgesv`/`zgesv` on the element
        # type, so each one needs its own solve to be covered at all.
        @testset "eltype $T" for T in (Float64, Float32, ComplexF64, ComplexF32)
            n = 40
            A = rand(T, n, n) + n * I
            b = rand(T, n)
            ref = Matrix(A) \ b
            tol = real(T) === Float32 ? 1.0f-3 : 1.0e-9

            sol = solve(LinearProblem(copy(A), copy(b)), SLATEFactorization())
            @test SciMLBase.successful_retcode(sol)
            @test sol.u ≈ ref rtol = tol
        end

        @testset "multiple right-hand sides" begin
            n = 30
            A = rand(n, n) + n * I
            B = rand(n, 3)
            sol = solve(LinearProblem(copy(A), copy(B)), SLATEFactorization())
            @test SciMLBase.successful_retcode(sol)
            @test sol.u ≈ Matrix(A) \ B rtol = 1.0e-9
        end

        @testset "explicit libpath is honoured" begin
            n = 20
            A = rand(n, n) + n * I
            b = rand(n)
            alg = SLATEFactorization(libpath = SLATE_jll.libslate_lapack_api)
            sol = solve(LinearProblem(copy(A), copy(b)), alg)
            @test SciMLBase.successful_retcode(sol)
            @test sol.u ≈ Matrix(A) \ b rtol = 1.0e-9
        end
    end
end
