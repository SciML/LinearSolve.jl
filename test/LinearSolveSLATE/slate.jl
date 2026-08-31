using LinearSolve, LinearAlgebra, Random, SciMLBase, Test
using SLATE_jll

# `test/Core/slate.jl` only checks that an unconfigured `SLATEFactorization` reports
# itself clearly. This group loads SLATE_jll, which is all it should take to make the
# solver usable, and then actually solves with it.
# SLATE_jll builds for Linux only, so elsewhere there is nothing to exercise.
if !SLATE_jll.is_available()
    @info "SLATE_jll has no build for this platform; skipping the SLATE solver tests"
    @testset "SLATE unavailable" begin
        @test !LinearSolve.slate_isavailable()
    end
else
    # Deliberately not set: the point is that loading SLATE_jll is sufficient on its
    # own. Setting this would test the environment variable instead of the extension.
    @assert !haskey(ENV, "SLATE_LAPACK_LIB")

    @testset "SLATE through SLATE_jll" begin
        @testset "loading the JLL is enough" begin
            @test LinearSolve._SLATE_JLL_LIBPATH[] == SLATE_jll.libslate_lapack_api
            @test LinearSolve.slate_isavailable()
        end

        Random.seed!(54)

        # The wrapper dispatches slate_sgesv/dgesv/cgesv/zgesv on the element type, so
        # each needs its own solve to be covered at all.
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

        # A hand-built SLATE has to keep taking precedence over the JLL, or loading
        # SLATE_jll for something else would silently redirect an explicit choice.
        @testset "an explicit libpath still wins" begin
            n = 20
            A = rand(n, n) + n * I
            b = rand(n)
            alg = SLATEFactorization(libpath = SLATE_jll.libslate_lapack_api)
            sol = solve(LinearProblem(copy(A), copy(b)), alg)
            @test SciMLBase.successful_retcode(sol)
            @test sol.u ≈ Matrix(A) \ b rtol = 1.0e-9
            @test first(LinearSolve._slate_library_candidates(SLATE_jll.libslate_lapack_api)) ==
                SLATE_jll.libslate_lapack_api
        end
    end
end
