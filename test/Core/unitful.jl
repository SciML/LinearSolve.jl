using LinearSolve, Unitful, LinearAlgebra, Test, Random

Random.seed!(173)

# `__init_u0_from_Ab` used to build the default `u0` with `fill!(u0, false)`.
# `false` is a strong zero for the usual number types, but it is not a valid
# element of a Unitful array, so `init` threw `DimensionError: nm and false are
# not dimensionally compatible` before any solver ran. See
# SciML/LinearSolve.jl#173.
@testset "Unitful right-hand side (#173)" begin
    # A dimensionless system matrix acting on a dimensioned right-hand side.
    # The units work out here, so this solves end to end and the solution
    # carries the units of `b`.
    A = rand(4, 4) + 4I
    b = rand(4)u"nm"
    xref = (A \ ustrip.(u"nm", b))u"nm"

    @testset "init keeps the units of b" begin
        cache = init(LinearProblem(A, b))
        @test eltype(cache.u) === eltype(b)
        @test all(iszero, cache.u)
        @test unit(eltype(cache.u)) == u"nm"
    end

    for alg in (nothing, LUFactorization(), GenericLUFactorization())
        @testset "$(alg === nothing ? "default" : nameof(typeof(alg)))" begin
            sol = alg === nothing ? solve(LinearProblem(A, b)) :
                solve(LinearProblem(A, b), alg)
            @test SciMLBase.successful_retcode(sol)
            @test unit(eltype(sol.u)) == u"nm"
            @test ustrip.(u"nm", sol.u) ≈ ustrip.(u"nm", xref) rtol = 1.0e-10
            @test ustrip.(u"nm", A * sol.u .- b) ≈ zeros(4) atol = 1.0e-12
        end
    end

    # A matrix right-hand side takes the other `__init_u0_from_Ab` method.
    @testset "matrix right-hand side" begin
        B = rand(4, 2)u"nm"
        cache = init(LinearProblem(A, B))
        @test eltype(cache.u) === eltype(B)
        @test all(iszero, cache.u)
        sol = solve(LinearProblem(A, B))
        @test ustrip.(u"nm", sol.u) ≈ A \ ustrip.(u"nm", B) rtol = 1.0e-10
    end
end
