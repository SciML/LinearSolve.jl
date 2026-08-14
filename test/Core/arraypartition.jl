using LinearSolve, RecursiveArrayTools, LinearAlgebra, Test, Random

# Krylov.jl allocates its workspace as `S(undef, n)` with `S = typeof(b)`, and
# `ArrayPartition(undef, n)` is defined only for a single partition, so a
# partitioned right-hand side used to fail before any iteration ran. The solve now
# runs on flat copies and writes back into the partitioned `u`.
# See SciML/LinearSolve.jl#384.
@testset "ArrayPartition right-hand side (#384)" begin
    Random.seed!(384)
    b = ArrayPartition([1.0, 2.0], [3.0, 4.0])
    A = rand(4, 4) + 4I
    reference = A \ Vector(b)

    @testset "the factorization paths were already fine" begin
        for alg in (nothing, LUFactorization(), QRFactorization())
            sol = alg === nothing ? solve(LinearProblem(A, b)) :
                solve(LinearProblem(A, b), alg)
            @test sol.retcode == LinearSolve.ReturnCode.Success
            @test Vector(sol.u) ≈ reference
        end
    end

    @testset "Krylov solvers" begin
        for (alg, M) in (
                (KrylovJL_GMRES(), A),
                (KrylovJL_BICGSTAB(), A),
                (KrylovJL(), A),
                (KrylovJL_CG(), (A + A') / 2 + 4I),
                (KrylovJL_MINRES(), (A + A') / 2 + 4I),
            )
            sol = solve(LinearProblem(M, b), alg)
            @test sol.retcode == LinearSolve.ReturnCode.Success
            @test Vector(sol.u) ≈ M \ Vector(b) rtol = 1.0e-6
            # The partitioned shape the caller passed in is what comes back.
            @test sol.u isa ArrayPartition
            @test length(sol.u.x) == length(b.x)
            @test map(length, sol.u.x) == map(length, b.x)
        end
    end

    @testset "more than two partitions, uneven" begin
        b3 = ArrayPartition([1.0], [2.0, 3.0], [4.0, 5.0, 6.0])
        A3 = rand(6, 6) + 6I
        sol = solve(LinearProblem(A3, b3), KrylovJL_GMRES())
        @test sol.retcode == LinearSolve.ReturnCode.Success
        @test Vector(sol.u) ≈ A3 \ Vector(b3) rtol = 1.0e-6
        @test map(length, sol.u.x) == (1, 2, 3)
    end

    # The flat cache is built once and reused, so updates to `b` and `A` have to
    # carry through to it rather than silently solving the original system.
    @testset "cached re-solves pick up new b and A" begin
        cache = init(LinearProblem(copy(A), copy(b)), KrylovJL_GMRES())
        @test Vector(solve!(cache).u) ≈ reference rtol = 1.0e-6

        b2 = ArrayPartition([9.0, 8.0], [7.0, 6.0])
        cache.b = b2
        @test Vector(solve!(cache).u) ≈ A \ Vector(b2) rtol = 1.0e-6

        A2 = rand(4, 4) + 8I
        cache.A = A2
        sol = solve!(cache)
        @test Vector(sol.u) ≈ A2 \ Vector(b2) rtol = 1.0e-6
        @test sol.u isa ArrayPartition
    end

    @testset "init keeps the partitioned u" begin
        cache = init(LinearProblem(A, b), KrylovJL_GMRES())
        @test cache.u isa ArrayPartition
        @test map(length, cache.u.x) == map(length, b.x)
    end
end
