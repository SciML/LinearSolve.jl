using LinearSolve, LinearAlgebra, SparseArrays, Test, Random

# `LowRankUpdatedMatrix` carries `A + U C V'`, so the update is part of the problem rather
# than the algorithm. The solve factorizes `A` once and applies the Woodbury identity.
# See SciML/LinearSolve.jl#136.
@testset "Low-rank updated matrix (#136)" begin
    Random.seed!(136)
    n = 40
    k = 3

    formed(A, U, V, C) = C === I ? Matrix(A) + U * V' : Matrix(A) + U * C * V'
    asmat(x, n) = x isa AbstractVector ? reshape(x, n, 1) : x

    @testset "matches the assembled matrix" begin
        for (name, U, V, C) in (
                ("rank-1 vectors", rand(n), rand(n), I),
                ("rank-k matrices", rand(n, k), rand(n, k), I),
                ("with a middle factor", rand(n, k), rand(n, k), rand(k, k) + k * I),
            )
            @testset "$name" begin
                A = rand(n, n) + n * I
                b = rand(n)
                ref = formed(A, asmat(U, n), asmat(V, n), C) \ b

                sol = solve(LinearProblem(LowRankUpdatedMatrix(A, U, V; C = C), b))
                @test sol.retcode == LinearSolve.ReturnCode.Success
                @test sol.u ≈ ref rtol = 1.0e-9
            end
        end
    end

    @testset "behaves as a matrix" begin
        A = rand(n, n) + n * I
        U = rand(n, k)
        V = rand(n, k)
        M = LowRankUpdatedMatrix(A, U, V)
        dense = A + U * V'

        @test size(M) == (n, n)
        @test size(M, 1) == n
        @test eltype(M) == Float64
        @test M[3, 7] ≈ dense[3, 7]
        x = rand(n)
        @test M * x ≈ dense * x rtol = 1.0e-10
        y = similar(x)
        @test mul!(y, M, x) ≈ dense * x rtol = 1.0e-10
    end

    @testset "rejects mismatched shapes" begin
        A = rand(n, n) + n * I
        @test_throws DimensionMismatch LowRankUpdatedMatrix(A, rand(n + 1), rand(n))
        @test_throws DimensionMismatch LowRankUpdatedMatrix(A, rand(n), rand(n + 1))
        @test_throws DimensionMismatch LowRankUpdatedMatrix(A, rand(n, 2), rand(n, 3))
    end

    @testset "the factorization is reused across right-hand sides" begin
        A = rand(n, n) + n * I
        U = rand(n, k)
        V = rand(n, k)
        dense = A + U * V'

        cache = init(LinearProblem(LowRankUpdatedMatrix(A, U, V), rand(n)))
        solve!(cache)
        for _ in 1:3
            b = rand(n)
            cache.b = b
            @test solve!(cache).u ≈ dense \ b rtol = 1.0e-9
        end
    end

    # Numerical agreement alone does not show the Woodbury path ran: assembling the sum and
    # factorizing it densely gives the same answer. `__init` copies the matrix unless the
    # caller aliases it, and without a `copy` method the wrapper was materialized
    # elementwise into a dense `Matrix`, so every solve silently took the generic path.
    @testset "the low-rank path is actually taken" begin
        A = rand(n, n) + n * I
        U = rand(n, k)
        V = rand(n, k)
        M = LowRankUpdatedMatrix(A, U, V)

        @test copy(M) isa LowRankUpdatedMatrix
        @test copy(M) ≈ A + U * V'

        cache = init(LinearProblem(M, rand(n)), LUFactorization())
        @test cache.A isa LowRankUpdatedMatrix
        solve!(cache)
        @test cache.cacheval isa LinearSolve.LowRankUpdatedCache
    end

    # Every algorithm the type advertises has to reach `do_factorization`. Three of the
    # originally listed ones route through their own `solve!` instead and threw a
    # `MethodError`, so the list and the behaviour are pinned together here.
    @testset "every advertised factorization solves" begin
        spd = (X = rand(n, n); X'X + n * I)
        sym = (X = rand(n, n); Symmetric(X + X' + n * I))
        bases = Dict(
            LUFactorization => rand(n, n) + n * I,
            QRFactorization => rand(n, n) + n * I,
            SVDFactorization => rand(n, n) + n * I,
            CholeskyFactorization => spd,
            BunchKaufmanFactorization => sym,
        )
        @test Set(LinearSolve._LOWRANK_ALGS) == Set(keys(bases))

        for Alg in LinearSolve._LOWRANK_ALGS
            @testset "$(nameof(Alg))" begin
                A = bases[Alg]
                U = rand(n, k)
                V = rand(n, k)
                b = rand(n)
                cache = init(LinearProblem(LowRankUpdatedMatrix(A, U, V), b), Alg())
                sol = solve!(cache)
                @test cache.cacheval isa LinearSolve.LowRankUpdatedCache
                @test sol.u ≈ (Matrix(A) + U * V') \ b rtol = 1.0e-8
            end
        end
    end

    # A matrix-free method needs no factorization at all: `mul!` is enough, so the wrapper
    # goes straight through without ever being assembled.
    @testset "a Krylov method consumes the wrapper directly" begin
        A = rand(n, n) + n * I
        U = rand(n, k)
        V = rand(n, k)
        b = rand(n)
        M = LowRankUpdatedMatrix(A, U, V)
        for alg in (KrylovJL_GMRES(), SimpleGMRES())
            sol = solve(LinearProblem(M, b), alg)
            @test sol.u ≈ (A + U * V') \ b rtol = 1.0e-6
        end
    end

    # The point of the type: a dense update to a sparse base would otherwise assemble to a
    # dense matrix and force a dense factorization.
    @testset "a sparse base keeps its factorization" begin
        m = 200
        A = spdiagm(-1 => -ones(m - 1), 0 => 4ones(m), 1 => -ones(m - 1))
        u = rand(m)
        v = rand(m)
        b = rand(m)
        cache = init(LinearProblem(LowRankUpdatedMatrix(A, u, v), b))
        sol = solve!(cache)
        @test sol.u ≈ (Matrix(A) + u * v') \ b rtol = 1.0e-8
        # The sparse factorization of `A` is what has to survive, not just the answer.
        @test cache.cacheval.fact isa SparseArrays.UMFPACK.UmfpackLU
    end

    # An update that makes the whole matrix singular shows up as a singular capacitance
    # matrix, and has to be reported rather than returned as a wrong answer.
    @testset "a singular update reports failure" begin
        A = Matrix{Float64}(I, 4, 4)
        u = [1.0, 0, 0, 0]
        v = [-1.0, 0, 0, 0]
        sol = solve(LinearProblem(LowRankUpdatedMatrix(A, u, v), rand(4)))
        @test sol.retcode != LinearSolve.ReturnCode.Success
    end
end
