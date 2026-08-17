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

    @testset "an explicit factorization algorithm works" begin
        A = rand(n, n) + n * I
        U = rand(n, k)
        V = rand(n, k)
        b = rand(n)
        M = LowRankUpdatedMatrix(A, U, V)
        for alg in (LUFactorization(), QRFactorization())
            @test solve(LinearProblem(M, b), alg).u ≈ (A + U * V') \ b rtol = 1.0e-9
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
        sol = solve(LinearProblem(LowRankUpdatedMatrix(A, u, v), b))
        @test sol.u ≈ (Matrix(A) + u * v') \ b rtol = 1.0e-8
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
