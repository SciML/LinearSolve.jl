using LinearSolve, LinearAlgebra, SparseArrays, Test, Random

# `LowRankUpdatedFactorization` solves `(A + U C V') x = b` by factorizing `A` once and
# applying the Woodbury identity, so a low-rank change never refactorizes the full matrix.
# See SciML/LinearSolve.jl#136.
@testset "Low-rank updated factorization (#136)" begin
    Random.seed!(136)
    n = 40
    k = 3

    @testset "matches the formed matrix" begin
        for (name, U, V, C) in (
                ("rank-1 vectors", rand(n), rand(n), I),
                ("rank-k matrices", rand(n, k), rand(n, k), I),
                ("with a middle factor", rand(n, k), rand(n, k), rand(k, k) + k * I),
            )
            @testset "$name" begin
                A = rand(n, n) + n * I
                b = rand(n)
                Um = U isa AbstractVector ? reshape(U, n, 1) : U
                Vm = V isa AbstractVector ? reshape(V, n, 1) : V
                formed = C === I ? A + Um * Vm' : A + Um * C * Vm'

                sol = solve(
                    LinearProblem(A, b),
                    LowRankUpdatedFactorization(; U = U, V = V, C = C)
                )
                @test sol.retcode == LinearSolve.ReturnCode.Success
                @test sol.u ≈ formed \ b rtol = 1.0e-10
            end
        end
    end

    @testset "the factorization is reused across right-hand sides" begin
        A = rand(n, n) + n * I
        U = rand(n, k)
        V = rand(n, k)
        formed = A + U * V'

        cache = init(LinearProblem(A, rand(n)), LowRankUpdatedFactorization(; U = U, V = V))
        solve!(cache)
        for _ in 1:3
            b = rand(n)
            cache.b = b
            @test solve!(cache).u ≈ formed \ b rtol = 1.0e-10
        end
    end

    @testset "a new matrix refactorizes" begin
        U = rand(n, k)
        V = rand(n, k)
        alg = LowRankUpdatedFactorization(; U = U, V = V)
        A = rand(n, n) + n * I
        b = rand(n)
        cache = init(LinearProblem(A, b), alg)
        @test solve!(cache).u ≈ (A + U * V') \ b rtol = 1.0e-10

        A2 = rand(n, n) + 2n * I
        cache.A = A2
        @test solve!(cache).u ≈ (A2 + U * V') \ b rtol = 1.0e-10
    end

    # The point of the identity: a dense update to a sparse matrix would otherwise
    # destroy the sparsity and force a dense factorization.
    @testset "a sparse base keeps its factorization" begin
        m = 200
        A = spdiagm(-1 => -ones(m - 1), 0 => 4ones(m), 1 => -ones(m - 1))
        u = rand(m)
        v = rand(m)
        b = rand(m)
        sol = solve(LinearProblem(A, b), LowRankUpdatedFactorization(; U = u, V = v))
        @test sol.u ≈ (Matrix(A) + u * v') \ b rtol = 1.0e-8
    end

    # A singular capacitance matrix is where an update that makes the whole system
    # singular shows up, and it must be reported rather than returning a wrong answer.
    @testset "a singular update reports failure" begin
        A = Matrix{Float64}(I, 4, 4)
        u = [1.0, 0, 0, 0]
        v = [-1.0, 0, 0, 0]   # A + u*v' is singular in the first entry
        sol = solve(
            LinearProblem(A, rand(4)), LowRankUpdatedFactorization(; U = u, V = v)
        )
        @test sol.retcode != LinearSolve.ReturnCode.Success
    end
end
