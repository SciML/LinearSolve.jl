using LinearAlgebra, LinearSolve
using Test
using RecursiveFactorization

@testset "Random Matrices" begin
    for i in 490:510
        A = rand(i, i)
        b = rand(i)
        prob = LinearProblem(A, b)
        x = solve(prob, ButterflyFactorization())
        @test norm(A * x .- b) <= 5.0e-5
    end
end

@testset "Cached adjoint solve" begin
    n = 16
    A = rand(n, n)
    b = rand(n)
    cache = init(LinearProblem(A, b), ButterflyFactorization())
    @test LinearSolve._can_reuse_cache_factorization(cache.alg, cache.cacheval)
    solve!(cache)
    adjoint_rhs = rand(n)
    adjoint_solution = LinearSolve._adjoint_factorization_solve(
        cache.alg, cache.cacheval, cache.A, adjoint_rhs
    )
    @test adjoint(A) * adjoint_solution ≈ adjoint_rhs
end

function wilkinson(N)
    A = zeros(N, N)
    A[1:(N + 1):(N * N)] .= 1
    A[:, end] .= 1
    for n in 1:(N - 1)
        for r in (n + 1):N
            @inbounds A[r, n] = -1
        end
    end
    return A
end

@testset "Wilkinson" begin
    for i in 790:810
        A = wilkinson(i)
        b = rand(i)
        prob = LinearProblem(A, b)
        x = solve(prob, ButterflyFactorization())
        @test norm(A * x .- b) <= 1.0e-9
    end
end
