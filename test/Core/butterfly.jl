using LinearAlgebra, LinearSolve, Random
using Test
using RecursiveFactorization

# `ButterflyFactorization` randomizes the matrix before factorizing, drawing from the
# global RNG, and the right hand sides below are random too, so seed for reproducibility.
Random.seed!(0x0b0776e5)

# The residual itself scales with `norm(A) * norm(x)`, which grows with `n` and with the
# conditioning of `A`, so an absolute bound on it is not a stable claim about the solver.
# The Wilkinson set below asserted `norm(A * x - b) <= 1e-9` and sat only ~200x under it,
# close enough to fail intermittently on CI. What a backward stable factorization actually
# bounds is the normwise backward error, so assert that.
backward_error(A, x, b) = norm(A * x .- b) / (norm(A) * norm(x) + norm(b))

@testset "Random Matrices" begin
    for i in 490:510
        A = rand(i, i)
        b = rand(i)
        prob = LinearProblem(A, b)
        x = solve(prob, ButterflyFactorization())
        @test backward_error(A, x, b) <= 1.0e-8
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
        @test backward_error(A, x, b) <= 1.0e-11
    end
end
