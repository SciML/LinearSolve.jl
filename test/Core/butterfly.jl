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

# `n = 8` is a multiple of 4, which the butterfly workspace takes without padding.
# `n = 6` is padded to 8.
@testset "Cache reuse: n = $n, alias_A = $alias_A" for n in (8, 6), alias_A in (false, true)
    A1 = rand(n, n) + n * I
    A2 = rand(n, n) + n * I
    b = rand(n)
    cache = init(
        LinearProblem(copy(A1), copy(b)), ButterflyFactorization();
        alias = LinearAliasSpecifier(; alias_A)
    )
    sol = solve!(cache)
    @test sol.u ≈ A1 \ b
    @test cache.u == sol.u
    @test cache.b == b
    @test solve!(cache).u ≈ A1 \ b
    A2c = copy(A2)
    cache.A = A2c
    @test solve!(cache).u ≈ A2 \ b
    # without `alias_A` an assigned matrix is left as it is
    alias_A || @test A2c == A2
    # the same array again, refilled in place
    copyto!(cache.A, A1)
    cache.A = cache.A
    @test solve!(cache).u ≈ A1 \ b
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
