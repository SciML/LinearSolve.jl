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
# The other sizes are padded to 8.
@testset "Cache reuse: n = $n" for n in (8, 5, 6, 7)
    A1 = rand(n, n) + n * I
    A2 = rand(n, n) + n * I
    b1 = rand(n)
    b2 = rand(n)
    cache = init(LinearProblem(A1, b1), ButterflyFactorization())
    sol = solve!(cache)
    @test sol.u ≈ A1 \ b1
    @test cache.u == sol.u
    @test cache.b == b1
    # a new `b`, same factorization
    cache.b = b2
    @test solve!(cache).u ≈ A1 \ b2
    # a new `A`
    cache.A = A2
    @test solve!(cache).u ≈ A2 \ b2
    # the same array again, refilled in place
    copyto!(cache.A, A1)
    cache.A = cache.A
    @test solve!(cache).u ≈ A1 \ b2
end

@testset "alias_A: n = $n" for n in (8, 6)
    A1 = rand(n, n) + n * I
    A2 = rand(n, n) + n * I
    b = rand(n)
    x1 = A1 \ b
    x2 = A2 \ b

    # Default: the matrix given to `init` and one assigned later are left as they are.
    A = copy(A1)
    cache = init(LinearProblem(A, b), ButterflyFactorization())
    @test solve!(cache).u ≈ x1
    @test A == A1
    A = copy(A2)
    cache.A = A
    @test solve!(cache).u ≈ x2
    @test A == A2

    # `alias_A = true`: a matrix that needs no padding is factorized where it is.
    A = copy(A1)
    cache = init(
        LinearProblem(A, b), ButterflyFactorization();
        alias = LinearAliasSpecifier(alias_A = true)
    )
    @test solve!(cache).u ≈ x1
    @test (A != A1) == (n == 8)
    A = copy(A2)
    cache.A = A
    @test solve!(cache).u ≈ x2
    @test (A != A2) == (n == 8)
end

@testset "Padding as in RecursiveFactorization: n = $n" for n in (5, 6, 7)
    pad! = Base.get_extension(LinearSolve, :LinearSolveRecursiveFactorizationExt).pad!
    A = rand(n, n)
    @test pad!(rand(8, 8), A) == RecursiveFactorization.pad!(A)
end

# `n = 102` is padded to 104. A new workspace would allocate several matrices.
@testset "Refactorization allocates no matrix: n = $n" for n in (100, 102)
    A1 = rand(n, n) + n * I
    A2 = rand(n, n) + n * I
    b = rand(n)
    cache = init(LinearProblem(A1, b), ButterflyFactorization())
    solve!(cache)
    refactorize!(cache, A) = (cache.A = A; solve!(cache))
    @test refactorize!(cache, A2).u ≈ A2 \ b
    @test @allocated(refactorize!(cache, A1)) < sizeof(A1)
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
