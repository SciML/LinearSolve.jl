using LinearSolve, LinearAlgebra, Test, Random, SciMLBase
using SpecializingFactorizations

@testset "SpecializingFactorizations extension loads" begin
    ext = Base.get_extension(LinearSolve, :LinearSolveSpecializingFactorizationsExt)
    @test ext isa Module
end

@testset "SpecializedLUFactorization" begin
    rng = Random.MersenneTwister(42)
    n = 100

    # General dense system
    A = Matrix{Float64}(I, n, n) + 0.1 * rand(rng, n, n)
    b = rand(rng, n)
    sol = solve(LinearProblem(A, b), SpecializedLUFactorization())
    @test sol.retcode == ReturnCode.Success
    @test A * sol.u ≈ b atol = 1.0e-8

    # Structured (tridiagonal) dense matrix should still solve correctly
    T = Matrix(Tridiagonal(rand(rng, n - 1), rand(rng, n) .+ 4, rand(rng, n - 1)))
    bt = rand(rng, n)
    solt = solve(LinearProblem(T, bt), SpecializedLUFactorization())
    @test solt.retcode == ReturnCode.Success
    @test T * solt.u ≈ bt atol = 1.0e-8

    # Caching: reuse the cache across a new A of the same size
    prob = LinearProblem(copy(A), copy(b))
    cache = init(prob, SpecializedLUFactorization())
    s1 = solve!(cache)
    @test A * s1.u ≈ b atol = 1.0e-8

    A2 = Matrix{Float64}(I, n, n) + 0.2 * rand(rng, n, n)
    cache.A = A2
    s2 = solve!(cache)
    @test A2 * s2.u ≈ b atol = 1.0e-8

    b2 = rand(rng, n)
    cache.b = b2
    s3 = solve!(cache)
    @test A2 * s3.u ≈ b2 atol = 1.0e-8

    # Float32 and Complex eltypes
    Af = Matrix{Float32}(I, n, n) + 0.1f0 * rand(rng, Float32, n, n)
    bf = rand(rng, Float32, n)
    solf = solve(LinearProblem(Af, bf), SpecializedLUFactorization())
    @test Af * solf.u ≈ bf atol = 1.0f-4

    Ac = Matrix{ComplexF64}(I, n, n) + 0.1 * (rand(rng, n, n) + im * rand(rng, n, n))
    bc = rand(rng, ComplexF64, n)
    solc = solve(LinearProblem(Ac, bc), SpecializedLUFactorization())
    @test Ac * solc.u ≈ bc atol = 1.0e-8
end

@testset "SpecializedQRFactorization" begin
    rng = Random.MersenneTwister(7)

    # Square system
    n = 60
    A = Matrix{Float64}(I, n, n) + 0.1 * rand(rng, n, n)
    b = rand(rng, n)
    sol = solve(LinearProblem(A, b), SpecializedQRFactorization())
    @test sol.retcode == ReturnCode.Success
    @test A * sol.u ≈ b atol = 1.0e-8

    # Overdetermined least-squares
    m, k = 100, 40
    M = randn(rng, m, k)
    c = randn(rng, m)
    solls = solve(LinearProblem(M, c), SpecializedQRFactorization())
    @test solls.retcode == ReturnCode.Success
    @test solls.u ≈ M \ c atol = 1.0e-8

    # Rank-deficient: minimum-norm least-squares (== pinv * b), never throws
    As = randn(rng, 50, 3) * randn(rng, 3, 50)
    d = randn(rng, 50)
    soldef = solve(LinearProblem(As, d), SpecializedQRFactorization())
    @test soldef.retcode == ReturnCode.Success
    @test soldef.u ≈ pinv(As) * d atol = 1.0e-6

    # Caching across a re-factor of the same shape
    prob = LinearProblem(copy(M), copy(c))
    cache = init(prob, SpecializedQRFactorization())
    s1 = solve!(cache)
    @test s1.u ≈ M \ c atol = 1.0e-8
    M2 = randn(rng, m, k)
    cache.A = M2
    s2 = solve!(cache)
    @test s2.u ≈ M2 \ c atol = 1.0e-8
end
