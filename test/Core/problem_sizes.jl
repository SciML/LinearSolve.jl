using LinearSolve, LinearAlgebra, Random, SparseArrays, StaticArrays, Test
using RecursiveFactorization

matrix_for(alg, n, m = n) = alg isa KLUFactorization ? sparse(rand(n, m)) : rand(n, m)

function check_bad_b_rejection(alg, A, b, n)
    cache = init(LinearProblem(A, b), alg)
    solve!(cache)
    old_b = cache.b
    old_u = cache.u
    old_isfresh = cache.isfresh
    bad_b = rand(n)
    @test_throws DimensionMismatch (cache.b = bad_b)
    @test cache.b === old_b
    @test cache.u === old_u
    @test cache.isfresh === old_isfresh
    return nothing
end

function check_bad_A_rejection(alg, A, b, bad_A)
    cache = init(LinearProblem(A, b), alg)
    solve!(cache)
    old_A = cache.A
    old_b = cache.b
    old_isfresh = cache.isfresh
    @test_throws DimensionMismatch (cache.A = bad_A)
    @test cache.A === old_A
    @test cache.b === old_b
    @test cache.isfresh === old_isfresh
    return nothing
end

@testset "Linear problem and cache dimensions" begin
    rng = MersenneTwister(1377)
    A = rand(rng, 10, 10) + 10I
    b = rand(rng, 10)

    for alg in (
            nothing, GenericLUFactorization(), LUFactorization(),
            RFLUFactorization(), QRFactorization(), KLUFactorization(),
        )
        @testset "$(nameof(typeof(alg)))" begin
            AA = alg isa KLUFactorization ? sparse(A) : A

            @test_throws DimensionMismatch init(LinearProblem(AA, rand(9)), alg)
            @test_throws DimensionMismatch init(LinearProblem(AA, rand(11)), alg)
            @test_throws DimensionMismatch init(LinearProblem(AA, rand(9, 2)), alg)

            check_bad_b_rejection(alg, AA, b, 9)
            check_bad_b_rejection(alg, AA, b, 11)
            check_bad_A_rejection(alg, AA, b, matrix_for(alg, 12))
            check_bad_A_rejection(alg, AA, b, matrix_for(alg, 9))
            check_bad_A_rejection(alg, AA, b, matrix_for(alg, 10, 11))

            cache = init(LinearProblem(AA, b), alg)
            cache.A = copy(AA)
            cache.b = copy(b)
            @test solve!(cache).u ≈ Matrix(AA) \ b

            cache = init(LinearProblem(AA, rand(10, 2)), alg)
            @test_throws DimensionMismatch (cache.b = rand(11, 2))
            B = rand(10, 3)
            cache.b = B
            cache.u = zeros(10, 3)
            @test size(cache.b) == (10, 3)
            @test solve!(cache).u ≈ Matrix(AA) \ B

            # `reinit!` explicitly replaces the problem and may change its size.
            cache = init(LinearProblem(AA, b), alg)
            A_new = matrix_for(alg, 12)
            b_new = rand(12)
            expected = Matrix(A_new) \ b_new
            reinit!(cache; A = copy(A_new), b = b_new, u = zeros(12))
            @test size(cache.A) == (12, 12)
            @test size(cache.b) == (12,)
            @test size(cache.u) == (12,)
            @test solve!(cache).u ≈ expected
            old_A, old_b, old_u = cache.A, cache.b, cache.u
            old_isfresh = cache.isfresh
            @test_throws DimensionMismatch reinit!(cache; A = matrix_for(alg, 13), b = b_new)
            @test cache.A === old_A
            @test cache.b === old_b
            @test cache.u === old_u
            @test cache.isfresh === old_isfresh

            # `resize!` explicitly announces a new problem size. Either A or b
            # may be assigned first, but a third size must still be rejected.
            cache = init(LinearProblem(AA, b), alg)
            resize!(cache, 12)
            cache.b = rand(12)
            @test_throws DimensionMismatch (cache.A = copy(AA))
            cache.A = matrix_for(alg, 12) + 12I
            cache.u = zeros(12)
            @test size(cache.A) == (12, 12)
            @test size(cache.b) == (12,)
            @test size(cache.u) == (12,)
        end
    end

    @testset "reinit! prepares conversions before mutation" begin
        cache = init(LinearProblem(A, b))
        solve!(cache)
        old_A, old_b, old_u = cache.A, cache.b, cache.u
        old_isfresh = cache.isfresh
        @test_throws MethodError reinit!(
            cache; A = rand(12, 12), b = rand(12), u = fill("invalid", 12)
        )
        @test cache.A === old_A
        @test cache.b === old_b
        @test cache.u === old_u
        @test cache.isfresh === old_isfresh
        @test getfield(cache, :resize_state) == 0
    end

    @testset "StaticArrays solve and cache paths" begin
        As = SMatrix{10, 10}(A)
        bs = SVector{10}(b)
        short_b = SVector{9}(rand(9))
        @test_throws DimensionMismatch solve(LinearProblem(As, short_b))
        @test_throws DimensionMismatch solve(LinearProblem(As, short_b), LUFactorization())

        cache = init(LinearProblem(As, bs), LUFactorization())
        @test_throws DimensionMismatch (cache.b = short_b)
        @test_throws DimensionMismatch (cache.A = SMatrix{12, 12}(rand(12, 12)))
        @test size(cache.b) == (10,)
        @test size(cache.A) == (10, 10)
    end
end
