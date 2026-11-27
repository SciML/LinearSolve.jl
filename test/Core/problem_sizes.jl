using LinearSolve, LinearAlgebra, Random, SparseArrays, Test
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
            cache.b = rand(10, 3)
            @test size(cache.b) == (10, 3)

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
end
