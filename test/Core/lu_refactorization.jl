using LinearSolve, LinearAlgebra, StableRNGs, Test

# Warm dense-LU refactorizations (`cache.A = X; solve!(cache)`) with
# `alias_A = true` must reuse the cacheval's pivot buffer instead of allocating
# a fresh `ipiv`/`LU` wrapper through `lu!` on every call. The LAPACK
# (`getrf!`) reuse path needs Julia >= 1.11; the generic-kernel path (NoPivot /
# non-BLAS eltypes) is allocation-free on every supported version.
const BLAS_IPIV_REUSE = VERSION >= v"1.11"

rng = StableRNG(42)

function refactor_solve!(cache, Awork, A)
    copyto!(Awork, A)
    cache.A = Awork
    return solve!(cache)
end

@testset "dense LU refactorization reuses pivot buffers" begin
    n = 51
    A1 = rand(rng, n, n) + n * I
    A2 = rand(rng, n, n) + n * I
    b = rand(rng, n)

    # `maxalloc` is the per-refactorization allocation ceiling: 0 wherever the
    # reuse path is active. On Julia 1.10 the BLAS (`getrf!`-with-`ipiv`)
    # method does not exist, so those paths keep the allocating `lu!` (no
    # assertion); the generic NoPivot kernel still reuses the pivot there, but
    # 1.10's compiler does not always elide the small `LU`/solution wrapper
    # constructions that 1.11+ removes, hence the small nonzero ceiling.
    @testset "$(name)" for (name, alg, maxalloc) in (
            ("LUFactorization RowMaximum", LUFactorization(), BLAS_IPIV_REUSE ? 0 : nothing),
            (
                "LUFactorization NoPivot", LUFactorization(pivot = NoPivot()),
                VERSION >= v"1.11" ? 0 : 64,
            ),
            ("default algorithm", nothing, BLAS_IPIV_REUSE ? 0 : nothing),
        )
        cache = init(
            LinearProblem(copy(A1), copy(b)), alg;
            alias = LinearAliasSpecifier(alias_A = true)
        )
        Awork = cache.A
        @test solve!(cache).u ≈ A1 \ b

        # correctness across repeated refactorize+solve cycles
        for Ak in (A2, A1, A2)
            sol = refactor_solve!(cache, Awork, Ak)
            @test sol.retcode == ReturnCode.Success
            @test sol.u ≈ Ak \ b
        end

        # post-warmup refactorization must not allocate beyond the ceiling,
        # and the cached pivot vector must be reused, not replaced
        refactor_solve!(cache, Awork, A1)
        ipiv_before = alg isa LUFactorization ? cache.cacheval.ipiv : nothing
        alloc = @allocated refactor_solve!(cache, Awork, A2)
        if maxalloc !== nothing
            @test alloc <= maxalloc
            if alg isa LUFactorization
                @test cache.cacheval.ipiv === ipiv_before
            end
        end
        @test cache.u ≈ A2 \ b
    end
end

@testset "singular refactorization reports Failure without throwing" begin
    n = 8
    Agood = rand(rng, n, n) + n * I
    Asing = copy(Agood)
    Asing[:, 1] .= 0 # exactly rank-deficient, singular under any pivoting
    b = rand(rng, n)

    @testset "$(name)" for (name, alg) in (
            ("RowMaximum", LUFactorization()),
            ("NoPivot", LUFactorization(pivot = NoPivot())),
        )
        cache = init(
            LinearProblem(copy(Agood), copy(b)), alg;
            alias = LinearAliasSpecifier(alias_A = true)
        )
        Awork = cache.A
        @test solve!(cache).retcode == ReturnCode.Success

        # warm cycle so the singular refactorization exercises the reuse path
        refactor_solve!(cache, Awork, Agood)
        @test refactor_solve!(cache, Awork, Asing).retcode == ReturnCode.Failure

        # the cache recovers on the next nonsingular refactorization
        sol = refactor_solve!(cache, Awork, Agood)
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ Agood \ b
    end
end

@testset "default algorithm QR safety fallback survives warm singular refactorization" begin
    n = 8
    Agood = rand(rng, n, n) + n * I
    Asing = copy(Agood)
    Asing[:, 1] .= 0
    b = rand(rng, n)

    cache = init(
        LinearProblem(copy(Agood), copy(b)), nothing;
        alias = LinearAliasSpecifier(alias_A = true)
    )
    Awork = cache.A
    @test solve!(cache).retcode == ReturnCode.Success
    refactor_solve!(cache, Awork, Agood)

    # LU fails on the singular A, so the default algorithm's safetyfallback
    # rescues through column-pivoted QR (least-squares) instead of erroring
    sol = refactor_solve!(cache, Awork, Asing)
    @test sol.retcode == ReturnCode.Success
    @test Asing * sol.u ≈ Asing * (qr(Asing, ColumnNorm()) \ b)

    sol = refactor_solve!(cache, Awork, Agood)
    @test sol.retcode == ReturnCode.Success
    @test sol.u ≈ Agood \ b
end

@testset "generic (non-BLAS) eltype reuses the cached pivot" begin
    n = 6
    A1 = big.(rand(rng, n, n)) + n * I
    A2 = big.(rand(rng, n, n)) + n * I
    b = big.(rand(rng, n))

    cache = init(
        LinearProblem(copy(A1), copy(b)), LUFactorization();
        alias = LinearAliasSpecifier(alias_A = true)
    )
    Awork = cache.A
    @test solve!(cache).u ≈ A1 \ b
    for Ak in (A2, A1, A2)
        sol = refactor_solve!(cache, Awork, Ak)
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ Ak \ b
    end
end

@testset "size change between refactorizations reallocates the pivot" begin
    n1, n2 = 5, 9
    A1 = rand(rng, n1, n1) + n1 * I
    b1 = rand(rng, n1)

    cache = init(
        LinearProblem(copy(A1), copy(b1)), LUFactorization();
        alias = LinearAliasSpecifier(alias_A = true)
    )
    @test solve!(cache).u ≈ A1 \ b1

    resize!(cache, n2)
    A2 = rand(rng, n2, n2) + n2 * I
    b2 = rand(rng, n2)
    cache.A = copy(A2)
    cache.b = copy(b2)
    cache.u = zeros(n2)
    @test solve!(cache).u ≈ A2 \ b2

    # warm cycles at the new size are allocation-free again
    Awork = cache.A
    refactor_solve!(cache, Awork, A2)
    refactor_solve!(cache, Awork, A2)
    alloc = @allocated refactor_solve!(cache, Awork, A2)
    if BLAS_IPIV_REUSE
        @test alloc == 0
    end
    @test cache.u ≈ A2 \ b2
end
