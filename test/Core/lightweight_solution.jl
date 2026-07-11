using LinearSolve, LinearAlgebra, SparseArrays, StableRNGs, Test

# LinearSolve 5.0: `solve!`/`solve` return a lightweight `LinearSolution` that
# no longer carries the `LinearCache` (`sol.cache === nothing`). Anyone who
# needs the cache after `solve!(cache)` already holds it. This keeps the
# warm-path wrapper elidable so repeated refactorize+solve cycles are
# allocation-free.

rng = StableRNG(7)

@testset "solve!/solve returns have cache === nothing" begin
    n = 12
    A = rand(rng, n, n) + n * I
    b = rand(rng, n)

    @testset "$(name)" for (name, alg) in (
            ("default algorithm", nothing),
            ("LUFactorization", LUFactorization()),
            ("GenericLUFactorization", GenericLUFactorization()),
            ("QRFactorization", QRFactorization()),
            ("KrylovJL_GMRES", KrylovJL_GMRES()),
            ("SVDFactorization", SVDFactorization()),
        )
        # tight reltol so the iterative solvers converge well below the
        # `isapprox` default tolerance; direct solvers ignore it
        cache = init(LinearProblem(copy(A), copy(b)), alg; reltol = 1.0e-12)
        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ A \ b
        @test sol.cache === nothing

        sol2 = solve(LinearProblem(copy(A), copy(b)), alg; reltol = 1.0e-12)
        @test sol2.cache === nothing
    end

    # sparse path
    As = sparse(A)
    sol = solve!(init(LinearProblem(As, copy(b)), KLUFactorization()))
    @test sol.u ≈ A \ b
    @test sol.cache === nothing

    # failure-path solutions are lightweight too
    Asing = zeros(n, n)
    sol = solve!(init(LinearProblem(Asing, copy(b)), LUFactorization(); verbose = false))
    @test sol.retcode == ReturnCode.Failure
    @test sol.cache === nothing

    # the default algorithm's singular -> column-pivoted QR safety fallback
    # also returns a cache-free solution
    sol = solve!(init(LinearProblem(copy(Asing), copy(b)), nothing; verbose = false))
    @test sol.cache === nothing
end

@testset "warm aliased refactorize+solve loop is allocation-free" begin
    n = 51
    A = rand(rng, n, n) + n * I
    B = rand(rng, n, n)   # matrix RHS

    cache = init(
        LinearProblem(copy(A), copy(B); u0 = zeros(n, n)), LUFactorization();
        alias = LinearAliasSpecifier(alias_A = true, alias_b = true)
    )
    solve!(cache)

    # Refresh the aliased owned buffer from A (it holds LU factors after each
    # solve) and mark it fresh, so every iteration refactorizes in place
    # before solving. Consumes the returned solution's `u` into `out` and
    # returns `nothing` so `@allocated` does not count a boxed return value.
    function warm_loop!(out, cache, Awork, A, N)
        for _ in 1:N
            copyto!(Awork, A)
            cache.A = Awork
            sol = solve!(cache)
            out[1] += sol.u[1, 1]
        end
        return nothing
    end

    Awork = cache.A
    out = zeros(1)
    warm_loop!(out, cache, Awork, A, 2)
    GC.gc()
    alloc = @allocated warm_loop!(out, cache, Awork, A, 100)
    if VERSION >= v"1.11"
        # The lightweight (cache-free) wrapper is fully elided and the LAPACK
        # `getrf!(A, ipiv)` pivot-reuse path is available: genuinely 0 bytes.
        @test alloc == 0
    else
        # Julia 1.10 lacks `LAPACK.getrf!(A, ipiv)`, so each refactorization
        # allocates a fresh `ipiv` inside `getrf!` (n * sizeof(Int) plus array
        # header), and 1.10's escape analysis does not always elide the small
        # solution wrapper. This floor is a pre-existing 1.10 limitation, not
        # a property of the solution wrapper: it is identical before and after
        # the 5.0 lightweight-solution change.
        @test alloc <= 100 * (n * sizeof(Int) + 96 + 64)
    end

    # correctness is unchanged across warm refactorizations
    for _ in 1:3
        copyto!(Awork, A)
        cache.A = Awork
        sol = solve!(cache)
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ A \ B
        @test sol.cache === nothing
    end
end
