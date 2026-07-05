using LinearSolve, LinearAlgebra, SparseArrays, Test
using StableRNGs

rng = StableRNG(123)

@testset "GESVFactorization" begin
    @testset "vector and matrix RHS" begin
        n = 20
        A = rand(rng, n, n) + 5I
        b = rand(rng, n)
        B = rand(rng, n, 4)

        sol = solve(LinearProblem(A, b), GESVFactorization())
        @test SciMLBase.successful_retcode(sol.retcode)
        @test sol.u ≈ A \ b

        solm = solve(LinearProblem(A, B), GESVFactorization())
        @test SciMLBase.successful_retcode(solm.retcode)
        @test solm.u ≈ A \ B
    end

    @testset "cache reuse: new b (getrs! path) and new A (refactorization)" begin
        n = 20
        A1 = rand(rng, n, n) + 5I
        A2 = rand(rng, n, n) + 5I
        b1 = rand(rng, n)
        b2 = rand(rng, n)

        cache = init(LinearProblem(copy(A1), copy(b1)), GESVFactorization())
        sol1 = solve!(cache)
        @test sol1.u ≈ A1 \ b1

        cache.b = copy(b2)
        sol2 = solve!(cache)
        @test sol2.u ≈ A1 \ b2

        cache.A = copy(A2)
        sol3 = solve!(cache)
        @test sol3.u ≈ A2 \ b2

        # matrix RHS through the same cache-reuse paths
        B1 = rand(rng, n, 3)
        B2 = rand(rng, n, 3)
        mcache = init(LinearProblem(copy(A1), copy(B1)), GESVFactorization())
        @test solve!(mcache).u ≈ A1 \ B1
        mcache.b = copy(B2)
        @test solve!(mcache).u ≈ A1 \ B2
        mcache.A = copy(A2)
        @test solve!(mcache).u ≈ A2 \ B2
    end

    @testset "alias_A semantics" begin
        n = 12
        A = rand(rng, n, n) + 5I
        b = rand(rng, n)

        # alias_A = false: the user's matrix must stay pristine
        Akeep = copy(A)
        cache = init(
            LinearProblem(A, copy(b)), GESVFactorization();
            alias = LinearAliasSpecifier(alias_A = false, alias_b = false)
        )
        sol = solve!(cache)
        @test sol.u ≈ Akeep \ b
        @test A == Akeep

        # alias_A = true: A may be overwritten by the factors, and the solve is
        # still correct
        A2 = rand(rng, n, n) + 5I
        A2keep = copy(A2)
        cache2 = init(
            LinearProblem(A2, copy(b)), GESVFactorization();
            alias = LinearAliasSpecifier(alias_A = true, alias_b = false)
        )
        sol2 = solve!(cache2)
        @test sol2.u ≈ A2keep \ b
    end

    @testset "singular matrix returns Failure retcode without throwing" begin
        n = 6
        As = zeros(n, n)
        b = rand(rng, n)
        sol = solve(LinearProblem(As, b), GESVFactorization())
        @test !SciMLBase.successful_retcode(sol.retcode)
        @test sol.retcode == ReturnCode.Failure

        # a cache whose fresh solve failed can be reused with a new nonsingular A
        cache = init(LinearProblem(zeros(n, n), copy(b)), GESVFactorization())
        @test !SciMLBase.successful_retcode(solve!(cache).retcode)
        Aok = rand(rng, n, n) + 5I
        cache.A = copy(Aok)
        solok = solve!(cache)
        @test SciMLBase.successful_retcode(solok.retcode)
        @test solok.u ≈ Aok \ b
    end

    @testset "allocations: warm b-only re-solve is allocation-free" begin
        n = 100
        A = rand(rng, n, n) + 10I
        b = rand(rng, n)
        bnew = rand(rng, n)

        resolve_b!(cache, bv) = (copyto!(cache.b, bv); solve!(cache); nothing)
        refactor!(cache, Am) = (copyto!(cache.A, Am); cache.A = cache.A; solve!(cache); nothing)

        cache = init(
            LinearProblem(copy(A), copy(b)), GESVFactorization();
            alias = LinearAliasSpecifier(alias_A = true, alias_b = true)
        )
        solve!(cache)
        resolve_b!(cache, bnew)
        @test (@allocated resolve_b!(cache, bnew)) == 0
        @test cache.u ≈ A \ bnew

        # fresh-A re-solve stays O(n): gesv! allocates only its pivot vector
        refactor!(cache, A)
        alloc = @allocated refactor!(cache, A)
        @test alloc < 20000
        @test cache.u ≈ A \ bnew
    end

    @testset "unsupported matrix types throw an informative error at init" begin
        A = sprand(rng, 10, 10, 0.5) + 5I
        b = rand(rng, 10)
        @test_throws ArgumentError init(LinearProblem(A, b), GESVFactorization())
        Abig = rand(rng, BigFloat, 4, 4) + 5I
        bbig = rand(rng, BigFloat, 4)
        @test_throws ArgumentError init(LinearProblem(Abig, bbig), GESVFactorization())
    end
end
