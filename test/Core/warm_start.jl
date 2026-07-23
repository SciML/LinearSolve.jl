using LinearSolve, LinearAlgebra, SparseArrays, Test
using LinearSolve.Krylov

n = 100
A1 = spdiagm(
    0 => 4.0 .+ (1:n) ./ n, 1 => -ones(n - 1), -1 => -ones(n - 1),
    5 => 0.3 * ones(n - 5)
)
A2 = A1 + spdiagm(0 => 0.1 * sin.(1:n))
b1 = collect(1.0:n)
b2 = sin.(1:n) .+ 2.0
tol = 1.0e-10

@testset "warm_start validation" begin
    # a plain Symbol is no longer accepted; the option is a WarmStart enum value
    @test_throws TypeError KrylovJL_GMRES(warm_start = :hegedus)
    @test KrylovJL_GMRES().warm_start === WarmStart.Auto
    @test KrylovJL_GMRES(warm_start = WarmStart.None).warm_start === WarmStart.None
    @test KrylovJL_GMRES(warm_start = WarmStart.Previous).warm_start === WarmStart.Previous
    @test KrylovJL_FGMRES(warm_start = WarmStart.Hegedus).warm_start === WarmStart.Hegedus
end

@testset "Auto behaves as a cold start standalone" begin
    # Auto (the default) must not warm start on its own: an identical resolve
    # pays the full iteration count, exactly like None.
    for alg in (KrylovJL_GMRES(), KrylovJL_GMRES(warm_start = WarmStart.None))
        cache = init(LinearProblem(A1, b1), alg; abstol = tol, reltol = tol)
        solve!(cache)
        cache.b = copy(b1)
        solve!(cache)
        @test cache.cacheval.stats.niter > 1
    end
end

@testset "correctness across repeated solves: $(alg.KrylovAlg) warm_start=$(alg.warm_start)" for alg in (
        KrylovJL_GMRES(),
        KrylovJL_GMRES(warm_start = WarmStart.Previous),
        KrylovJL_GMRES(warm_start = WarmStart.Hegedus),
        KrylovJL_FGMRES(warm_start = WarmStart.Previous),
        KrylovJL_FGMRES(warm_start = WarmStart.Hegedus),
    )
    cache = init(LinearProblem(A1, b1), alg; abstol = tol, reltol = tol)
    sol1 = solve!(cache)
    @test norm(A1 * sol1.u - b1) < 1.0e-6

    # new RHS, same operator: previous solution is reused as the initial guess
    cache.b = b2
    sol2 = solve!(cache)
    @test norm(A1 * sol2.u - b2) < 1.0e-6

    # new operator
    cache.A = A2
    sol3 = solve!(cache)
    @test norm(A2 * sol3.u - b2) < 1.0e-6
end

@testset "warm start engages: resolve of identical system is free" begin
    for mode in (WarmStart.Previous, WarmStart.Hegedus)
        cache = init(
            LinearProblem(A1, b1), KrylovJL_GMRES(warm_start = mode);
            abstol = tol, reltol = tol
        )
        solve!(cache)
        cache.b = copy(b1)
        solve!(cache)
        @test cache.cacheval.stats.niter <= 1
    end
    # cold start pays the full iteration count on the identical resolve
    cache = init(LinearProblem(A1, b1), KrylovJL_GMRES(); abstol = tol, reltol = tol)
    solve!(cache)
    cache.b = copy(b1)
    solve!(cache)
    @test cache.cacheval.stats.niter > 1
end

@testset "hegedus never increases the initial residual" begin
    # scaled RHS: the raw previous solution is a poor guess (wrong magnitude),
    # the Hegedüs rescaling recovers it exactly, so the resolve is again free
    cache = init(
        LinearProblem(A1, b1), KrylovJL_GMRES(warm_start = WarmStart.Hegedus);
        abstol = tol, reltol = tol
    )
    solve!(cache)
    cache.b = 1.0e-3 .* b1
    solve!(cache)
    @test cache.cacheval.stats.niter <= 1
    @test norm(A1 * cache.u - 1.0e-3 .* b1) < 1.0e-8
end

@testset "warm start with preconditioners" begin
    precs = (A, p) -> (Diagonal(diag(A)), I)
    for mode in (WarmStart.Previous, WarmStart.Hegedus)
        cache = init(
            LinearProblem(A1, b1), KrylovJL_GMRES(; precs, warm_start = mode);
            abstol = tol, reltol = tol
        )
        sol1 = solve!(cache)
        @test norm(A1 * sol1.u - b1) < 1.0e-6
        cache.b = b2
        sol2 = solve!(cache)
        @test norm(A1 * sol2.u - b2) < 1.0e-6
    end
end

@testset "complex systems" begin
    Ac = A1 + im * spdiagm(0 => 0.1 * cos.(1:n))
    bc = b1 + im * b2
    for mode in (WarmStart.Previous, WarmStart.Hegedus)
        cache = init(
            LinearProblem(Ac, bc), KrylovJL_GMRES(warm_start = mode);
            abstol = tol, reltol = tol
        )
        sol1 = solve!(cache)
        @test norm(Ac * sol1.u - bc) < 1.0e-6
        cache.b = bc .+ 1.0
        sol2 = solve!(cache)
        @test norm(Ac * sol2.u - (bc .+ 1.0)) < 1.0e-6
    end
end

@testset "batched RHS ignores warm_start" begin
    B = [b1 b2]
    sol = solve(
        LinearProblem(A1, B), KrylovJL_GMRES(warm_start = WarmStart.Hegedus);
        abstol = tol, reltol = tol
    )
    @test norm(A1 * sol.u - B) < 1.0e-6
end

@testset "zero previous solution falls back to cold start" begin
    cache = init(
        LinearProblem(A1, b1), KrylovJL_GMRES(warm_start = WarmStart.Previous);
        abstol = tol, reltol = tol
    )
    sol = solve!(cache)
    @test norm(A1 * sol.u - b1) < 1.0e-6
end
