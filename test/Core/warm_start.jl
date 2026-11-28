using LinearSolve, LinearAlgebra, SparseArrays, Test
using LinearSolve.Krylov
using RecursiveArrayTools, SciMLOperators

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

@testset "hegedus rejects a non-predictive previous direction" begin
    b_previous = [1.0, 0.0]
    b_nearly_orthogonal = [0.1, sqrt(0.99)]
    for alg in (
            KrylovJL_GMRES(warm_start = WarmStart.Hegedus),
            KrylovJL_FGMRES(warm_start = WarmStart.Hegedus),
        )
        cache = init(
            LinearProblem(Diagonal(ones(2)), b_previous), alg;
            abstol = 0.0, reltol = 0.997
        )
        solve!(cache)
        cache.b = b_nearly_orthogonal
        solve!(cache)
        @test cache.cacheval.stats.niter > 0
        @test cache.u ≈ b_nearly_orthogonal
    end
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

@testset "warm start temporaries do not allocate per solve" begin
    solve_bytes(cache) = @allocated solve!(cache)
    precs = (A, p) -> (Diagonal(Vector(diag(A))), I)
    # close enough to `b2` for the Hegedüs guess to be accepted
    b3 = b2 .+ 0.1 .* cos.(1:n)
    P = Diagonal(Vector(diag(A1)))
    @testset "$constructor $mode" for constructor in (KrylovJL_GMRES, KrylovJL_FGMRES),
            mode in (WarmStart.Previous, WarmStart.Hegedus)

        cache = init(
            LinearProblem(A1, copy(b2)), constructor(; precs, warm_start = mode);
            abstol = tol, reltol = tol
        )
        solve!(cache)
        for b in (b3, b2)
            cache.b = b
            solve!(cache)
        end
        cache.b = b3
        twin = deepcopy(cache)
        # a single temporary of `b`'s size would exceed this
        @test solve_bytes(cache) < sizeof(b3)
        cold = solve!(
            init(LinearProblem(A1, copy(b3)), constructor(; precs); abstol = tol, reltol = tol)
        )
        # fewer iterations only if the guess was used
        @test solve!(twin).iters < cold.iters
        threshold = tol + tol * norm(P \ b3)
        @test norm(P \ (b3 - A1 * cache.u)) <= threshold + 100 * eps() * norm(P \ b3)
    end
end

@testset "warm start after the system size changes: $mode" for mode in
    (WarmStart.Previous, WarmStart.Hegedus)

    precs = (A, p) -> (Diagonal(Vector(diag(A))), I)
    cache = init(
        LinearProblem(A1, copy(b1)), KrylovJL_GMRES(; precs, warm_start = mode);
        abstol = tol, reltol = tol
    )
    solve!(cache)
    solve!(cache)
    for m in (n ÷ 2, n)
        A, b = A1[1:m, 1:m], b1[1:m]
        reinit!(cache; A, b, u = ones(m))
        @test norm(A * solve!(cache).u - b) < 1.0e-6
    end
end

@testset "replacing b with another size releases the temporary" begin
    precs = (A, p) -> (Diagonal(Vector(diag(A))), I)
    cache = init(
        LinearProblem(A1, copy(b1)), KrylovJL_GMRES(; precs, warm_start = WarmStart.Hegedus);
        abstol = tol, reltol = tol
    )
    solve!(cache)
    solve!(cache)
    scratch = cache.scratch
    @test scratch !== nothing
    cache.b = copy(b2)
    @test cache.scratch === scratch
    # a zero guess returns before the temporary is looked at again
    m = n ÷ 2
    reinit!(cache; A = A1[1:m, 1:m], b = zeros(m), u = zeros(m))
    @test cache.scratch === nothing
    @test iszero(solve!(cache).u)
    @test cache.scratch === nothing
end

@testset "partitioned right-hand side whose layout changes" begin
    function copy_flat!(w, v, u, p, t)
        for i in 1:5
            w[i] = v[i]
        end
        return w
    end
    split_at(x, k) = ArrayPartition(fill(x, k), fill(x, 5 - k))
    proto = split_at(1.0, 2)
    A = FunctionOperator(copy_flat!, proto, proto; isinplace = true, islinear = true)
    @testset "$mode" for mode in (WarmStart.Previous, WarmStart.Hegedus)
        cache = init(LinearProblem(A, split_at(1.0, 2)), KrylovJL_GMRES(warm_start = mode))
        solve!(cache)
        cache.b = split_at(2.0, 3)
        @test collect(solve!(cache).u) ≈ fill(2.0, 5)
    end
end

@testset "partitioned right-hand side: unused temporaries are not allocated" begin
    solve_bytes(cache) = @allocated solve!(cache)
    b = ArrayPartition(ones(1000), ones(1000))
    A = FunctionOperator((w, v, u, p, t) -> (w .= 2 .* v), b, b; isinplace = true, islinear = true)
    cache = init(LinearProblem(A, b), KrylovJL_GMRES(warm_start = WarmStart.Previous))
    solve!(cache)
    solve!(cache)
    # one partition's worth of bytes
    @test solve_bytes(cache) < sizeof(b.x[1])
    @test collect(cache.u) ≈ fill(0.5, 2000)
end
