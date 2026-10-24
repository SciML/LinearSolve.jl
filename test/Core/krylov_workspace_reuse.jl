using LinearSolve, LinearAlgebra, SparseArrays, SciMLOperators, Random, Test, Krylov

function krylov_a_update_solve_allocations(cache, Awork, A)
    copyto!(Awork, A)
    cache.A = Awork
    solve!(cache)
    copyto!(Awork, A)
    cache.A = Awork
    return @allocated solve!(cache)
end

@testset "KrylovJL reuses workspace on same-size A update" begin
    for (alg, bound) in ((KrylovJL_GMRES(), 256), (KrylovJL_CG(), 256))
        n = 80
        R = sprand(n, n, 0.1)
        A = R' * R + 10.0 * I
        b = rand(n)
        cache = init(LinearProblem(copy(A), copy(b)), alg)
        sol0 = solve!(cache)
        u0 = copy(sol0.u)
        iters0 = sol0.iters
        ws = cache.cacheval

        Awork = cache.A
        cache.u .= 0
        alloc = krylov_a_update_solve_allocations(cache, Awork, A)
        @test cache.cacheval === ws
        @test alloc <= bound

        cache.u .= 0
        copyto!(Awork, A)
        cache.A = Awork
        sol1 = solve!(cache)
        @test cache.cacheval === ws
        @test sol1.iters == iters0
        @test sol1.u ≈ u0

        R2 = sprand(n, n, 0.1)
        A2 = R2' * R2 + 10.0 * I
        cache.u .= 0
        copyto!(Awork, A2)
        cache.A = Awork
        sol2 = solve!(cache)
        @test cache.cacheval === ws
        @test sol2.u ≈ A2 \ b rtol = 1.0e-6 atol = 1.0e-6
    end

    begin
        n = 40
        R = sprand(n, n, 0.1)
        A = R' * R + 10.0 * I
        b = rand(n)
        cache = init(LinearProblem(copy(A), copy(b)), KrylovJL_GMRES())
        solve!(cache)
        ws = cache.cacheval
        n2 = 55
        R2 = sprand(n2, n2, 0.1)
        A2 = R2' * R2 + 10.0 * I
        b2 = rand(n2)
        resize!(cache, n2)
        cache.A = A2
        cache.b = b2
        cache.u = zeros(n2)
        solve!(cache)
        @test cache.cacheval !== ws
        @test cache.cacheval.m == n2 && cache.cacheval.n == n2
        @test cache.u ≈ A2 \ b2 rtol = 1.0e-6 atol = 1.0e-6
    end

    begin
        R = sprand(30, 30, 0.1)
        A = R' * R + 10.0 * I
        b = rand(30)
        alg = KrylovJL_GMRES()
        cache = init(LinearProblem(copy(A), copy(b)), alg)
        solve!(cache)
        @test !LinearSolve._krylov_can_reuse_workspace(
            cache.cacheval, alg, Float32.(A), Float32.(b), Float32.(cache.u)
        )
    end

end

@testset "KrylovJL non-Matrix A honors gmres_restart and MINRES window" begin
    Random.seed!(42)
    n = 200
    A = Tridiagonal(fill(-1.0, n - 1), fill(2.0, n), fill(-0.5, n - 1))
    b = rand(n)
    restart = 10
    alg = KrylovJL_GMRES(gmres_restart = restart)

    sol_ref = solve(LinearProblem(Matrix(A), copy(b)), alg)
    @test sol_ref.retcode == ReturnCode.Success
    for Awrap in (A, MatrixOperator(Matrix(A)))
        cache = init(LinearProblem(Awrap, copy(b)), alg)
        sol = solve!(cache)
        @test length(cache.cacheval.V) == restart
        @test sol.iters == sol_ref.iters
        @test sol.retcode == sol_ref.retcode
        @test sol.u ≈ sol_ref.u rtol = 1.0e-10 atol = 1.0e-10
    end

    As = Symmetric(Tridiagonal(fill(-1.0, n - 1), fill(4.0, n), fill(-1.0, n - 1)))
    bs = rand(n)
    window = 2
    cache_m = init(LinearProblem(As, copy(bs)), KrylovJL_MINRES(window = window))
    sol_m = solve!(cache_m)
    @test length(cache_m.cacheval.err_vec) == window
    @test sol_m.retcode == ReturnCode.Success
end

@testset "KrylovJL block output overlapping RHS" begin
    rng = MersenneTwister(1378)
    for alg in (KrylovJL_GMRES(), KrylovJL_MINRES()), overlap in (:same, :view)
        R = randn(rng, 12, 12)
        A = R' * R + 10I
        b = randn(rng, 12, 2)
        u = overlap === :same ? b : view(b, :, :)
        rhs = copy(b)
        cache = init(LinearProblem(A, b; u0 = u), alg; abstol = 1.0e-10, reltol = 1.0e-10)
        for _ in 1:2
            b .= rhs
            cache.A = copy(A)
            sol = solve!(cache)
            @test sol.retcode == ReturnCode.Success
            @test sol.u ≈ A \ rhs rtol = 1.0e-6
        end
    end
end

@testset "KrylovJL decreased subspace memory" begin
    rng = MersenneTwister(1378)
    n = 80
    A = Tridiagonal(fill(-1.0, n - 1), fill(2.0, n), fill(-0.5, n - 1))
    b = randn(rng, n)
    for algs in (
            (KrylovJL_GMRES(gmres_restart = 20), KrylovJL_GMRES(gmres_restart = 5)),
            (KrylovJL(; KrylovAlg = Krylov.fgmres!, memory = 20, restart = true), KrylovJL(; KrylovAlg = Krylov.fgmres!, memory = 5, restart = true)),
            (KrylovJL(; KrylovAlg = Krylov.fom!, memory = 20, restart = true), KrylovJL(; KrylovAlg = Krylov.fom!, memory = 5, restart = true)),
            (KrylovJL(; KrylovAlg = Krylov.diom!, memory = 20), KrylovJL(; KrylovAlg = Krylov.diom!, memory = 5)),
            (KrylovJL(; KrylovAlg = Krylov.dqgmres!, memory = 20), KrylovJL(; KrylovAlg = Krylov.dqgmres!, memory = 5)),
        )
        cache = init(LinearProblem(A, b), algs[1]; reltol = 1.0e-10, abstol = 1.0e-10, maxiters = 80)
        solve!(cache)
        cache.alg = algs[2]
        cache.A = copy(A)
        sol = solve!(cache)
        fresh = solve(LinearProblem(A, b), algs[2]; reltol = 1.0e-10, abstol = 1.0e-10, maxiters = 80)
        @test length(cache.cacheval.V) == 5
        @test sol.iters == fresh.iters
        @test sol.retcode == fresh.retcode
        @test sol.u ≈ fresh.u rtol = 1.0e-10 atol = 1.0e-10
    end
    alg = KrylovJL(; KrylovAlg = Krylov.gpmr!, memory = 5, restart = true)
    ws = Krylov.GpmrWorkspace(A, b; memory = 20)
    @test !LinearSolve._krylov_can_reuse_workspace(ws, alg, A, b, ws.x)

    cache = init(LinearProblem(A, hcat(b, reverse(b))), KrylovJL_GMRES(); reltol = 1.0e-10)
    solve!(cache)
    cache.alg = KrylovJL_GMRES(gmres_restart = 5)
    cache.A = copy(A)
    sol = solve!(cache)
    fresh = solve(LinearProblem(A, cache.b), cache.alg; reltol = 1.0e-10)
    @test length(cache.cacheval.V) == 5
    @test sol.iters == fresh.iters
    @test sol.u ≈ fresh.u rtol = 1.0e-10 atol = 1.0e-10

end
