using LinearSolve, LinearAlgebra, SparseArrays, Test

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
        if isdefined(LinearSolve, :_krylov_can_reuse_workspace)
            @test !LinearSolve._krylov_can_reuse_workspace(
                cache.cacheval, alg, Float32.(A), Float32.(b), Float32.(cache.u)
            )
        end
    end
end
