using AllocCheck, ForwardDiff, LinearAlgebra, LinearSolve, SparseArrays, Test

if Sys.islinux()
    import LAPACK_jll, blis_jll
end

@check_allocs function allocation_checked_direct_lu_refactor_solve!(
        cache, Awork, A, alg
    )
    copyto!(Awork, A)
    cache.A = Awork
    info = LinearSolve._direct_lu_factorize!(cache.cacheval, Awork, alg)
    iszero(info) || return info
    LinearSolve._direct_lu_solve!(cache.cacheval, cache.u, cache.b, alg)
    cache.isfresh = false
    return info
end

function test_allocation_free_refactorization(alg, ::Type{T}) where {T}
    A1 = T[4 1; 2 3]
    A2 = T[3 -1; 1 2]
    b = T[1, 2]
    cache = init(LinearProblem(copy(A1), copy(b)), alg)
    Awork = cache.A

    @test solve!(cache).u ≈ A1 \ b
    info = allocation_checked_direct_lu_refactor_solve!(cache, Awork, A2, alg)
    @test iszero(info)
    @test cache.u ≈ A2 \ b

    copyto!(Awork, A1)
    cache.A = Awork
    @test solve!(cache).u ≈ A1 \ b
    copyto!(Awork, A2)
    cache.A = Awork
    if VERSION >= v"1.12"
        @test @allocated(solve!(cache)) == 0
    else
        solve!(cache)
    end
    return @test cache.u ≈ A2 \ b
end

@testset "Direct BLAS refactorization solve! is allocation-free" begin
    if LinearSolve.useopenblas
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            test_allocation_free_refactorization(OpenBLASLUFactorization(), T)
        end
    end

    if Base.get_extension(LinearSolve, :LinearSolveBLISExt) !== nothing
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            test_allocation_free_refactorization(LinearSolve.BLISLUFactorization(), T)
        end
    end
end

if LinearSolve.appleaccelerate_isavailable()
    @testset "Apple Accelerate refactorization solve! is allocation-free" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            test_allocation_free_refactorization(AppleAccelerateLUFactorization(), T)
        end
    end
end

# The ForwardDiff split path rebuilds the `p` partial right-hand sides through
# `xp_linsolve_rhs!` on every solve!, and everything it touches is cache-owned
# scratch. So the budget is a hard zero rather than "no worse than the primal
# solve": a per-call allocation here is per-Newton-step garbage in an AD-driven
# ODE solve, which is where this path is actually used.
const DUAL_XP_LINSOLVE_RHS! = getproperty(
    Base.get_extension(LinearSolve, :LinearSolveForwardDiffExt), :xp_linsolve_rhs!
)

function dual_linear_problem(n, p)
    A = [
        ForwardDiff.Dual{Nothing}(
                float(i == j ? 10 + i : 0.3 * (i + j)), ntuple(k -> 0.1k + 0.01 * (i + j), p)
            ) for i in 1:n, j in 1:n
    ]
    b = [ForwardDiff.Dual{Nothing}(float(i), ntuple(k -> 0.05k + 0.1i, p)) for i in 1:n]
    return A, b
end

function dual_rhs_kernel(cache)
    return DUAL_XP_LINSOLVE_RHS!(
        cache.linear_cache.u, getfield(cache, :partials_A),
        getfield(cache, :partials_b), cache
    )
end

# Both measurements go through a function barrier deliberately. At global scope
# the returned LinearSolution cannot be proven dead and `@allocated` reports a
# spurious 32 bytes on every version, which is more than the regressions this is
# guarding against and would mask them entirely.
function dual_solve_allocations(cache)
    solve!(cache)
    solve!(cache)
    return @allocated solve!(cache)
end

function dual_rhs_allocations(cache)
    solve!(cache)
    dual_rhs_kernel(cache)
    dual_rhs_kernel(cache)
    return @allocated dual_rhs_kernel(cache)
end

@testset "ForwardDiff Dual solve! is allocation-free" begin
    # Shapes chosen to straddle DUAL_RHS_GEMV_CUTOFF so both branches of the
    # fused kernel are covered: m*n*p is 108 and 8192 (hand loop) against 49152
    # (gemv). The two small ones also guard against a per-element or per-partial
    # allocation hiding in a case small enough for the optimizer to unroll.
    @testset "n = $n, p = $p" for (n, p) in ((6, 3), (32, 8), (64, 12))
        A, b = dual_linear_problem(n, p)
        A_primal = ForwardDiff.value.(A)
        b_primal = ForwardDiff.value.(b)

        # The fused dense-∂A rhs construction on its own.
        @test dual_rhs_allocations(init(LinearProblem(A, b), LUFactorization())) == 0

        # Whole solve!, for each way partials can enter the problem: Dual b only
        # and Dual A only take different `xp_linsolve_rhs!` methods than both.
        @test dual_solve_allocations(init(LinearProblem(A, b), LUFactorization())) == 0
        @test dual_solve_allocations(
            init(LinearProblem(A_primal, b), LUFactorization())
        ) == 0
        @test dual_solve_allocations(
            init(LinearProblem(A, b_primal), LUFactorization())
        ) == 0
    end
end

# SupernodalLU's sweep proof lives in `qa/supernodal_allocations.jl`, which only
# the QA group includes: this file also runs under `[AppleAccelerate]`, which
# covers `pre`, and that proof only holds on 1.12 (it bottoms out in stdlib
# `ldiv!`/`mul!` wrappers).  The direct-BLAS proofs above hold on every release.
