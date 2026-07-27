using AllocCheck, LinearAlgebra, LinearSolve, SparseArrays, Test

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

# SupernodalLU: the triangular sweeps run entirely off buffers owned by the
# factorization, whose sizes are known from the symbolic analysis, so they
# carry no growth branch and AllocCheck can prove them allocation-free
# statically (rather than sampling `@allocated`).  The user-facing `solve!`
# keeps a one-time scratch sizing, so it is asserted at runtime instead.
@check_allocs allocation_checked_supernodal_sweeps!(y, F) =
    LinearSolve.SupernodalLU._solve_panels!(y, F)

# What the sweeps can be *proved* about depends on the Julia version, because
# above `PANEL_BLAS_CUTOFF` they hand off to LinearAlgebra's `ldiv!`/`mul!`,
# whose wrappers are not statically clean on every release: 1.10 and 1.13 leave
# a `generic_trimatdiv!` dynamic dispatch plus boxed `MulAddMul`/`SubArray`
# values that AllocCheck reports (`triangular.jl`, `matmul.jl` — stdlib frames,
# no SupernodalLU code involved), and 1.11 allocates them for real on the
# `SubArray`-matrix path the multi-RHS sweep takes.  1.12 folds all of it away.
# So pin each assertion to the versions that can carry it rather than weakening
# it everywhere; the sweeps carry no growth branch on any version, which is what
# these are here to protect.
const STATIC_SWEEP_PROOF = v"1.12" <= VERSION < v"1.13"
const RUNTIME_MULTIRHS_ZERO = VERSION < v"1.11" || VERSION >= v"1.12"

function poisson2d_qa(k)
    n = k * k
    Is = Int[]; Js = Int[]; V = Float64[]
    idx(i, j) = (j - 1) * k + i
    for j in 1:k, i in 1:k
        c = idx(i, j)
        push!(Is, c); push!(Js, c); push!(V, 4.0)
        i > 1 && (push!(Is, c); push!(Js, idx(i - 1, j)); push!(V, -1.0))
        i < k && (push!(Is, c); push!(Js, idx(i + 1, j)); push!(V, -1.0))
        j > 1 && (push!(Is, c); push!(Js, idx(i, j - 1)); push!(V, -1.0))
        j < k && (push!(Is, c); push!(Js, idx(i, j + 1)); push!(V, -1.0))
    end
    return sparse(Is, Js, V, n, n)
end

@testset "SupernodalLU solve sweeps are provably allocation-free" begin
    for A in (
            SparseArrays.spdiagm(
                0 => fill(4.0, 200), 1 => fill(-1.0, 199), -1 => fill(-1.0, 199)
            ),
            poisson2d_qa(30),                 # real panels: maxnu > 1
        )
        n = size(A, 1)
        F = LinearSolve.SupernodalLU.snlu(A)
        LinearSolve.SupernodalLU._ensure_panel_scratch!(F, 1)
        y = ones(n)
        if STATIC_SWEEP_PROOF
            allocation_checked_supernodal_sweeps!(y, F)   # throws if it can allocate
            @test true
        end
        # the full solve!, which owns the one-time sizing, is zero at runtime
        b = ones(n)
        x = similar(b)
        LinearSolve.SupernodalLU.solve!(x, F, b; refine = 0)
        @test @allocated(LinearSolve.SupernodalLU.solve!(x, F, b; refine = 0)) == 0
        @test norm(A * x - b) <= 1.0e-8 * norm(b)
        # multi-RHS reuses the factor-owned scratch once sized
        B = ones(n, 3)
        X = similar(B)
        LinearSolve.SupernodalLU.solve!(X, F, B; refine = 0)
        if RUNTIME_MULTIRHS_ZERO
            @test @allocated(LinearSolve.SupernodalLU.solve!(X, F, B; refine = 0)) == 0
        end
    end
end
