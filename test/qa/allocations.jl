using AllocCheck, LinearAlgebra, LinearSolve, Test

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
