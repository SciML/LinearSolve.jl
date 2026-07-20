using LinearSolve, LinearAlgebra, StableRNGs, Test

if Sys.islinux()
    import LAPACK_jll, blis_jll
end

const DIRECT_BLAS_WRAPPER_CEILING = VERSION >= v"1.12" ? 0 : 48
const direct_blas_rng = StableRNG(42)

function direct_blas_refactor_solve!(cache, Awork, A)
    copyto!(Awork, A)
    cache.A = Awork
    return solve!(cache)
end

function test_direct_blas_refactorization(alg, ::Type{T}) where {T}
    n = 51
    A1 = rand(direct_blas_rng, T, n, n) + n * I
    A2 = rand(direct_blas_rng, T, n, n) + n * I
    Asing = copy(A1)
    Asing[:, 1] .= zero(T)
    b = rand(direct_blas_rng, T, n)
    cache = init(LinearProblem(copy(A1), copy(b)), alg)
    Awork = cache.A

    @test solve!(cache).u ≈ A1 \ b
    for Ak in (A2, A1, A2)
        sol = direct_blas_refactor_solve!(cache, Awork, Ak)
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ Ak \ b
    end

    direct_blas_refactor_solve!(cache, Awork, A1)
    ipiv_before = cache.cacheval.ipiv
    alloc = @allocated direct_blas_refactor_solve!(cache, Awork, A2)
    @test alloc <= DIRECT_BLAS_WRAPPER_CEILING
    @test cache.cacheval.ipiv === ipiv_before
    @test cache.cacheval.factors === Awork
    @test cache.u ≈ A2 \ b
    @test LinearSolve._cache_factorization(cache.cacheval) \ b ≈ A2 \ b

    @test direct_blas_refactor_solve!(cache, Awork, Asing).retcode ==
        ReturnCode.Failure
    sol = direct_blas_refactor_solve!(cache, Awork, A1)
    @test sol.retcode == ReturnCode.Success
    return @test sol.u ≈ A1 \ b
end

function test_direct_blas_resize(alg)
    n1, n2 = 5, 9
    A1 = rand(direct_blas_rng, n1, n1) + n1 * I
    b1 = rand(direct_blas_rng, n1)
    cache = init(LinearProblem(copy(A1), copy(b1)), alg)
    @test solve!(cache).u ≈ A1 \ b1
    ipiv_before = cache.cacheval.ipiv

    resize!(cache, n2)
    A2 = rand(direct_blas_rng, n2, n2) + n2 * I
    b2 = rand(direct_blas_rng, n2)
    cache.A = copy(A2)
    cache.b = copy(b2)
    cache.u = zeros(n2)
    @test solve!(cache).u ≈ A2 \ b2
    @test length(cache.cacheval.ipiv) == n2
    @test cache.cacheval.ipiv !== ipiv_before

    Awork = cache.A
    direct_blas_refactor_solve!(cache, Awork, A2)
    return @test @allocated(direct_blas_refactor_solve!(cache, Awork, A2)) <=
        DIRECT_BLAS_WRAPPER_CEILING
end

if LinearSolve.useopenblas
    @testset "OpenBLAS reuses its direct LU workspace" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            @testset "$T" test_direct_blas_refactorization(
                OpenBLASLUFactorization(), T
            )
        end
        test_direct_blas_resize(OpenBLASLUFactorization())
    end
end

if Base.get_extension(LinearSolve, :LinearSolveBLISExt) !== nothing
    @testset "BLIS reuses its direct LU workspace" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            @testset "$T" test_direct_blas_refactorization(
                LinearSolve.BLISLUFactorization(), T
            )
        end
        test_direct_blas_resize(LinearSolve.BLISLUFactorization())
    end
end
