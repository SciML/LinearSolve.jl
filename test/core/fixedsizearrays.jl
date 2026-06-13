using LinearSolve, FixedSizeArrays, LinearAlgebra, Test
using LinearSolve: defaultalg, OperatorAssumptions

# A `FixedSizeArray` is a dense, contiguous, column-major CPU array
# (`FixedSizeMatrix{T} <: DenseMatrix{T}`) that is not a `Base.Array`. It is
# directly LAPACK/BLAS-eligible, so the dense default and the dense LU caches
# should handle it like any other dense matrix. See SciML/LinearSolve.jl#1034.

@testset "defaultalg routes dense FixedSizeMatrix to a direct factorization" begin
    n = 100
    M = rand(n, n) + n * I
    b = rand(n)
    A = FixedSizeArray(M)
    v = FixedSizeArray(b)

    alg = defaultalg(A, v, OperatorAssumptions(true)).alg
    # Must be a dense direct factorization, never the matrix-free Krylov default.
    @test alg !== LinearSolve.DefaultAlgorithmChoice.KrylovJL_GMRES

    sol = solve(LinearProblem(A, v))
    @test sol.u isa FixedSizeArray
    # Direct LU accuracy, not the ~1e-8 of a matrix-free GMRES fallback.
    @test norm(M * Array(sol.u) - b) < 1.0e-10
end

# The BLAS backends (`MKL`, `AppleAccelerate`, `OpenBLAS`) are picked by the
# default for dense, pointer-addressable CPU memory — not specifically a
# `Base.Array`. A `FixedSizeMatrix` is exactly that, so it must resolve to the
# same algorithm a `Matrix` of the same size/eltype does, including MKL/Apple
# where those binaries are present.
@testset "defaultalg routing is container-agnostic for dense memory" begin
    assump = OperatorAssumptions(true)
    for n in (5, 100, 600), T in (Float32, Float64)
        M = rand(T, n, n) + n * I
        b = rand(T, n)
        ref = defaultalg(M, b, assump).alg
        fsa = defaultalg(FixedSizeArray(M), FixedSizeArray(b), assump).alg
        @test fsa === ref
        @test fsa !== LinearSolve.DefaultAlgorithmChoice.KrylovJL_GMRES
    end
end

@testset "dense LU factorizations accept FixedSizeArray" begin
    n = 100
    algs = Any[
        LUFactorization(),
        GenericLUFactorization(),
        QRFactorization(),
        OpenBLASLUFactorization(),
    ]

    for T in (Float32, Float64, ComplexF64)
        M = rand(T, n, n) + n * I
        b = rand(T, n)
        A = FixedSizeArray(M)
        v = FixedSizeArray(b)
        ref = M \ b
        tol = real(T) == Float32 ? 1.0f-3 : 1.0e-8
        for alg in algs
            sol = solve(LinearProblem(A, v), alg)
            @test sol.u isa FixedSizeArray
            @test norm(Array(sol.u) - ref) < tol
        end
    end
end

@testset "default solve and re-solve with FixedSizeArray" begin
    n = 60
    M = rand(n, n) + n * I
    b = rand(n)
    A = FixedSizeArray(M)
    v = FixedSizeArray(b)

    cache = init(LinearProblem(A, v))
    sol1 = solve!(cache)
    @test norm(M * Array(sol1.u) - b) < 1.0e-10

    # Re-solve with a new b reuses the cached factorization.
    b2 = rand(n)
    cache.b = FixedSizeArray(b2)
    sol2 = solve!(cache)
    @test norm(M * Array(sol2.u) - b2) < 1.0e-10
end

# The BLAS-direct LU caches (`MKL`, `OpenBLAS`, `AppleAccelerate`, `BLIS`) store
# a factorization built from `A` into a type-parameterized `cacheval` slot, so
# `init_cacheval` must produce a slot whose container matches `A`. This holds
# regardless of whether the corresponding BLAS binary is present, so the type
# check runs everywhere even when the solver itself can't.
@testset "BLAS LU init_cacheval slot tracks the FixedSizeArray container" begin
    n = 20
    A = FixedSizeArray(rand(n, n) + n * I)
    v = FixedSizeArray(rand(n))
    assump = OperatorAssumptions(true)
    for alg in (
            MKLLUFactorization(), OpenBLASLUFactorization(),
            AppleAccelerateLUFactorization(),
        )
        slot = LinearSolve.init_cacheval(
            alg, A, v, v, nothing, nothing, 0, 0.0, 0.0, true, assump
        )
        @test slot[1].factors isa FixedSizeArray
        @test slot[1].ipiv isa FixedSizeArray
    end
end
