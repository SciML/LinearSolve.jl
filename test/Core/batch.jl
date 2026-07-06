using LinearSolve, LinearAlgebra, SparseArrays, StaticArrays, Test
using Random

Random.seed!(1234)

const n = 12
const k = 4

@testset "Dense factorizations vs A \\ B" begin
    for T in (Float64, ComplexF64)
        A = rand(T, n, n) + n * I
        B = rand(T, n, k)
        Xref = A \ B
        algs = Any[
            LUFactorization(), GenericLUFactorization(),
            QRFactorization(), QRFactorization(ColumnNorm()),
            SVDFactorization(),
        ]
        LinearSolve.usemkl && push!(algs, MKLLUFactorization())
        LinearSolve.useopenblas && push!(algs, OpenBLASLUFactorization())
        LinearSolve.appleaccelerate_isavailable() &&
            push!(algs, AppleAccelerateLUFactorization())
        @testset "$T $(nameof(typeof(alg)))" for alg in algs
            sol = solve(LinearProblem(A, B), alg)
            @test SciMLBase.successful_retcode(sol)
            @test size(sol.u) == (n, k)
            @test sol.u ≈ Xref
        end

        # SPD / symmetric algorithms
        Aspd = Matrix(Hermitian(A' * A + n * I))
        Awrap = T <: Complex ? Hermitian(Aspd) : Symmetric(Aspd)
        sol = solve(LinearProblem(Awrap, B), CholeskyFactorization())
        @test sol.u ≈ Awrap \ B

        if T <: Real
            sol = solve(LinearProblem(Symmetric(Aspd), B), BunchKaufmanFactorization())
            @test sol.u ≈ Symmetric(Aspd) \ B

            sol = solve(LinearProblem(A, B), NormalCholeskyFactorization())
            @test sol.u ≈ Xref rtol = 1.0e-6
        end

        # Default algorithm
        sol = solve(LinearProblem(A, B))
        @test size(sol.u) == (n, k)
        @test sol.u ≈ Xref
    end
end

@testset "Residual safety check with matrix B" begin
    A = rand(n, n) + n * I
    B = rand(n, k)
    sol = solve(LinearProblem(A, B), LUFactorization(residualsafety = true))
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ A \ B
end

@testset "Generic element types (GenericLUFactorization)" begin
    A = big.(rand(n, n)) + n * I
    B = big.(rand(n, k))
    sol = solve(LinearProblem(A, B), GenericLUFactorization())
    @test sol.u ≈ A \ B
end

@testset "Non-square (least squares / minimum norm), k != n" begin
    m2, n2, k2 = 20, 10, 3
    A = rand(m2, n2)
    B = rand(m2, k2)
    for alg in (QRFactorization(), SVDFactorization(), nothing)
        sol = solve(LinearProblem(A, B), alg)
        @test size(sol.u) == (n2, k2)
        @test sol.u ≈ A \ B rtol = 1.0e-8
    end
end

@testset "Structured matrices via default algorithm" begin
    B = rand(n, k)

    A = Diagonal(rand(n) .+ 1)
    sol = solve(LinearProblem(A, B))
    @test sol.u ≈ Matrix(A) \ B

    A = Tridiagonal(rand(n - 1), rand(n) .+ 4, rand(n - 1))
    sol = solve(LinearProblem(A, B))
    @test sol.u ≈ Matrix(A) \ B

    A = SymTridiagonal(fill(4.0, n), fill(1.0, n - 1))
    sol = solve(LinearProblem(A, B))
    @test sol.u ≈ Matrix(A) \ B
end

@testset "Sparse A" begin
    Random.seed!(42)
    As = sprand(30, 30, 0.3) + 30I
    B = rand(30, k)
    Xref = Matrix(As) \ B
    # Default (PureKLU slot) and explicit sparse LU algorithms
    for alg in (nothing, UMFPACKFactorization(), KLUFactorization())
        sol = solve(LinearProblem(As, B), alg)
        @test size(sol.u) == (30, k)
        @test sol.u ≈ Xref
    end

    # Sparse Cholesky (CHOLMOD)
    Asym = sparse(Symmetric(As + As'))
    sol = solve(LinearProblem(Symmetric(Asym), B), CHOLMODFactorization())
    @test sol.u ≈ Matrix(Asym) \ B

    # Non-square sparse default (column-pivoted sparse QR), k != n
    Ans = sprand(40, 20, 0.5)
    Bns = rand(40, k)
    sol = solve(LinearProblem(Ans, Bns))
    @test size(sol.u) == (20, k)
    @test sol.u ≈ Matrix(Ans) \ Bns rtol = 1.0e-8
end

@testset "Cache reuse: new b and new A" begin
    A1 = rand(n, n) + n * I
    A2 = rand(n, n) + n * I
    B1 = rand(n, k)
    B2 = rand(n, k)
    for alg in (LUFactorization(), QRFactorization(), nothing)
        cache = init(LinearProblem(A1, B1), alg)
        @test solve!(cache).u ≈ A1 \ B1
        cache.b = B2
        @test solve!(cache).u ≈ A1 \ B2
        # `cache.A = ...` aliases (in-place factorizations mutate it), so hand the
        # cache its own copy and compare against the pristine A2.
        cache.A = copy(A2)
        @test solve!(cache).u ≈ A2 \ B2
    end
end

@testset "Singular A returns failure retcode without throwing" begin
    A = zeros(n, n)
    B = rand(n, k)
    sol = solve(LinearProblem(A, B), LUFactorization())
    @test !SciMLBase.successful_retcode(sol)

    # Default polyalgorithm on a rank-deficient matrix must not throw
    # (QR fallback path with matrix B)
    A = rand(n, n)
    A[:, 1] .= A[:, 2]
    sol = solve(LinearProblem(A, B))
    @test sol.u isa Matrix
end

@testset "Block Krylov methods solve batched RHS" begin
    A = rand(n, n) + n * I
    B = rand(n, k)
    sol = solve(LinearProblem(A, B), KrylovJL_GMRES())
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ A \ B

    S = A + A' + 2n * I # symmetric for MINRES
    sol = solve(LinearProblem(S, B), KrylovJL_MINRES())
    @test SciMLBase.successful_retcode(sol)
    @test sol.u ≈ S \ B
end

@testset "Iterative algorithms without block variants error informatively" begin
    A = rand(n, n) + n * I
    B = rand(n, k)
    @test_throws ArgumentError solve(LinearProblem(A, B), SimpleGMRES())
    @test_throws ArgumentError solve(LinearProblem(A, B), KrylovJL_CG())
    @test_throws ArgumentError solve(LinearProblem(A, B), SimpleLUFactorization())
    err = try
        solve(LinearProblem(A, B), KrylovJL_CG())
    catch e
        e
    end
    @test occursin("batched", err.msg)
    @test occursin("KrylovJL", err.msg)
end

@testset "Static arrays" begin
    A = (@SMatrix rand(4, 4)) + 4 * I
    B = @SMatrix rand(4, 2)
    sol = solve(LinearProblem(A, B))
    @test sol.u ≈ A \ B
end

@testset "Single-column matrix B matches vector b" begin
    A = rand(n, n) + n * I
    b = rand(n)
    B = reshape(copy(b), n, 1)
    solvec = solve(LinearProblem(A, b), LUFactorization())
    solmat = solve(LinearProblem(A, B), LUFactorization())
    @test size(solmat.u) == (n, 1)
    @test vec(solmat.u) ≈ solvec.u
end

@testset "alias_A = true: LUFactorization refactorizes in place" begin
    na = 100
    D1 = rand(na, na) + na * I
    D2 = rand(na, na) + na * I
    B = rand(na, k)

    function realias_solve!(cache, Abuf, D)
        copyto!(Abuf, D)
        cache.A = Abuf
        return solve!(cache)
    end

    # alias_A = true: the user permitted overwriting A, so refactorization runs
    # lu! on cache.A directly; repeated solves stay correct as long as the
    # caller refills A between solves.
    Abuf = copy(D1)
    cache = init(
        LinearProblem(Abuf, copy(B)), LUFactorization();
        alias = LinearAliasSpecifier(alias_A = true)
    )
    @test cache.A === Abuf # not copied at init
    @test solve!(cache).u ≈ D1 \ B
    @test !(Abuf ≈ D1) # was overwritten by its LU factors
    sol = realias_solve!(cache, Abuf, D2)
    @test sol.u ≈ D2 \ B
    @test cache.A === Abuf

    # Reuse must not allocate a fresh n×n copy: O(n) small allocations only.
    realias_solve!(cache, Abuf, D1) # warm up this exact path
    alloc = @allocated realias_solve!(cache, Abuf, D2)
    @test alloc < 20000 # n^2 * 8 = 80000 bytes would mean A was copied

    # default (alias_A = false): cache.A is left untouched by refactorization.
    cache = init(LinearProblem(copy(D1), copy(B)), LUFactorization())
    @test solve!(cache).u ≈ D1 \ B
    @test cache.A ≈ D1
    cache.A = copy(D2)
    @test solve!(cache).u ≈ D2 \ B
    @test cache.A ≈ D2
end
