using LinearSolve, LinearAlgebra, RecursiveFactorization, SparseArrays, StaticArrays, Test

alglist = (
    LUFactorization,
    QRFactorization,
    KrylovJL_GMRES,
    GenericLUFactorization,
    RFLUFactorization,
    SVDFactorization,
    NormalCholeskyFactorization,
    KrylovJL_CRAIGMR,
    KrylovJL_LSMR,
)

@testset "Success" begin
    for alg in alglist
        A = [2.0 1.0; -1.0 1.0]
        b = [-1.0, 1.0]
        prob = LinearProblem(A, b)
        linsolve = init(prob, alg())
        sol = solve!(linsolve)
        @test SciMLBase.successful_retcode(sol.retcode)
    end
end

lualgs = (
    LUFactorization(),
    QRFactorization(),
    GenericLUFactorization(),
    LinearSolve.DefaultLinearSolver(
        LinearSolve.DefaultAlgorithmChoice.LUFactorization; safetyfallback = false
    ),
    RFLUFactorization(),
    NormalCholeskyFactorization(),
)
@testset "Failure" begin
    for alg in lualgs
        @show alg
        A = [1.0 1.0; 1.0 1.0]
        b = [-1.0, 1.0]
        prob = LinearProblem(A, b)
        linsolve = init(prob, alg)
        sol = solve!(linsolve)
        if alg isa NormalCholeskyFactorization
            # This is a known and documented incorrectness in NormalCholeskyFactorization
            # due to numerical instability in its method that is fundamental.
            @test SciMLBase.successful_retcode(sol.retcode)
        else
            @test !SciMLBase.successful_retcode(sol.retcode)
        end
    end
end

rankdeficientalgs = (
    QRFactorization(LinearAlgebra.ColumnNorm()),
    KrylovJL_GMRES(),
    SVDFactorization(),
    KrylovJL_CRAIGMR(),
    KrylovJL_LSMR(),
    LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.LUFactorization),
)

@testset "Rank Deficient Success" begin
    for alg in rankdeficientalgs
        @show alg
        A = [1.0 1.0; 1.0 1.0]
        b = [-1.0, 1.0]
        prob = LinearProblem(A, b)
        linsolve = init(prob, alg)
        sol = solve!(linsolve)
        @test SciMLBase.successful_retcode(sol.retcode)
    end
end

staticarrayalgs = (
    DirectLdiv!(),
    LUFactorization(),
    CholeskyFactorization(),
    NormalCholeskyFactorization(),
    SVDFactorization(),
)
@testset "StaticArray Success" begin
    A = Float64[1 2 3; 4 3.5 1.7; 5.2 1.8 9.7]
    A = A * A'
    b = Float64[2, 5, 8]
    prob1 = LinearProblem(SMatrix{3, 3}(A), SVector{3}(b))
    sol = solve(prob1)
    @test SciMLBase.successful_retcode(sol.retcode)

    for alg in staticarrayalgs
        sol = solve(prob1, alg)
        @test SciMLBase.successful_retcode(sol.retcode)
    end

    @test_broken sol = solve(prob1, QRFactorization()) # Needs StaticArrays `qr` fix
end

@testset "Non-finite or singular factorization reports Failure (#1349)" begin
    b = [1.0, 2.0]
    algs = (
        LUFactorization(), GenericLUFactorization(), GESVFactorization(),
        GenericFactorization(lu!), GenericFactorization(lu),
        GenericFactorization(qr), GenericFactorization(),
    )
    for A in ([1.0 NaN; 2.0 3.0], [1.0 Inf; 2.0 3.0], [1.0 1.0; 1.0 1.0])
        for alg in algs
            # `GenericFactorization()` (`factorize`) factorizes eagerly in
            # `init_cacheval`, so a singular matrix still throws at `init`;
            # its `solve!`-time behaviour is covered by the refactorization
            # cases below.
            if A == [1.0 1.0; 1.0 1.0] && alg isa GenericFactorization{typeof(factorize)}
                continue
            end
            sol = solve(LinearProblem(copy(A), copy(b)), alg)
            @test sol.retcode == ReturnCode.Failure
        end
    end

    # A non-finite right-hand side cannot produce a successful solve.
    for alg in algs
        sol = solve(LinearProblem([2.0 1.0; 1.0 3.0], [NaN, 2.0]), alg)
        @test sol.retcode == ReturnCode.Failure
    end

    # Refactorization onto a bad matrix (the ODE W-matrix case) must report
    # Failure through the retcode so the caller can reject the step.
    for alg in (
            LUFactorization(), GenericLUFactorization(),
            GenericFactorization(lu!), GenericFactorization(lu),
            GenericFactorization(),
        )
        cache = init(LinearProblem([2.0 1.0; 1.0 3.0], copy(b)), alg)
        @test solve!(cache).retcode == ReturnCode.Success
        cache.A = [1.0 NaN; 2.0 3.0]
        @test solve!(cache).retcode == ReturnCode.Failure
        cache.A = [1.0 1.0; 1.0 1.0]
        @test solve!(cache).retcode == ReturnCode.Failure
    end

    # A wrapped factorization that throws on failure (`PosDefException` from
    # `cholesky` here) is likewise reported through the retcode.
    cache = init(
        LinearProblem([2.0 1.0; 1.0 2.0], copy(b)), GenericFactorization(cholesky)
    )
    @test solve!(cache).retcode == ReturnCode.Success
    cache.A = [1.0 2.0; 2.0 1.0]
    @test solve!(cache).retcode == ReturnCode.Failure

    # Well-posed solves are unchanged.
    for alg in algs
        sol = solve(LinearProblem([2.0 1.0; 1.0 3.0], copy(b)), alg)
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ [0.2, 0.6]
    end
end

@testset "StaticArray Singular" begin
    A2 = SA[0.0 0.0; 1.0 -1.0]
    A4 = SMatrix{4, 4}(
        [
            1.0 2.0 3.0 4.0
            2.0 4.0 6.0 8.0
            0.0 1.0 0.0 1.0
            1.0 0.0 1.0 0.0
        ]
    )
    for (A, b, B) in (
            (A2, SA[1.0, 2.0], SMatrix{2, 2}(1.0I)),
            (A4, SA[1.0, 2.0, 3.0, 4.0], SMatrix{4, 4}(1.0I)),
        )
        for rhs in (b, B)
            # Default alg rescues singular LU with an SVD least-squares solve,
            # returning the finite min-norm pseudo-solution (like the dense
            # default's pivoted-QR rescue).
            sol = solve(LinearProblem(A, rhs))
            @test SciMLBase.successful_retcode(sol.retcode)
            @test all(isfinite, sol.u)
            @test sol.u isa (rhs isa SVector ? SVector : SMatrix)
            @test sol.u ≈ pinv(Matrix(A)) * (rhs isa SVector ? Vector(rhs) : Matrix(rhs))

            # Explicit LUFactorization reports Failure without throwing,
            # matching the dense LUFactorization behavior on singular input.
            sole = solve(LinearProblem(A, rhs), LUFactorization())
            @test sole.retcode == ReturnCode.Failure
            @test all(isfinite, sole.u)
        end
    end

    # Nonsingular static solves match `\` (up to FMA contraction differences
    # in StaticArrays' inlined small-size formulas, per review the check is
    # allowed to inline rather than preserving bit-identity)
    A2n = SA[2.0 1.0; 1.0 3.0]
    A4n = SMatrix{4, 4}(
        [
            4.0 1.0 0.0 2.0
            1.0 5.0 1.0 0.0
            0.0 1.0 6.0 1.0
            2.0 0.0 1.0 7.0
        ]
    )
    for (A, b, B) in (
            (A2n, SA[1.0, 2.0], SA[1.0 0.5; 2.0 1.5]),
            (
                A4n, SA[1.0, 2.0, 3.0, 4.0],
                SMatrix{4, 2}([1.0 0.5; 2.0 1.5; 3.0 2.5; 4.0 3.5]),
            ),
        )
        for rhs in (b, B)
            @test solve(LinearProblem(A, rhs)).u ≈ A \ rhs rtol = 4 * eps()
            @test solve(LinearProblem(A, rhs), LUFactorization()).u === lu(A) \ rhs
        end
    end

    # GESVFactorization on static problems: the static analog of the dense
    # gesv driver — one-shot solve, Failure through the retcode on singular,
    # no SVD rescue.
    for (A, b) in ((A2n, SA[1.0, 2.0]), (A4n, SVector{4}(1.0:4.0)))
        sol = solve(LinearProblem(A, b), GESVFactorization())
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ A \ b rtol = 4 * eps()
    end
    B2 = SMatrix{2, 2}(1.0I)
    @test solve(LinearProblem(A2n, B2), GESVFactorization()).u ≈ A2n \ B2
    As2 = SA[0.0 0.0; 1.0 -1.0]
    As4 = SMatrix{4, 4}([1.0 2.0 3.0 4.0; 2.0 4.0 6.0 8.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0])
    for (As, b) in ((As2, SA[1.0, 2.0]), (As4, SVector{4}(1.0:4.0)))
        sol = solve(LinearProblem(As, b), GESVFactorization())
        @test sol.retcode == ReturnCode.Failure
        @test all(iszero, sol.u)
    end
    @test_throws ArgumentError solve(
        LinearProblem(SMatrix{3, 2}(ones(3, 2)), SA[1.0, 2.0, 3.0]), GESVFactorization()
    )

    # Non-square static problems are unchanged by the singular handling
    Ans = SMatrix{3, 2}([1.0 2.0; 3.0 4.0; 5.0 6.0])
    bns = SA[1.0, 2.0, 3.0]
    solns = solve(LinearProblem(Ans, bns))
    @test SciMLBase.successful_retcode(solns.retcode)
    @test solns.u ≈ Matrix(Ans) \ Vector(bns)
end


struct FinitenessCountFloat <: AbstractFloat
    value::Float64
end
const finiteness_checks = Ref(0)
Base.zero(::Type{FinitenessCountFloat}) = FinitenessCountFloat(0.0)
Base.conj(x::FinitenessCountFloat) = x
Base.real(x::FinitenessCountFloat) = x
function Base.isfinite(x::FinitenessCountFloat)
    finiteness_checks[] += 1
    return isfinite(x.value)
end

@testset "Structured factorization finiteness" begin
    @testset "Active SymTridiagonal storage" begin
        for padding in (NaN, Inf, -Inf), n in (1, 2, 4)
            A = SymTridiagonal(fill(2.0, n), [fill(-0.5, n - 1); padding])
            b = ones(n)
            for alg in (GenericFactorization(), GenericFactorization(ldlt))
                sol = solve(LinearProblem(A, b), alg)
                @test sol.retcode == ReturnCode.Success
                @test sol.u ≈ Matrix(A) \ b
            end
        end
    end

    @testset "Sparse symmetric finiteness traversal" begin
        for n in (32, 64), wrapper in (Symmetric, Hermitian), uplo in (:U, :L)
            P = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(4.0, n), 1 => fill(-1.0, n - 1))
            counted = SparseMatrixCSC(
                n, n, copy(P.colptr), copy(rowvals(P)), FinitenessCountFloat.(nonzeros(P))
            )
            finiteness_checks[] = 0
            @test LinearSolve._matrix_allfinite(wrapper(counted, uplo))
            @test finiteness_checks[] <= nnz(P)

            i, j = uplo == :U ? (2, 1) : (1, 2)
            P[i, j] = NaN
            A = wrapper(P, uplo)
            @test LinearSolve._matrix_allfinite(A)
            sol = solve(LinearProblem(A, ones(n)), GenericFactorization())
            @test sol.retcode == ReturnCode.Success
            @test A * sol.u ≈ ones(n)
            cache = init(LinearProblem(A, ones(n)), GenericFactorization())
            P[j, i] = Inf
            cache.A = A
            @test !LinearSolve._matrix_allfinite(A)
            @test solve!(cache).retcode == ReturnCode.Failure
        end
    end
end
