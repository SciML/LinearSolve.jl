using LinearSolve, Test
using SparseArrays, LinearAlgebra
using Krylov
using Random

m, n = 13, 3

A = rand(m, n)
b = rand(m)
prob = LinearProblem(A, b)
res = A \ b
@test solve(prob).u ≈ res
@test !LinearSolve.needs_square_A(QRFactorization())
@test solve(prob, QRFactorization()) ≈ res
@test !LinearSolve.needs_square_A(FastQRFactorization())
@test solve(prob, FastQRFactorization()) ≈ res
@test !LinearSolve.needs_square_A(KrylovJL_LSMR())
@test solve(prob, KrylovJL_LSMR()) ≈ res
@test !LinearSolve.needs_square_A(KrylovJL(KrylovAlg = Krylov.lsqr!))
@test solve(prob, KrylovJL(KrylovAlg = Krylov.lsqr!)) ≈ res
@test !LinearSolve.needs_square_A(KrylovJL(KrylovAlg = Krylov.cgls!))
@test solve(prob, KrylovJL(KrylovAlg = Krylov.cgls!)) ≈ res
@test !LinearSolve.needs_square_A(KrylovJL(KrylovAlg = Krylov.crls!))
@test solve(prob, KrylovJL(KrylovAlg = Krylov.crls!)) ≈ res
@test !LinearSolve.needs_square_A(KrylovJL(KrylovAlg = Krylov.lslq!))
@test solve(prob, KrylovJL(KrylovAlg = Krylov.lslq!)) ≈ res

A = sprand(m, n, 0.5)
b = rand(m)
prob = LinearProblem(A, b)
res = A \ b
@test solve(prob).u ≈ res
@test solve(prob, QRFactorization()) ≈ res
@test solve(prob, KrylovJL_LSMR()) ≈ res

A = sprand(n, m, 0.5)
b = rand(n)
prob = LinearProblem(A, b)
res = Matrix(A) \ b
@test !LinearSolve.needs_square_A(KrylovJL_CRAIGMR())
@test solve(prob, KrylovJL_CRAIGMR()) ≈ res
@test !LinearSolve.needs_square_A(KrylovJL(KrylovAlg = Krylov.cgne!))
@test solve(prob, KrylovJL(KrylovAlg = Krylov.cgne!)) ≈ res
@test !LinearSolve.needs_square_A(KrylovJL(KrylovAlg = Krylov.craig!))
@test solve(prob, KrylovJL(KrylovAlg = Krylov.craig!)) ≈ res
@test !LinearSolve.needs_square_A(KrylovJL(KrylovAlg = Krylov.crmr!))
@test solve(prob, KrylovJL(KrylovAlg = Krylov.crmr!)) ≈ res
@test !LinearSolve.needs_square_A(KrylovJL(KrylovAlg = Krylov.lnlq!))
@test solve(prob, KrylovJL(KrylovAlg = Krylov.lnlq!)) ≈ res

A = sprandn(1000, 100, 0.1)
b = randn(1001)
prob = LinearProblem(A, view(b, 1:1000))
linsolve = init(prob, QRFactorization())
solve!(linsolve)

A = randn(1000, 100)
b = randn(1000)
@test isapprox(solve(LinearProblem(A, b)).u, Symmetric(A' * A) \ (A' * b))
solve(LinearProblem(A, b)).u;
@test !LinearSolve.needs_square_A(NormalCholeskyFactorization())
solve(LinearProblem(A, b), (LinearSolve.NormalCholeskyFactorization())).u;
@test !LinearSolve.needs_square_A(NormalBunchKaufmanFactorization())
solve(LinearProblem(A, b), (LinearSolve.NormalBunchKaufmanFactorization())).u;
solve(
    LinearProblem(A, b),
    assumptions = (
        OperatorAssumptions(
            false;
            condition = OperatorCondition.WellConditioned
        )
    )
).u;

A = sprandn(5000, 100, 0.1)
b = randn(5000)
@test isapprox(solve(LinearProblem(A, b)).u, ldlt(A' * A) \ (A' * b))
solve(LinearProblem(A, b)).u;
solve(LinearProblem(A, b), (LinearSolve.NormalCholeskyFactorization())).u;
solve(
    LinearProblem(A, b),
    assumptions = (
        OperatorAssumptions(
            false;
            condition = OperatorCondition.WellConditioned
        )
    )
).u;

# Underdetermined
m, n = 2, 3

A = rand(m, n)
b = rand(m)
prob = LinearProblem(A, b)
res = A \ b
@test solve(prob).u ≈ res

# `needs_square_A` is enforced at `init`: an algorithm that requires a square `A`
# and is handed a non-square one now throws an `ArgumentError` naming the
# least-squares alternatives, instead of leaking a `DimensionMismatch` (LU,
# Cholesky), an `ArgumentError` about Bunch-Kaufman validity, or a `FieldError`
# about `Array` having no `diag` field from deep inside the factorization.
# Regression test for https://github.com/SciML/LinearSolve.jl/issues/546
@testset "needs_square_A enforcement" begin
    A_tall = rand(12, 4)
    b_tall = rand(12)
    A_wide = rand(4, 12)
    b_wide = rand(4)
    res_tall = A_tall \ b_tall

    @testset "square-only algorithms are rejected" begin
        for alg in (
                LUFactorization(), GenericLUFactorization(), SimpleLUFactorization(),
                CholeskyFactorization(), BunchKaufmanFactorization(),
                DiagonalFactorization(), LDLtFactorization(),
            )
            @test LinearSolve.needs_square_A(alg)
            # Rejected for both orientations, and at `init` rather than `solve`.
            @test_throws ArgumentError init(
                LinearProblem(copy(A_tall), copy(b_tall)), alg
            )
            @test_throws ArgumentError solve(
                LinearProblem(copy(A_tall), copy(b_tall)), alg
            )
            @test_throws ArgumentError solve(
                LinearProblem(copy(A_wide), copy(b_wide)), alg
            )
        end
        # The message names what to use instead.
        err = try
            solve(LinearProblem(copy(A_tall), copy(b_tall)), LUFactorization())
        catch e
            sprint(showerror, e)
        end
        @test occursin("requires a square `A`", err)
        @test occursin("12x4", err)
        @test occursin("QRFactorization(ColumnNorm())", err)
        @test occursin("SVDFactorization()", err)
    end

    @testset "non-square-capable algorithms are unaffected" begin
        # These were all marked as needing a square `A` while solving non-square
        # systems correctly, so the trait had to be corrected before it could be
        # enforced.
        for alg in (
                SVDFactorization(), GenericFactorization(),
                GenericFactorization(fact_alg = qr), QRFactorization(),
                QRFactorization(ColumnNorm()), NormalCholeskyFactorization(),
                KrylovJL_LSMR(),
            )
            @test !LinearSolve.needs_square_A(alg)
            @test solve(LinearProblem(copy(A_tall), copy(b_tall)), alg).u ≈ res_tall
        end
    end

    @testset "the default algorithm is never rejected" begin
        # `solve(prob)` resolves to a `DefaultLinearSolver` before `init` runs, so
        # the trait must be false for it too or every non-square default solve
        # would throw.
        @test !LinearSolve.needs_square_A(
            LinearSolve.defaultalg(A_tall, b_tall, OperatorAssumptions(false))
        )
        @test solve(LinearProblem(copy(A_tall), copy(b_tall))).u ≈ res_tall
        @test solve(LinearProblem(copy(A_wide), copy(b_wide))).u ≈ A_wide \ b_wide

        A_sparse = sprand(12, 4, 0.7)
        while rank(Matrix(A_sparse)) < 4
            A_sparse = sprand(12, 4, 0.7)
        end
        b_sparse = rand(12)
        @test solve(LinearProblem(copy(A_sparse), copy(b_sparse))).u ≈
            Matrix(A_sparse) \ b_sparse
        # The sparse non-square default itself must not be rejected either.
        @test !LinearSolve.needs_square_A(SparseColumnPivotedQRFactorization())
    end

    @testset "square problems are unaffected" begin
        A_sq = rand(6, 6)
        b_sq = rand(6)
        for alg in (LUFactorization(), CholeskyFactorization(), nothing)
            A_use = alg isa CholeskyFactorization ? A_sq'A_sq + 6I : A_sq
            sol = alg === nothing ? solve(LinearProblem(copy(A_use), copy(b_sq))) :
                solve(LinearProblem(copy(A_use), copy(b_sq)), alg)
            @test sol.u ≈ A_use \ b_sq
        end
    end
end

# Rank-deficient least squares: the default is unpivoted QR, which cannot solve a
# rank-deficient system — it used to return all-zeros with `ReturnCode.Failure`
# (exactly-singular) or an overflowing solution with `ReturnCode.Success`
# (numerically singular). The default now falls back to column-pivoted QR, which
# truncates the rank the same way `A \ b` does.
# Regression test for https://github.com/SciML/LinearSolve.jl/issues/531
@testset "Rank-deficient least squares" begin
    # Seeded: these probe behavior around the rank-detection threshold, so an
    # unlucky draw decides which side of the cutoff a matrix falls on. Keep them
    # reproducible rather than intermittently red.
    Random.seed!(0x0531)

    @testset "tall, exactly rank-deficient" begin
        A = rand(10, 4)
        A[:, 1] .= 0    # a column of all zeros
        b = rand(10)
        res = A \ b
        sol = solve(LinearProblem(copy(A), copy(b)))
        @test sol.retcode === ReturnCode.Success
        @test sol.u ≈ res
        # Column-pivoted QR and SVD reach the same answer directly.
        @test solve(LinearProblem(copy(A), copy(b)), QRFactorization(ColumnNorm())).u ≈ res
        @test solve(LinearProblem(copy(A), copy(b)), SVDFactorization()).u ≈ res
    end

    @testset "tall, rank-deficient by duplicate column" begin
        A = rand(10, 4)
        A[:, 3] .= A[:, 2]
        b = rand(10)
        sol = solve(LinearProblem(copy(A), copy(b)))
        @test sol.retcode === ReturnCode.Success
        @test sol.u ≈ A \ b
    end

    @testset "tall, numerically rank-deficient" begin
        # Not exactly singular, so the factorization "succeeds" and the failure is
        # silent without the rank check.
        #
        # The scaling has to put the deficiency clearly below the rank threshold.
        # `_qr_rank_deficient` compares against `min(m, n) * eps * max|R[i, i]|`,
        # the same relative cutoff LAPACK's `xGELSY` (and therefore `A \ b`) uses,
        # and a ~1e-14 scaling lands *on* that boundary: once the second column is
        # orthogonalized against the first, `R[2, 2]` is eps-level noise that
        # straddles the cutoff from draw to draw. There, unpivoted QR plus this
        # heuristic and the pivoted QR behind `\` can legitimately disagree --
        # measured over 300 seeds, a 1e-14 scaling disagreed on ~3% of draws, which
        # is what made this test intermittently fail. 1e-18 is unambiguously
        # rank-deficient and agreed on every draw.
        A = rand(10, 4)
        A[:, 1] .= 1.0e-18 .* A[:, 2]
        b = rand(10)
        sol = solve(LinearProblem(copy(A), copy(b)))
        @test sol.retcode === ReturnCode.Success
        @test sol.u ≈ A \ b
        @test all(isfinite, sol.u)
    end

    @testset "wide, rank-deficient" begin
        A = rand(3, 6)
        A[:, 2] .= 0
        b = rand(3)
        sol = solve(LinearProblem(copy(A), copy(b)))
        @test sol.retcode === ReturnCode.Success
        @test sol.u ≈ A \ b
    end

    @testset "square but singular" begin
        A = rand(5, 5)
        A[:, 3] .= 0
        b = rand(5)
        sol = solve(
            LinearProblem(copy(A), copy(b)),
            assumptions = OperatorAssumptions(
                true; condition = OperatorCondition.VeryIllConditioned
            )
        )
        @test sol.retcode === ReturnCode.Success
        @test sol.u ≈ qr(A, ColumnNorm()) \ b
    end

    @testset "cache reuse after the pivoted-QR fallback" begin
        A = rand(10, 4)
        A[:, 1] .= 0
        b1 = rand(10)
        b2 = rand(10)
        cache = init(LinearProblem(copy(A), copy(b1)))
        @test solve!(cache).u ≈ A \ b1
        @test cache.cacheval.fell_back_to_qr
        # Only `b` changes: the pivoted QR is reused, not the unpivoted one.
        cache.b = b2
        @test solve!(cache).u ≈ A \ b2
        # A fresh full-rank `A` resets the fallback and stays on unpivoted QR.
        # `cache.A = X` stores `X` itself and the in-place QR overwrites it, so
        # compute the reference first and hand the cache a copy.
        A_full = rand(10, 4)
        res_full = A_full \ b2
        cache.A = copy(A_full)
        @test !cache.cacheval.fell_back_to_qr
        @test solve!(cache).u ≈ res_full
        @test !cache.cacheval.fell_back_to_qr
    end

    @testset "full rank does not trigger the fallback" begin
        A = rand(10, 4)
        b = rand(10)
        cache = init(LinearProblem(copy(A), copy(b)))
        @test solve!(cache).u ≈ A \ b
        @test !cache.cacheval.fell_back_to_qr
    end
end

# Least-squares Krylov solvers with preconditioning: identity-equivalent counting
# preconditioner verifies Pl/Pr are actually forwarded to Krylov (not silently dropped).
mutable struct CountingDiagPrec
    d::Vector{Float64}
    calls::Int
end
CountingDiagPrec(d::AbstractVector) = CountingDiagPrec(collect(Float64, d), 0)
Base.size(P::CountingDiagPrec) = (length(P.d), length(P.d))
Base.size(P::CountingDiagPrec, ::Integer) = length(P.d)
Base.eltype(::CountingDiagPrec) = Float64
function LinearAlgebra.ldiv!(y::AbstractVector, P::CountingDiagPrec, x::AbstractVector)
    P.calls += 1
    @. y = x / P.d
    return y
end
function LinearAlgebra.ldiv!(P::CountingDiagPrec, x::AbstractVector)
    P.calls += 1
    @. x = x / P.d
    return x
end


@testset "LS family preconditioning" begin
    m, n = 30, 10
    A = randn(m, n)
    b = randn(m)
    res = A \ b

    ls_algs = [
        (KrylovJL_LSMR(), "LSMR", :both),
        (KrylovJL(KrylovAlg = Krylov.lsqr!), "LSQR", :both),
        (KrylovJL(KrylovAlg = Krylov.lslq!), "LSLQ", :both),
        (KrylovJL(KrylovAlg = Krylov.cgls!), "CGLS", :left_only),
        (KrylovJL(KrylovAlg = Krylov.crls!), "CRLS", :left_only),
    ]

    for (alg, name, support) in ls_algs
        @testset "$name" begin
            Pl = CountingDiagPrec(ones(m))
            Pr = CountingDiagPrec(ones(n))
            sol = solve(LinearProblem(A, b), alg; Pl = Pl, Pr = Pr)
            @test sol.u ≈ res
            @test Pl.calls > 0
            support === :both && @test Pr.calls > 0
        end
    end
end
