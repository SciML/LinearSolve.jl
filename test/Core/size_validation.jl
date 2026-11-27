using LinearSolve, LinearAlgebra, SparseArrays, StaticArrays, ForwardDiff, Test
using RecursiveFactorization
using SciMLOperators: MatrixOperator, FunctionOperator

@testset "Problem and cache size validation (#1377)" begin
    A10 = rand(10, 10) + 10I
    b10 = rand(10)

    @testset "init / solve reject mismatched b across algorithms" begin
        algs = (
            nothing,
            LUFactorization(),
            GenericLUFactorization(),
            RFLUFactorization(),
            QRFactorization(),
            QRFactorization(ColumnNorm()),
            SVDFactorization(),
            CholeskyFactorization(),
            KrylovJL_GMRES(),
            SimpleLUFactorization(),
            SimpleGMRES(),
        )
        Aspd = A10' * A10 + 10I
        for alg in algs
            A_use = alg isa CholeskyFactorization ? Aspd : A10
            for bad_b in (rand(8), rand(12), rand(8, 2), rand(12, 2))
                prob = LinearProblem(copy(A_use), copy(bad_b))
                @test_throws DimensionMismatch init(prob, alg)
                @test_throws DimensionMismatch solve(prob, alg)
            end
        end

        As = sparse(A10)
        for alg in (nothing, KLUFactorization(), PureKLUFactorization(), UMFPACKFactorization())
            for bad_b in (rand(8), rand(12))
                prob = LinearProblem(As, bad_b)
                @test_throws DimensionMismatch init(prob, alg)
                @test_throws DimensionMismatch solve(prob, alg)
            end
        end
    end

    @testset "init / solve reject mismatched u0 and invalid shapes" begin
        for bad_u0 in (rand(8), rand(12), zeros(10, 1), zeros(10, 2))
            @test_throws DimensionMismatch init(LinearProblem(A10, b10; u0 = bad_u0))
        end
        @test_throws DimensionMismatch init(LinearProblem(rand(10), rand(10)))
        @test_throws DimensionMismatch init(LinearProblem(rand(2, 2, 2), rand(2)))
        @test_throws DimensionMismatch init(LinearProblem(A10, zeros(10, 1, 1)))
        @test_throws DimensionMismatch init(LinearProblem(A10, b10; u0 = zeros(10, 1, 1)))
    end

    @testset "Non-square and batched RHS dimensions" begin
        Atall = rand(10, 4)
        btall = rand(10)
        @test solve(LinearProblem(Atall, btall)).u ≈ Atall \ btall
        @test_throws DimensionMismatch init(LinearProblem(Atall, rand(4)))
        @test_throws DimensionMismatch init(LinearProblem(Atall, btall; u0 = zeros(10)))
        @test solve(LinearProblem(Atall, btall; u0 = zeros(4))).u ≈ Atall \ btall

        B10 = rand(10, 3)
        @test solve(LinearProblem(A10, B10)).u ≈ A10 \ B10
        @test solve(LinearProblem(A10, B10; u0 = zeros(10, 3))).u ≈ A10 \ B10
        @test_throws DimensionMismatch init(LinearProblem(A10, B10; u0 = zeros(10)))
        @test_throws DimensionMismatch init(LinearProblem(A10, B10; u0 = zeros(10, 2)))
        @test_throws DimensionMismatch init(LinearProblem(A10, B10; u0 = zeros(8, 3)))
    end

    @testset "StaticArrays and SciMLOperators" begin
        As = SMatrix{10, 10}(A10)
        @test_throws DimensionMismatch solve(LinearProblem(As, @SVector rand(8)))
        @test_throws DimensionMismatch solve(LinearProblem(As, @SVector rand(12)), LUFactorization())
        # Singular static matrix must still validate b and u0 before returning Failure.
        As_sing = @SMatrix zeros(10, 10)
        @test_throws DimensionMismatch solve(LinearProblem(As_sing, @SVector rand(8)), LUFactorization())
        @test_throws DimensionMismatch solve(
            LinearProblem(As_sing, @SVector rand(10); u0 = @SVector rand(8)),
            LUFactorization()
        )

        op = MatrixOperator(A10)
        @test_throws DimensionMismatch solve(LinearProblem(op, rand(8)))
        fop = FunctionOperator((w, v, u, p, t) -> mul!(w, A10, v), zeros(10), zeros(10))
        @test_throws DimensionMismatch solve(LinearProblem(fop, rand(8)))
    end

    @testset "cache.b / cache.A / cache.u / reinit! validation" begin
        for alg in (
                nothing, LUFactorization(), GenericLUFactorization(),
                QRFactorization(), KrylovJL_GMRES(),
            )
            c = init(LinearProblem(copy(A10), copy(b10)), alg)
            @test solve!(c).u ≈ A10 \ b10 rtol = 1.0e-6

            c.b = rand(8)
            @test_throws DimensionMismatch solve!(c)
            c.b = rand(12)
            @test_throws DimensionMismatch solve!(c)
            c.b = copy(b10)

            c.A = rand(12, 12) + 12I
            @test_throws DimensionMismatch solve!(c)
            c.A = rand(8, 8) + 8I
            @test_throws DimensionMismatch solve!(c)
            c.A = copy(A10)

            c.u = rand(8)
            @test_throws DimensionMismatch solve!(c)
            c.u = zeros(10)
            @test solve!(c).u ≈ A10 \ b10 rtol = 1.0e-6
        end

        # reinit! validates atomically before mutating cache fields
        c = init(LinearProblem(copy(A10), copy(b10)))
        @test solve!(c).u ≈ A10 \ b10
        @test_throws DimensionMismatch reinit!(c; b = rand(8))
        @test length(c.b) == 10
        @test_throws DimensionMismatch reinit!(c; A = rand(12, 12) + 12I)
        @test size(c.A) == (10, 10)
        @test solve!(c).u ≈ A10 \ b10

        A12 = rand(12, 12) + 12I
        b12 = rand(12)
        reinit!(c; A = copy(A12), b = copy(b12), u = zeros(12))
        @test solve!(c).u ≈ A12 \ b12

        # Sequential resize in b-first order also works once all fields match
        c2 = init(LinearProblem(copy(A10), copy(b10)), LUFactorization())
        c2.b = copy(b12)
        c2.A = copy(A12)
        c2.u = zeros(12)
        @test solve!(c2).u ≈ A12 \ b12
    end

    @testset "ForwardDiff DualLinearCache validation" begin
        p = ForwardDiff.Dual(2.0, 1.0)
        Ad = [i == j ? 10.0 + p : 0.1 + 0 * p for i in 1:4, j in 1:4]
        bd = [float(i) * p for i in 1:4]

        @test_throws DimensionMismatch init(LinearProblem(Ad, bd[1:3]))
        @test_throws DimensionMismatch solve(LinearProblem(Ad, [1.0, 2.0, 3.0]), LUFactorization())
        @test_throws DimensionMismatch init(LinearProblem(Ad, bd; u0 = zeros(typeof(p), 3)))

        for alg in (LUFactorization(), GenericLUFactorization(), nothing)
            dc = init(LinearProblem(copy(Ad), copy(bd)), alg)
            ref = Ad \ bd
            @test ForwardDiff.value.(solve!(dc).u) ≈ ForwardDiff.value.(ref)

            @test_throws DimensionMismatch (dc.b = bd[1:3])
            @test_throws DimensionMismatch (dc.b = [bd; p])
            @test_throws DimensionMismatch (dc.A = fill(0.1 + p, 5, 5) + 5I)
            @test_throws DimensionMismatch (dc.u = zeros(typeof(p), 3))
            @test_throws DimensionMismatch reinit!(dc; A = copy(Ad), b = bd[1:3])

            # Failed assignments must not corrupt the inner primal buffer
            sol = solve!(dc)
            @test ForwardDiff.value.(sol.u) ≈ ForwardDiff.value.(ref)
            @test ForwardDiff.partials.(sol.u) ≈ ForwardDiff.partials.(ref)
        end
    end
end
