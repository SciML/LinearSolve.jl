using LinearSolve, LinearAlgebra, RecursiveFactorization, StaticArrays, Test

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

    # Nonsingular static solves stay bit-identical to `\`
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
            @test solve(LinearProblem(A, rhs)).u === A \ rhs
            @test solve(LinearProblem(A, rhs), LUFactorization()).u === lu(A) \ rhs
        end
    end

    # Non-square static problems are unchanged by the singular handling
    Ans = SMatrix{3, 2}([1.0 2.0; 3.0 4.0; 5.0 6.0])
    bns = SA[1.0, 2.0, 3.0]
    solns = solve(LinearProblem(Ans, bns))
    @test SciMLBase.successful_retcode(solns.retcode)
    @test solns.u ≈ Matrix(Ans) \ Vector(bns)
end
