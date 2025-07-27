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
    KrylovJL_LSMR
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
        LinearSolve.DefaultAlgorithmChoice.LUFactorization; safetyfallback = false),
    RFLUFactorization(),
    NormalCholeskyFactorization()
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
    LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.LUFactorization)
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
    SVDFactorization()
)
@testset "StaticArray Success" begin
    A = Float64[1 2 3; 4 3.5 1.7; 5.2 1.8 9.7]
    A = A*A'
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
