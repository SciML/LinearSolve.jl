using LinearSolve, RecursiveFactorization

alglist = (
    LUFactorization,
    QRFactorization,
    DiagonalFactorization,
    DirectLdiv!,
    SparspakFactorization,
    KLUFactorization,
    UMFPACKFactorization,
    KrylovJL_GMRES,
    GenericLUFactorization,
    RFLUFactorization,
    LDLtFactorization,
    BunchKaufmanFactorization,
    CHOLMODFactorization,
    SVDFactorization,
    CholeskyFactorization,
    NormalCholeskyFactorization,
    AppleAccelerateLUFactorization,
    MKLLUFactorization,
    KrylovJL_CRAIGMR,
    KrylovJL_LSMR
)

@testset "Success" begin
    for alg in alglist
        A = [2.0 1.0; -1.0 1.0]
        b = [-1.0, 1.0]
        prob = LinearProblem(A, b)
        linsolve = init(prob, alg)
        sol = solve!(linsolve)
        @test SciMLBase.successful_retcode(sol.retcode) || sol.retcode == ReturnCode.Default # The latter seems off...
    end
end

@testset "Failure" begin
    for alg in alglist
        A = [1.0 1.0; 1.0 1.0]
        b = [-1.0, 1.0]
        prob = LinearProblem(A, b)
        linsolve = init(prob, alg)
        sol = solve!(linsolve)
        @test !SciMLBase.successful_retcode(sol.retcode)
    end
end
