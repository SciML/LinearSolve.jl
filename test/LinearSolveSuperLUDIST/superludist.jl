using LinearSolve
using MPI
using SparseArrays
using SuperLUDIST
using Test

MPI.Initialized() || MPI.Init()
const SuperLUDISTExt = Base.get_extension(LinearSolve, :LinearSolveSuperLUDISTExt)

function residual_ok(A, x, b; atol = 1.0e-8, rtol = 1.0e-8)
    return isapprox(A * x, b; atol = atol, rtol = rtol)
end

@testset "SuperLUDISTFactorization sparse solve and reuse" begin
    A = sparse(
        [
            4.0 1.0 0.0
            2.0 3.0 1.0
            0.0 1.0 2.0
        ]
    )
    b1 = [1.0, 2.0, 3.0]
    b2 = [3.0, 2.0, 1.0]
    A2 = sparse(
        [
            5.0 1.0 0.0
            2.0 4.0 1.0
            0.0 1.0 3.0
        ]
    )

    alg = SuperLUDISTFactorization(; comm = MPI.COMM_SELF)
    lincache = init(LinearProblem(A, b1), alg)
    sol = solve!(lincache)
    @test sol.retcode == ReturnCode.Success
    @test residual_ok(A, sol.u, b1)
    SuperLUDISTExt.cleanup_superludist_cache!(lincache)

    cache = init(LinearProblem(A, b1), alg)
    sol1 = solve!(cache)
    @test sol1.retcode == ReturnCode.Success
    @test residual_ok(A, sol1.u, b1)

    cache.b = b2
    sol2 = solve!(cache)
    @test sol2.retcode == ReturnCode.Success
    @test residual_ok(A, sol2.u, b2)

    cache.A = A2
    sol3 = solve!(cache)
    @test sol3.retcode == ReturnCode.Success
    @test residual_ok(A2, sol3.u, b2)
    SuperLUDISTExt.cleanup_superludist_cache!(cache)
end

@testset "SuperLUDISTFactorization rejects unsupported complex inputs" begin
    A = sparse(
        ComplexF64[
            3.0 + 0im 1.0 - 1im
            2.0 + 1im 4.0 + 0im
        ]
    )
    x = ComplexF64[1.0 - 1im, 2.0 + 0.5im]
    b = A * x

    sol = solve(LinearProblem(A, b), SuperLUDISTFactorization())
    @test sol.retcode == ReturnCode.Failure
end

@testset "SuperLUDISTFactorization rejects unsupported scalar types" begin
    A = sparse(
        BigFloat[
            4 1
            2 3
        ]
    )
    b = BigFloat[1, 2]

    sol = solve(LinearProblem(A, b), SuperLUDISTFactorization())
    @test sol.retcode == ReturnCode.Failure
end
