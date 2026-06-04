using LinearAlgebra
using LinearSolve
using MPI
using MUMPS
using SparseArrays
using Test

MPI.Initialized() || MPI.Init()
const MUMPSExt = Base.get_extension(LinearSolve, :LinearSolveMUMPSExt)

function residual_ok(A, x, b; atol = 1.0e-8, rtol = 1.0e-8)
    return isapprox(A * x, b; atol = atol, rtol = rtol)
end

@testset "MUMPSFactorization real sparse solve" begin
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

    alg = MUMPSFactorization()
    sol = LinearSolve.solve(LinearProblem(A, b1), alg)
    @test sol.retcode == ReturnCode.Success
    @test residual_ok(A, sol.u, b1)
    MUMPSExt.cleanup_mumps_cache!(sol)

    cache = LinearSolve.init(LinearProblem(A, b1), alg)
    sol1 = LinearSolve.solve!(cache)
    @test sol1.retcode == ReturnCode.Success
    @test residual_ok(A, sol1.u, b1)

    cache.b = b2
    sol2 = LinearSolve.solve!(cache)
    @test sol2.retcode == ReturnCode.Success
    @test residual_ok(A, sol2.u, b2)

    cache.A = A2
    sol3 = LinearSolve.solve!(cache)
    @test sol3.retcode == ReturnCode.Success
    @test residual_ok(A2, sol3.u, b2)
    MUMPSExt.cleanup_mumps_cache!(cache)
end

@testset "MUMPSFactorization complex and transposed solve" begin
    A = sparse(
        ComplexF64[
            3.0 + 0im 1.0 - 1im
            2.0 + 1im 4.0 + 0im
        ]
    )
    x = ComplexF64[1.0 - 1im, 2.0 + 0.5im]
    b = transpose(A) * x

    sol = LinearSolve.solve(
        LinearProblem(A, b),
        MUMPSFactorization(; transposed = true)
    )
    @test sol.retcode == ReturnCode.Success
    @test sol.u ≈ x atol = 1.0e-8 rtol = 1.0e-8
    MUMPSExt.cleanup_mumps_cache!(sol)
end

@testset "MUMPSFactorization rejects unsupported scalar types" begin
    A = sparse(
        BigFloat[
            4 1
            2 3
        ]
    )
    b = BigFloat[1, 2]

    sol = LinearSolve.solve(
        LinearProblem(A, b),
        MUMPSFactorization()
    )
    @test sol.retcode == ReturnCode.Failure
end
