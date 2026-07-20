using LinearSolve, ComponentArrays, LinearAlgebra, Test
using LinearSolve: OperatorAssumptions

@testset "BLAS LU cache tracks ComponentMatrix factors" begin
    u0 = ComponentArray(a = ones(4, 2), b = ones(4, 2))
    ax = only(ComponentArrays.getaxes(u0))
    A = ComponentArray(Matrix{Float64}(I, length(u0), length(u0)), (ax, ax))
    b = copy(u0)
    assump = OperatorAssumptions(true)

    @test A isa DenseMatrix
    @test !(A isa Matrix)
    @test similar(A, 0, 0) isa Matrix
    @test lu!(copy(A)).factors isa ComponentMatrix

    for alg in (MKLLUFactorization(), OpenBLASLUFactorization())
        slot = LinearSolve.init_cacheval(
            alg,
            A,
            b,
            b,
            nothing,
            nothing,
            0,
            0.0,
            0.0,
            true,
            assump,
        )
        workspace = alg isa MKLLUFactorization ? slot[1] : slot
        @test workspace.factors isa ComponentMatrix
    end

    slot = LinearSolve.init_cacheval(
        AppleAccelerateLUFactorization(),
        A,
        b,
        b,
        nothing,
        nothing,
        0,
        0.0,
        0.0,
        true,
        assump,
    )
    @test slot.factors isa ComponentMatrix
    @test slot.ipiv isa Vector{Cint}
end

@testset "dense ComponentMatrix default solve" begin
    u0 = ComponentArray(a = ones(4, 2), b = ones(4, 2))
    ax = only(ComponentArrays.getaxes(u0))
    n = length(u0)
    M = Matrix{Float64}(I, n, n) .+ 0.01 .* reshape(1:(n * n), n, n)
    A = ComponentArray(M, (ax, ax))
    b = copy(u0)

    sol = solve(LinearProblem(A, b))
    @test norm(M * Array(sol.u) - Array(b)) < 1.0e-10

    sol = solve(LinearProblem(A, b), OpenBLASLUFactorization())
    @test norm(M * Array(sol.u) - Array(b)) < 1.0e-10
end
