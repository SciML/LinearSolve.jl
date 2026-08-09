using SparseArrays
using LinearAlgebra
using LinearSolve
using Test

n = 10
dx = 1 / n
dx2 = dx^-2
vals = Vector{BigFloat}(undef, 0)
cols = Vector{Int}(undef, 0)
rows = Vector{Int}(undef, 0)
for i in 1:n
    if i != 1
        push!(vals, dx2)
        push!(cols, i - 1)
        push!(rows, i)
    end
    push!(vals, -2dx2)
    push!(cols, i)
    push!(rows, i)
    if i != n
        push!(vals, dx2)
        push!(cols, i + 1)
        push!(rows, i)
    end
end
mat = sparse(rows, cols, vals, n, n)
rhs = big.(zeros(n))
rhs[begin] = rhs[end] = -2
prob = LinearProblem(mat, rhs)
@test Base.get_extension(LinearSolve, :LinearSolveSparspakExt) === nothing
@test LinearSolve.defaultalg(mat, rhs).alg ===
    LinearSolve.DefaultAlgorithmChoice.KLUFactorization
sol = solve(prob).u
@test sol isa Vector{BigFloat}

STRUMPACKExt = Base.get_extension(LinearSolve, :LinearSolveSTRUMPACKExt)
if STRUMPACKExt === nothing || !STRUMPACKExt.strumpack_isavailable()
    @test_throws ["STRUMPACKFactorization", "STRUMPACK_jll"] STRUMPACKFactorization()
else
    @test STRUMPACKFactorization() isa STRUMPACKFactorization
end

# no-RF dense band: blocked GenericLU owns N ≤ 32 everywhere, ≤ 128 under OpenBLAS
@test Base.get_extension(LinearSolve, :LinearSolveRecursiveFactorizationExt) === nothing
if LinearSolve.appleaccelerate_isavailable()
    @test LinearSolve.defaultalg(nothing, zeros(32)).alg ===
        LinearSolve.DefaultAlgorithmChoice.AppleAccelerateLUFactorization
else
    above_band = LinearSolve.usemkl ?
        LinearSolve.DefaultAlgorithmChoice.MKLLUFactorization :
        LinearSolve.DefaultAlgorithmChoice.LUFactorization
    @test LinearSolve.defaultalg(nothing, zeros(32)).alg ===
        LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization
    if LinearSolve.isopenblas()
        @test LinearSolve.defaultalg(nothing, zeros(128)).alg ===
            LinearSolve.DefaultAlgorithmChoice.GenericLUFactorization
        @test LinearSolve.defaultalg(nothing, zeros(129)).alg === above_band
    else
        @test LinearSolve.defaultalg(nothing, zeros(33)).alg === above_band
    end
    let A = rand(32, 32) + 32I, b = rand(32)
        sol = solve(LinearProblem(A, b))
        @test norm(A * sol.u - b) < 1.0e-8
    end
end

using Sparspak
sol = solve(prob).u
@test sol isa Vector{BigFloat}
