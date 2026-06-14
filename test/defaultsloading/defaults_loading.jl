using SparseArrays
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
# Generic-eltype (BigFloat) sparse LU defaults to PureKLU (a hard dependency)
# since #1037, so this solves without `using Sparspak` being loaded first.
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

using Sparspak
sol = solve(prob).u
@test sol isa Vector{BigFloat}
