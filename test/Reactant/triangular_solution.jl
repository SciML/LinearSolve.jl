using LinearAlgebra: UnitLowerTriangular
using LinearSolve
using Reactant
using Test

function triangular_solution(A, b)
    return solve(LinearProblem(UnitLowerTriangular(A), b), DirectLdiv!()).u
end

@testset "native unit-triangular solution array" begin
    A = Reactant.to_rarray(Float32[0 2; 3 0])
    b = Reactant.to_rarray(Float32[1, 3])
    @test Array(@jit triangular_solution(A, b)) == Float32[1, 0]
end
