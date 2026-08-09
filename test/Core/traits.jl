#
using LinearSolve, LinearAlgebra, Test
using LinearSolve: _isidentity_struct
using SciMLOperators: IdentityOperator

N = 4

@testset "Traits" begin
    @test _isidentity_struct(I)
    @test _isidentity_struct(1.0 * I)
    @test _isidentity_struct(IdentityOperator(N))
    @test !_isidentity_struct(2.0 * I)
    @test !_isidentity_struct(rand(N, N))
    @test !_isidentity_struct(Matrix(I, N, N))
end
