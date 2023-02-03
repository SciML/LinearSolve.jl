#
using LinearSolve, LinearAlgebra, Test
using LinearSolve: _isidentity_struct

N = 4

@testset "Traits" begin

    @test _isidentity_struct(I)
    @test _isidentity_struct(1.0 * I)
    @test _isidentity_struct(SciMLBase.IdentityOperator{N}())
    @test _isidentity_struct(SciMLBase.DiffEqIdentity(rand(4)))
    @test ! _isidentity_struct(2.0 * I)
    @test ! _isidentity_struct(rand(N, N))
    @test ! _isidentity_struct(Matrix(I, N, N))
end
