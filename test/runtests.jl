using LinearSolve
using Test

@testset "LinearSolve.jl" begin
    using LinearAlgebra
    n = 100
    x = Array(range(start=-1,stop=1,length=n))
    uu= @. sin(pi*x)

    dx = 2/(n-1)

    AA = Tridiagonal(ones(n-1),-2ones(n),ones(n-1))/(dx*dx)
    bb = @. -(pi^2)*uu
    id = Matrix(I,n,n)
    R  = id[2:end-1,:]

    A = R * AA * R'
    u = R * uu
    b = A * u #R * bb

    @test isapprox(A*u,b;atol=1e-2)
    prob = LinearProblem(A, b)

    x = zero(b)

    # Factorization
    @test A * solve(prob, LUFactorization();)  ≈ b
    @test A * solve(prob, QRFactorization();)  ≈ b
    @test A * solve(prob, SVDFactorization();) ≈ b

    # Krylov
    @test A * solve(prob, KrylovJL(A, b)) ≈ b

    # make algorithm callable - interoperable with DiffEq ecosystem
    @test A * LUFactorization()(x,A,b)  ≈ b 
    @test A * QRFactorization()(x,A,b)  ≈ b 
    @test A * SVDFactorization()(x,A,b) ≈ b 
    @test A * KrylovJL()(x,A,b)         ≈ b 

    # in place
    LUFactorization()(x,A,b) 
    @test A * x ≈ b
    QRFactorization()(x,A,b) 
    @test A * x ≈ b
    SVDFactorization()(x,A,b)
    @test A * x ≈ b
    KrylovJL()(x,A,b)        
    @test A * x ≈ b

    # test on some ODEProblem
#   using OrdinaryDiffEq
end
