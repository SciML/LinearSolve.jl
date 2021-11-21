using LinearSolve
using Test

@testset "LinearSolve.jl" begin
    using LinearAlgebra
    n = 32

    A = Matrix(I,n,n)
    b = ones(n)
    A1 = A/1; b1 = rand(n); x1 = zero(b)
    A2 = A/2; b2 = rand(n); x2 = zero(b)
    A3 = A/3; b3 = rand(n); x3 = zero(b)

    prob1 = LinearProblem(A1, b1; u0=x1)
    prob2 = LinearProblem(A2, b2; u0=x2)
    prob3 = LinearProblem(A3, b3; u0=x3)

    for alg in (
                :LUFactorization,
                :QRFactorization,
                :SVDFactorization,

#               :DefaultLinSolve,

                :KrylovJL, :KrylovJL_CG, :KrylovJL_GMRES, :KrylovJL_BICGSTAB,
                :KrylovJL_MINRES,
                :IterativeSolversJL, :IterativeSolversJL_GMRES,
                :IterativeSolversJL_BICGSTAB, :IterativeSolversJL_MINRES
               )
        @eval begin
            y = solve($prob1, $alg())
            @test $A1 *  y  ≈ $b1 # out of place
            @test $A1 * $x1 ≈ $b1 # in place

            y = $alg()($x2, $A2, $b2)              # alg is callable
            @test $A2 *  y  ≈ $b2
            @test $A2 * $x2 ≈ $b2

            cache = SciMLBase.init($prob1, $alg()) # initialize cache
            y = cache($x3, $A1, $b1)               # cache is callable
            @test $A1 *  y  ≈ $b1
            @test $A1 * $x3 ≈ $b1

            y = cache($x3, $A1, $b2)               # reuse factorization
            @test $A1 *  y  ≈ $b2
            @test $A1 * $x3 ≈ $b2

            y = cache($x3, $A2, $b3)               # new factorization
            @test $A2 *  y  ≈ $b3                  # same old cache
            @test $A2 * $x3 ≈ $b3
        end
    end

end
