using LinearSolve
using Test

@testset "LinearSolve.jl" begin
    using LinearAlgebra
    n = 32
    dx = 2/(n-1)

    xx = Array(range(start=-1,stop=1,length=n))
    AA = Tridiagonal(ones(n-1),-2ones(n),ones(n-1))/(dx*dx) # rank-deficient sys
    uu = @. sin(pi*xx)
    bb = AA * uu #@. -(pi^2)*uu

    id = Matrix(I,n,n)
    R  = id[2:end-1,:]

    x = R * xx
    A = R * AA * R' # full rank system
    u = R * uu
    b = A * u #R * bb

    # test on some ODEProblem
#   using OrdinaryDiffEq
#   #   add this problem to DiffEqProblemLibrary
#   kx = 1
#   kt = 1
#   ut(x,t) = sin(kx*pi*x)*cos(kt*pi*t)
#   ic(x)   = ut(x,0.0)
#   f(x,t)  = ut(x,t)*(kx*pi)^2 - sin(kx*pi*x)*sin(kt*pi*t)*(kt*pi)
#   u0 = ic.(x)
#   dudt!(du,u,p,t) = -A*u + f.(x,t)
#   dt = 0.01
#   tspn = (0.0,1.0)
#   func = ODEFunction(dudt!)
#   prob = ODEProblem(func,u0,tspn)

    x = zero(b)
    A1 =  A; b1 =  b
    A2 = 2A; b2 = 3b
    A3 = 3A; b3 = 2b

    prob1 = LinearProblem(A1, b1; u0=x)
    prob2 = LinearProblem(A2, b2; u0=x)
    prob3 = LinearProblem(A3, b3; u0=x)

    for alg in (
                :LUFactorization,
                :QRFactorization,
                :SVDFactorization,

#               :DefaultLinSolve,

                :KrylovJL,
#               :KrylovJL,
#               :KrylovKitJL,

               )
        @eval begin
            y = solve($prob1, $alg())
            @test $A1 *  y ≈ $b1
            @test $A1 * $x ≈ $b1

            y = $alg()($x, $A2, $b2)
            @test $A2 *  y ≈ $b2
            @test $A2 * $x ≈ $b2

            cache = SciMLBase.init($prob1, $alg())
            y = cache($x, $A3, $b3)
            @test $A3 * $x ≈ $b3
            @test $A3 *  y ≈ $b3
        end
    end


#   using OrdinaryDiffEq
#   using DiffEqProblemLibrary.ODEProblemLibrary
#   ODEProblemLibrary.importodeproblems()
#   prob = ODEProblemLibrary.prob_ode_linear
#   @show prob
#   sol  = solve(prob, Rodas5(linsolve=KrylovJL()); saveat=0.1)
#   @show sol.retcode

end
