using LinearSolve
using Test

@testset "LinearSolve.jl" begin
    using LinearAlgebra
    n = 100
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

    x    = zero(b)
    prob = LinearProblem(A, b;u0=x)

    # Factorization
    for alg in (:LUFactorization, :QRFactorization, :SVDFactorization,
                :KrylovJL)
        @eval begin
            @test $A * solve($prob, $alg();) ≈ $b
            $alg()($x,$A,$b)
            @test $A * $x ≈ $b
        end
    end

    # test on some ODEProblem
    using OrdinaryDiffEq
    using DiffEqProblemLibrary.ODEProblemLibrary
    ODEProblemLibrary.importodeproblems()

    #   add this problem to DiffEqProblemLibrary
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

    prob = ODEProblemLibrary.prob_ode_linear
    sol  = solve(prob, Rodas5(linsolve=SVDFactorization()); saveat=0.1)
    @show sol.retcode

end
