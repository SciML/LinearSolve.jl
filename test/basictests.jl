using LinearSolve, LinearAlgebra, SparseArrays
using Test

n = 8
A = Matrix(I,n,n)
b = ones(n)
A1 = A/1; b1 = rand(n); x1 = zero(b)
A2 = A/2; b2 = rand(n); x2 = zero(b)

prob1 = LinearProblem(A1, b1; u0=x1)
prob2 = LinearProblem(A2, b2; u0=x2)

cache_kwargs = (;verbose=true, abstol=1e-8, reltol=1e-8, maxiter=30,)

function test_interface(alg, prob1, prob2)
    A1 = prob1.A; b1 = prob1.b; x1 = prob1.u0
    A2 = prob2.A; b2 = prob2.b; x2 = prob2.u0

    y = solve(prob1, alg; cache_kwargs...)
    @test A1 *  y  ≈ b1

    cache = SciMLBase.init(prob1,alg; cache_kwargs...) # initialize cache
    y = solve(cache)
    @test A1 *  y  ≈ b1

    cache = LinearSolve.set_A(cache,copy(A2))
    y = solve(cache)
    @test A2 *  y  ≈ b1

    cache = LinearSolve.set_b(cache,b2)
    y = solve(cache)
    @test A2 *  y  ≈ b2

    return
end

@testset "LinearSolve" begin

@testset "Default Linear Solver" begin
    test_interface(nothing, prob1, prob2)

    A1 = prob1.A; b1 = prob1.b; x1 = prob1.u0
    y = solve(prob1)
    @test A1 *  y  ≈ b1

    _prob = LinearProblem(SymTridiagonal(A1), b1; u0=x1)
    y = solve(_prob)
    @test A1 *  y  ≈ b1

    _prob = LinearProblem(Tridiagonal(A1), b1; u0=x1)
    y = solve(_prob)
    @test A1 *  y  ≈ b1

    _prob = LinearProblem(Symmetric(A1), b1; u0=x1)
    y = solve(_prob)
    @test A1 *  y  ≈ b1

    _prob = LinearProblem(Hermitian(A1), b1; u0=x1)
    y = solve(_prob)
    @test A1 *  y  ≈ b1


    _prob = LinearProblem(sparse(A1), b1; u0=x1)
    y = solve(_prob)
    @test A1 *  y  ≈ b1
end

@testset "UMFPACK Factorization" begin
    A1 = A/1; b1 = rand(n); x1 = zero(b)
    A2 = A/2; b2 = rand(n); x2 = zero(b)

    prob1 = LinearProblem(sparse(A1), b1; u0=x1)
    prob2 = LinearProblem(sparse(A2), b2; u0=x2)
    test_interface(UMFPACKFactorization(), prob1, prob2)

    # Test that refactoring wrong throws.
    cache = SciMLBase.init(prob1,UMFPACKFactorization(); cache_kwargs...) # initialize cache
    y = solve(cache)
    cache = LinearSolve.set_A(cache,sprand(n, n, 0.8))
    @test_throws ArgumentError solve(cache)
end

@testset "KLU Factorization" begin
    A1 = A/1; b1 = rand(n); x1 = zero(b)
    A2 = A/2; b2 = rand(n); x2 = zero(b)

    prob1 = LinearProblem(sparse(A1), b1; u0=x1)
    prob2 = LinearProblem(sparse(A2), b2; u0=x2)
    test_interface(KLUFactorization(), prob1, prob2)

    # Test that refactoring wrong throws.
    cache = SciMLBase.init(prob1,KLUFactorization(); cache_kwargs...) # initialize cache
    y = solve(cache)
    X = copy(A1)
    X[8,8] = 0.0
    X[7,8] = 1.0
    cache = LinearSolve.set_A(cache,sparse(X))
    @test_throws ArgumentError solve(cache)
end

@testset "Concrete Factorizations" begin
    for alg in (
                LUFactorization(),
                QRFactorization(),
                SVDFactorization(),
                RFLUFactorization()
               )
        @testset "$alg" begin
            test_interface(alg, prob1, prob2)
        end
    end
end

@testset "Generic Factorizations" begin
    for fact_alg in (
                     lu, lu!,
                     qr, qr!,
                     cholesky,
                     #cholesky!,
    #                ldlt, ldlt!,
                     bunchkaufman, bunchkaufman!,
                     lq, lq!,
                     svd, svd!,
                     LinearAlgebra.factorize,
                    )
        @testset "fact_alg = $fact_alg" begin
            alg = GenericFactorization(fact_alg=fact_alg)
            test_interface(alg, prob1, prob2)
        end
    end
end

@testset "KrylovJL" begin
    kwargs = (;gmres_restart=5,)
    for alg in (
                ("Default",KrylovJL(kwargs...)),
                ("CG",KrylovJL_CG(kwargs...)),
                ("GMRES",KrylovJL_GMRES(kwargs...)),
    #           ("BICGSTAB",KrylovJL_BICGSTAB(kwargs...)),
                ("MINRES",KrylovJL_MINRES(kwargs...)),
               )
        @testset "$(alg[1])" begin
            test_interface(alg[2], prob1, prob2)
        end
    end
end

@testset "IterativeSolversJL" begin
    kwargs = (;gmres_restart=5,)
    for alg in (
                ("Default", IterativeSolversJL(kwargs...)),
                ("CG", IterativeSolversJL_CG(kwargs...)),
                ("GMRES",IterativeSolversJL_GMRES(kwargs...)),
    #           ("BICGSTAB",IterativeSolversJL_BICGSTAB(kwargs...)),
    #            ("MINRES",IterativeSolversJL_MINRES(kwargs...)),
               )
        @testset "$(alg[1])" begin
            test_interface(alg[2], prob1, prob2)
        end
    end
end

@testset "PardisoJL" begin
    @test_throws UndefVarError alg = PardisoJL()

    using Pardiso, SparseArrays

    A1 = sparse([ 1. 0 -2  3
                 0  5  1  2
                -2  1  4 -7
                 3  2 -7  5 ])
    b1 = rand(4)
    prob1 = LinearProblem(A1, b1)

    lambda = 3
    e = ones(n)
    e2 = ones(n-1)
    A2 = spdiagm(-1 => im*e2, 0 => lambda*e, 1 => -im*e2)
    b2 = rand(n) + im * zeros(n)

    prob2 = LinearProblem(A2, b2)

    for alg in (
                PardisoJL(),
                MKLPardisoFactorize(),
                MKLPardisoIterate(),
               )

        u = solve(prob1, alg; cache_kwargs...).u
        @test A1 * u ≈ b1

        u = solve(prob2, alg; cache_kwargs...).u
        @test eltype(u) <: Complex
        @test_broken A2 * u ≈ b2
    end

end

@testset "Preconditioners" begin
    @testset "Vector Diagonal Preconditioner" begin
        s = rand(n)
        Pl, Pr = Diagonal(s),LinearSolve.InvPreconditioner(Diagonal(s))

        x = rand(n,n)
        y = rand(n,n)

        mul!(y, Pl, x); @test y ≈ s .* x
        mul!(y, Pr, x); @test y ≈ s .\ x

        y .= x; ldiv!(Pl, x); @test x ≈ s .\ y
        y .= x; ldiv!(Pr, x); @test x ≈ s .* y

        ldiv!(y, Pl, x); @test y ≈ s .\ x
        ldiv!(y, Pr, x); @test y ≈ s .* x
    end

    @testset "ComposePreconditioenr" begin
        s1 = rand(n)
        s2 = rand(n)

        x = rand(n,n)
        y = rand(n,n)

        P1 = Diagonal(s1)
        P2 = Diagonal(s2)

        P  = LinearSolve.ComposePreconditioner(P1,P2)

        # ComposePreconditioner
        ldiv!(y, P, x);      @test y ≈ ldiv!(P2, ldiv!(P1, x))
        y .= x; ldiv!(P, x); @test x ≈ ldiv!(P2, ldiv!(P1, y))
    end
end

end # testset
