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

@testset "Default Linear Solver" begin
    test_interface(nothing, prob1, prob2)

    A1 = prob1.A; b1 = prob1.b; x1 = prob1.u0
    y = solve(prob1)
    @test A1 *  y  ≈ b1

    _prob = LinearProblem(SymTridiagonal(A1.A), b1; u0=x1)
    y = solve(prob1)
    @test A1 *  y  ≈ b1

    _prob = LinearProblem(Tridiagonal(A1.A), b1; u0=x1)
    y = solve(prob1)
    @test A1 *  y  ≈ b1

    _prob = LinearProblem(Symmetric(A1.A), b1; u0=x1)
    y = solve(prob1)
    @test A1 *  y  ≈ b1

    _prob = LinearProblem(Hermitian(A1.A), b1; u0=x1)
    y = solve(prob1)
    @test A1 *  y  ≈ b1

    _prob = LinearProblem(sparse(A1.A), b1; u0=x1)
    y = solve(prob1)
    @test A1 *  y  ≈ b1
end

@testset "Concrete Factorizations" begin
    for alg in (
                LUFactorization(),
                QRFactorization(),
                SVDFactorization()
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
