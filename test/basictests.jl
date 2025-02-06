using LinearSolve, LinearAlgebra, SparseArrays, MultiFloats, ForwardDiff
using SciMLOperators, RecursiveFactorization, Sparspak, FastLapackInterface
using IterativeSolvers, KrylovKit, MKL_jll, KrylovPreconditioners
using Test
import Random

const Dual64 = ForwardDiff.Dual{Nothing, Float64, 1}

n = 8
A = Matrix(I, n, n)
b = ones(n)
# Real-valued systems
A1 = A / 1;
b1 = rand(n);
x1 = zero(b);
# A2 is similar to A1; created to test cache reuse
A2 = A / 2;
b2 = rand(n);
x2 = zero(b);
# Complex systems + mismatched types with eltype(tol)
A3 = A1 .|> ComplexF32
b3 = b1 .|> ComplexF32
x3 = x1 .|> ComplexF32
# A4 is similar to A3; created to test cache reuse
A4 = A2 .|> ComplexF32
b4 = b2 .|> ComplexF32
x4 = x2 .|> ComplexF32

prob1 = LinearProblem(A1, b1; u0 = x1)
prob2 = LinearProblem(A2, b2; u0 = x2)
prob3 = LinearProblem(A3, b3; u0 = x3)
prob4 = LinearProblem(A4, b4; u0 = x4)

cache_kwargs = (; verbose = true, abstol = 1e-8, reltol = 1e-8, maxiter = 30)

function test_interface(alg, prob1, prob2)
    A1, b1 = prob1.A, prob1.b
    A2, b2 = prob2.A, prob2.b

    sol = solve(prob1, alg; cache_kwargs...)
    @test A1 * sol.u ≈ b1

    sol = solve(prob2, alg; cache_kwargs...)
    @test A2 * sol.u ≈ b2

    # Test cache reuse: base mechanism
    cache = SciMLBase.init(prob1, alg; cache_kwargs...) # initialize cache
    sol = solve!(cache)
    @test A1 * sol.u ≈ b1

    # Test cache reuse: only A changes
    cache.A = deepcopy(A2)
    sol = solve!(cache; cache_kwargs...)
    @test A2 * sol.u ≈ b1

    # Test cache reuse: both A and b change
    cache.A = deepcopy(A2)
    cache.b = b2
    sol = solve!(cache; cache_kwargs...)
    @test A2 * sol.u ≈ b2

    return
end

@testset "LinearSolve" begin
    @testset "Default Linear Solver" begin
        test_interface(nothing, prob1, prob2)
        test_interface(nothing, prob3, prob4)

        A1 = prob1.A * prob1.A'
        b1 = prob1.b
        x1 = prob1.u0
        y = solve(prob1)
        @test A1 * y ≈ b1

        _prob = LinearProblem(Diagonal(A1), b1; u0 = x1)
        y = solve(_prob)
        @test A1 * y ≈ b1

        _prob = LinearProblem(SymTridiagonal(A1), b1; u0 = x1)
        y = solve(_prob)
        @test A1 * y ≈ b1

        _prob = LinearProblem(Tridiagonal(A1), b1; u0 = x1)
        y = solve(_prob)
        @test A1 * y ≈ b1

        _prob = LinearProblem(Symmetric(A1), b1; u0 = x1)
        y = solve(_prob)
        @test A1 * y ≈ b1

        _prob = LinearProblem(Hermitian(A1), b1; u0 = x1)
        y = solve(_prob)
        @test A1 * y ≈ b1

        if VERSION > v"1.9-"
            _prob = LinearProblem(sparse(A1), b1; u0 = x1)
            y = solve(_prob)
            @test A1 * y ≈ b1
        end
    end

    @testset "UMFPACK Factorization" begin
        A1 = sparse(A / 1)
        b1 = rand(n)
        x1 = zero(b)
        A2 = sparse(A / 2)
        b2 = rand(n)
        x2 = zero(b)

        prob1 = LinearProblem(A1, b1; u0 = x1)
        prob2 = LinearProblem(A2, b2; u0 = x2)
        test_interface(UMFPACKFactorization(), prob1, prob2)
        test_interface(UMFPACKFactorization(reuse_symbolic = false), prob1, prob2)

        # Test that refactoring is checked and handled.
        cache = SciMLBase.init(prob1, UMFPACKFactorization(); cache_kwargs...) # initialize cache
        y = solve!(cache)
        cache.A = A2
        @test A2 * solve!(cache) ≈ b1
        X = sprand(n, n, 0.8)
        cache.A = X
        @test X * solve!(cache) ≈ b1
    end

    @testset "KLU Factorization" begin
        A1 = sparse(A / 1)
        b1 = rand(n)
        x1 = zero(b)
        A2 = sparse(A / 2)
        b2 = rand(n)
        x2 = zero(b)

        prob1 = LinearProblem(A1, b1; u0 = x1)
        prob2 = LinearProblem(A2, b2; u0 = x2)
        test_interface(KLUFactorization(), prob1, prob2)
        test_interface(KLUFactorization(reuse_symbolic = false), prob1, prob2)

        # Test that refactoring wrong is checked and handled.
        cache = SciMLBase.init(prob1, KLUFactorization(); cache_kwargs...) # initialize cache
        y = solve!(cache)
        cache.A = A2
        @test A2 * solve!(cache) ≈ b1
        X = sprand(n, n, 0.8)
        cache.A = X
        @test X * solve!(cache) ≈ b1
    end

    @testset "Sparspak Factorization (Float64)" begin
        A1 = sparse(A / 1)
        b1 = rand(n)
        x1 = zero(b)
        A2 = sparse(A / 2)
        b2 = rand(n)
        x2 = zero(b)

        prob1 = LinearProblem(A1, b1; u0 = x1)
        prob2 = LinearProblem(A2, b2; u0 = x2)
        test_interface(SparspakFactorization(), prob1, prob2)
    end

    @testset "Sparspak Factorization (Float64x1)" begin
        A1 = sparse(A / 1) .|> Float64x1
        b1 = rand(n) .|> Float64x1
        x1 = zero(b) .|> Float64x1
        A2 = sparse(A / 2) .|> Float64x1
        b2 = rand(n) .|> Float64x1
        x2 = zero(b) .|> Float64x1

        prob1 = LinearProblem(A1, b1; u0 = x1)
        prob2 = LinearProblem(A2, b2; u0 = x2)
        test_interface(SparspakFactorization(), prob1, prob2)
    end

    @testset "Sparspak Factorization (Float64x2)" begin
        A1 = sparse(A / 1) .|> Float64x2
        b1 = rand(n) .|> Float64x2
        x1 = zero(b) .|> Float64x2
        A2 = sparse(A / 2) .|> Float64x2
        b2 = rand(n) .|> Float64x2
        x2 = zero(b) .|> Float64x2

        prob1 = LinearProblem(A1, b1; u0 = x1)
        prob2 = LinearProblem(A2, b2; u0 = x2)
        test_interface(SparspakFactorization(), prob1, prob2)
    end

    @testset "Sparspak Factorization (Dual64)" begin
        A1 = sparse(A / 1) .|> Dual64
        b1 = rand(n) .|> Dual64
        x1 = zero(b) .|> Dual64
        A2 = sparse(A / 2) .|> Dual64
        b2 = rand(n) .|> Dual64
        x2 = zero(b) .|> Dual64

        prob1 = LinearProblem(A1, b1; u0 = x1)
        prob2 = LinearProblem(A2, b2; u0 = x2)
        test_interface(SparspakFactorization(), prob1, prob2)
    end

    if VERSION >= v"1.9"
        @testset "FastLAPACK Factorizations" begin
            A1 = A / 1
            b1 = rand(n)
            x1 = zero(b)
            A2 = A / 2
            b2 = rand(n)
            x2 = zero(b)

            prob1 = LinearProblem(A1, b1; u0 = x1)
            prob2 = LinearProblem(A2, b2; u0 = x2)
            test_interface(LinearSolve.FastLUFactorization(), prob1, prob2)
            test_interface(LinearSolve.FastQRFactorization(), prob1, prob2)

            # TODO: Resizing tests. Upstream doesn't currently support it.
            # Need to be absolutely certain we never segfault with incorrect
            # ws sizes.
        end
    end

    test_algs = [
        LUFactorization(),
        QRFactorization(),
        SVDFactorization(),
        RFLUFactorization(),
        LinearSolve.defaultalg(prob1.A, prob1.b)
    ]

    if VERSION >= v"1.9" && LinearSolve.usemkl
        push!(test_algs, MKLLUFactorization())
    end

    @testset "Concrete Factorizations" begin
        for alg in test_algs
            @testset "$alg" begin
                test_interface(alg, prob1, prob2)
                VERSION >= v"1.9" && test_interface(alg, prob3, prob4)
            end
        end
        if LinearSolve.appleaccelerate_isavailable()
            test_interface(AppleAccelerateLUFactorization(), prob1, prob2)
            test_interface(AppleAccelerateLUFactorization(), prob3, prob4)
        end
    end

    @testset "Generic Factorizations" begin
        for fact_alg in (lu, lu!,
            qr, qr!,
            cholesky,
            # cholesky!,
            # ldlt, ldlt!,
            bunchkaufman, bunchkaufman!,
            lq, lq!,
            svd, svd!,
            LinearAlgebra.factorize)
            @testset "fact_alg = $fact_alg" begin
                alg = GenericFactorization(fact_alg = fact_alg)
                test_interface(alg, prob1, prob2)
                test_interface(alg, prob3, prob4)
            end
        end
    end

    @testset "Simple GMRES: restart = $restart" for restart in (true, false)
        test_interface(SimpleGMRES(; restart), prob1, prob2)
    end

    @testset "KrylovJL" begin
        kwargs = (; gmres_restart = 5)
        precs = (A, p = nothing) -> (BlockJacobiPreconditioner(A, 2), I)
        algorithms = (
            ("Default", KrylovJL(kwargs...)),
            ("CG", KrylovJL_CG(kwargs...)),
            ("GMRES", KrylovJL_GMRES(kwargs...)),
            ("GMRES_prec", KrylovJL_GMRES(; precs, ldiv = false, kwargs...)),
            # ("BICGSTAB",KrylovJL_BICGSTAB(kwargs...)),
            ("MINRES", KrylovJL_MINRES(kwargs...)),
            ("MINARES", KrylovJL_MINARES(kwargs...))
        )
        for (name, algorithm) in algorithms
            @testset "$name" begin
                test_interface(algorithm, prob1, prob2)
                test_interface(algorithm, prob3, prob4)
            end
        end
    end

    @testset "Reuse precs" begin
        num_precs_calls = 0

        function countingprecs(A, p = nothing)
            num_precs_calls += 1
            (BlockJacobiPreconditioner(A, 2), I)
        end

        n = 10
        A = spdiagm(-1 => -ones(n - 1), 0 => fill(10.0, n), 1 => -ones(n - 1))
        b = rand(n)
        p = LinearProblem(A, b)
        x0 = solve(p, KrylovJL_CG(precs = countingprecs, ldiv = false))
        cache = x0.cache
        x0 = copy(x0)
        for i in 4:(n - 3)
            A[i, i + 3] -= 1.0e-4
            A[i - 3, i] -= 1.0e-4
        end
        LinearSolve.reinit!(cache; A, reuse_precs = true)
        x1 = copy(solve!(cache))
        @test all(x0 .< x1) && num_precs_calls == 1
    end

    if VERSION >= v"1.9-"
        @testset "IterativeSolversJL" begin
            kwargs = (; gmres_restart = 5)
            for alg in (("Default", IterativeSolversJL(kwargs...)),
                ("CG", IterativeSolversJL_CG(kwargs...)),
                ("GMRES", IterativeSolversJL_GMRES(kwargs...)),
                ("IDRS", IterativeSolversJL_IDRS(kwargs...))                #           ("BICGSTAB",IterativeSolversJL_BICGSTAB(kwargs...)),                #            ("MINRES",IterativeSolversJL_MINRES(kwargs...)),
            )
                @testset "$(alg[1])" begin
                    test_interface(alg[2], prob1, prob2)
                    test_interface(alg[2], prob3, prob4)
                end
            end
        end
    end

    if VERSION > v"1.9-"
        @testset "KrylovKit" begin
            kwargs = (; gmres_restart = 5)
            for alg in (("Default", KrylovKitJL(kwargs...)),
                ("CG", KrylovKitJL_CG(kwargs...)),
                ("GMRES", KrylovKitJL_GMRES(kwargs...)))
                @testset "$(alg[1])" begin
                    test_interface(alg[2], prob1, prob2)
                    test_interface(alg[2], prob3, prob4)
                end
                @test alg[2] isa KrylovKitJL
            end
        end
    end

    if VERSION > v"1.9-"
        @testset "CHOLMOD" begin
            # Create a posdef symmetric matrix
            A = sprand(100, 100, 0.01)
            A = A + A' + 100 * I

            # rhs
            b = rand(100)

            # Set the problem
            prob = LinearProblem(A, b)
            sol = solve(prob)

            # Enforce symmetry to use Cholesky, since A is symmetric and posdef
            prob2 = LinearProblem(Symmetric(A), b)
            sol2 = solve(prob2)
            @test abs(norm(A * sol2.u .- b) - norm(A * sol.u .- b)) < 1e-12
        end
    end

    @testset "Preconditioners" begin
        @testset "Vector Diagonal Preconditioner" begin
            x = rand(n, n)
            y = rand(n, n)

            s = rand(n)
            Pl = Diagonal(s) |> MatrixOperator
            Pr = Diagonal(s) |> MatrixOperator |> inv
            Pr = cache_operator(Pr, x)

            mul!(y, Pl, x)
            @test y ≈ s .* x
            mul!(y, Pr, x)
            @test y ≈ s .\ x

            y .= x
            ldiv!(Pl, x)
            @test x ≈ s .\ y
            y .= x
            ldiv!(Pr, x)
            @test x ≈ s .* y

            ldiv!(y, Pl, x)
            @test y ≈ s .\ x
            ldiv!(y, Pr, x)
            @test y ≈ s .* x
        end

        @testset "ComposePreconditioenr" begin
            s1 = rand(n)
            s2 = rand(n)

            x = rand(n, n)
            y = rand(n, n)

            P1 = Diagonal(s1)
            P2 = Diagonal(s2)

            P = LinearSolve.ComposePreconditioner(P1, P2)

            # ComposePreconditioner
            ldiv!(y, P, x)
            @test y ≈ ldiv!(P2, ldiv!(P1, x))
            y .= x
            ldiv!(P, x)
            @test x ≈ ldiv!(P2, ldiv!(P1, y))
        end
    end

    @testset "Sparse Precaching" begin
        n = 4
        Random.seed!(10)
        A = sprand(n, n, 0.8)
        A2 = 2.0 .* A
        b1 = rand(n)
        b2 = rand(n)

        prob = LinearProblem(copy(A), copy(b1))
        linsolve = init(prob, UMFPACKFactorization())
        sol11 = solve!(linsolve)
        linsolve.b = copy(b2)
        sol12 = solve!(linsolve)
        linsolve.A = copy(A2)
        sol13 = solve!(linsolve)

        prob = LinearProblem(copy(A), copy(b1))
        linsolve = init(prob, KLUFactorization())
        sol21 = solve!(linsolve)
        linsolve.b = copy(b2)
        sol22 = solve!(linsolve)
        linsolve.A = copy(A2)
        sol23 = solve!(linsolve)

        @test sol11.u ≈ sol21.u
        @test sol12.u ≈ sol22.u
        @test sol13.u ≈ sol23.u
    end

    @testset "Solve Function" begin
        A1 = rand(n) |> Diagonal
        b1 = rand(n)
        x1 = zero(b1)
        A2 = rand(n) |> Diagonal
        b2 = rand(n)
        x2 = zero(b1)

        @testset "LinearSolveFunction" begin
            function sol_func(A, b, u, p, newA, Pl, Pr, solverdata; verbose = true,
                    kwargs...)
                if verbose == true
                    println("out-of-place solve")
                end
                u .= A \ b
            end

            function sol_func!(A, b, u, p, newA, Pl, Pr, solverdata; verbose = true,
                    kwargs...)
                if verbose == true
                    println("in-place solve")
                end
                ldiv!(u, A, b)
            end

            prob1 = LinearProblem(A1, b1; u0 = x1)
            prob2 = LinearProblem(A1, b1; u0 = x1)

            for alg in (LinearSolveFunction(sol_func),
                LinearSolveFunction(sol_func!))
                test_interface(alg, prob1, prob2)
            end
        end

        @testset "DirectLdiv!" begin
            function get_operator(A, u; add_inverse = true)
                function f(u, p, t)
                    println("using FunctionOperator OOP mul")
                    A * u
                end
                function f(du, u, p, t)
                    println("using FunctionOperator IIP mul")
                    mul!(du, A, u)
                end

                function fi(du, u, p, t)
                    println("using FunctionOperator OOP div")
                    A \ u
                end
                function fi(du, u, p, t)
                    println("using FunctionOperator IIP div")
                    ldiv!(du, A, u)
                end

                if add_inverse
                    FunctionOperator(f, u; op_inverse = fi)
                else
                    FunctionOperator(f, u)
                end
            end

            op1 = get_operator(A1, x1 * 0)
            op2 = get_operator(A2, x2 * 0)
            op3 = get_operator(A1, x1 * 0; add_inverse = false)
            op4 = get_operator(A2, x2 * 0; add_inverse = false)

            prob1 = LinearProblem(op1, b1; u0 = x1)
            prob2 = LinearProblem(op2, b2; u0 = x2)
            prob3 = LinearProblem(op1, b1; u0 = x1)
            prob4 = LinearProblem(op2, b2; u0 = x2)

            @test LinearSolve.defaultalg(op1, x1).alg ===
                  LinearSolve.DefaultAlgorithmChoice.DirectLdiv!
            @test LinearSolve.defaultalg(op2, x2).alg ===
                  LinearSolve.DefaultAlgorithmChoice.DirectLdiv!
            @test LinearSolve.defaultalg(op3, x1).alg ===
                  LinearSolve.DefaultAlgorithmChoice.KrylovJL_GMRES
            @test LinearSolve.defaultalg(op4, x2).alg ===
                  LinearSolve.DefaultAlgorithmChoice.KrylovJL_GMRES
            test_interface(DirectLdiv!(), prob1, prob2)
            test_interface(nothing, prob1, prob2)
            test_interface(KrylovJL_GMRES(), prob3, prob4)
            test_interface(nothing, prob3, prob4)
        end
    end
end # testset

# https://github.com/SciML/LinearSolve.jl/issues/347
A = rand(4, 4);
b = rand(4);
u0 = zeros(4);
lp = LinearProblem(A, b; u0 = view(u0, :));
truesol = solve(lp, LUFactorization())
krylovsol = solve(lp, KrylovJL_GMRES())
@test truesol ≈ krylovsol

# Block Diagonals
using BlockDiagonals

@testset "Block Diagonal Specialization" begin
    A = BlockDiagonal([rand(2, 2) for _ in 1:3])
    b = rand(size(A, 1))

    if VERSION > v"1.9-"
        x1 = zero(b)
        x2 = zero(b)
        prob1 = LinearProblem(A, b, x1)
        prob2 = LinearProblem(A, b, x2)
        test_interface(SimpleGMRES(), prob1, prob2)
    end

    x1 = zero(b)
    x2 = zero(b)
    prob1 = LinearProblem(Array(A), b, x1)
    prob2 = LinearProblem(Array(A), b, x2)

    test_interface(SimpleGMRES(; blocksize = 2), prob1, prob2)

    @test solve(prob1, SimpleGMRES(; blocksize = 2)).u ≈ solve(prob2, SimpleGMRES()).u
end

@testset "AbstractSparseMatrixCSC" begin
    struct MySparseMatrixCSC{Tv, Ti} <: SparseArrays.AbstractSparseMatrixCSC{Tv, Ti}
        csc::SparseMatrixCSC{Tv, Ti}
    end

    Base.size(m::MySparseMatrixCSC) = size(m.csc)
    SparseArrays.getcolptr(m::MySparseMatrixCSC) = SparseArrays.getcolptr(m.csc)
    SparseArrays.rowvals(m::MySparseMatrixCSC) = SparseArrays.rowvals(m.csc)
    SparseArrays.nonzeros(m::MySparseMatrixCSC) = SparseArrays.nonzeros(m.csc)

    N = 10_000
    A = spdiagm(1 => -ones(N - 1), 0 => fill(10.0, N), -1 => -ones(N - 1))
    u0 = ones(size(A, 2))

    b = A * u0
    B = MySparseMatrixCSC(A)
    pr = LinearProblem(B, b)

    # test default algorithn
    @time "solve MySparseMatrixCSC" u=solve(pr)
    @test norm(u - u0, Inf) < 1.0e-13

    # test Krylov algorithm with reinit!
    pr = LinearProblem(B, b)
    solver = KrylovJL_CG()
    cache = init(pr, solver, maxiters = 1000, reltol = 1.0e-10)
    u = solve!(cache)
    A1 = spdiagm(1 => -ones(N - 1), 0 => fill(100.0, N), -1 => -ones(N - 1))
    b1 = A1 * u0
    B1 = MySparseMatrixCSC(A1)
    @test norm(u - u0, Inf) < 1.0e-8
    reinit!(cache; A = B1, b = b1)
    u = solve!(cache)
    @test norm(u - u0, Inf) < 1.0e-8

    # test factorization with reinit!
    pr = LinearProblem(B, b)
    solver = SparspakFactorization()
    cache = init(pr, solver)
    u = solve!(cache)
    @test norm(u - u0, Inf) < 1.0e-8
    reinit!(cache; A = B1, b = b1)
    u = solve!(cache)
    @test norm(u - u0, Inf) < 1.0e-8
end
