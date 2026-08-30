using LinearSolve, LinearAlgebra, SparseArrays, MultiFloats, ForwardDiff
using SciMLOperators: SciMLOperators, MatrixOperator, FunctionOperator, WOperator,
    cache_operator
using RecursiveFactorization, Sparspak, FastLapackInterface
using IterativeSolvers, KrylovKit, MKL_jll
using Test
import CliqueTrees, Random

# Try to load BLIS extension
try
    using blis_jll, LAPACK_jll
catch LoadError
    # BLIS dependencies not available, tests will be skipped
end

try
    using AlgebraicMultigrid
catch
    # AlgebraicMultigrid not available, AMG tests will be skipped
end

try
    using LinearSolvePyAMG
catch
    # LinearSolvePyAMG not available, PyAMG tests will be skipped
end

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

A5_ = A - 0.01Tridiagonal(ones(n, n)) + sparse([1], [8], 0.5, n, n)
A5 = sparse(transpose(A5_) * A5_)
x5 = zeros(n)
u5 = ones(n)
b5 = A5 * u5

prob1 = LinearProblem(A1, b1; u0 = x1)
prob2 = LinearProblem(A2, b2; u0 = x2)
prob3 = LinearProblem(A3, b3; u0 = x3)
prob4 = LinearProblem(A4, b4; u0 = x4)
prob5 = LinearProblem(A5, b5)

cache_kwargs = (; abstol = 1.0e-8, reltol = 1.0e-8, maxiter = 30)

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

function test_tolerance_update(alg, prob, u)
    cache = init(prob, alg)
    LinearSolve.update_tolerances!(cache; reltol = 1.0e-2, abstol = 1.0e-8)
    u1 = copy(solve!(cache).u)

    LinearSolve.update_tolerances!(cache; reltol = 1.0e-8, abstol = 1.0e-8)
    u2 = solve!(cache).u

    @test norm(u2 - u) < norm(u1 - u)

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

    @testset "KLU factor parts" begin
        KLU = LinearSolve.KLU
        # Block-triangular so the off-diagonal `F` part is non-empty
        Abt = sparse([rand(3, 3) + 3I rand(3, 3); zeros(3, 3) rand(3, 3) + 3I])
        for At in (sparse(A / 1), Abt)
            K = KLU.klu(At)
            @test (K.Rs .\ At)[K.p, K.q] ≈ K.L * K.U + K.F
            @test sort(K.p) == 1:size(At, 1)
            @test sort(K.q) == 1:size(At, 1)
            @test length(K.R) == K.nblocks + 1
            @test K.R[1] == 1
            @test K.R[end] == size(At, 1) + 1
        end
        # unfactored handles still report what is missing
        @test_throws ArgumentError KLU.KLUFactorization(sparse(A / 1)).lnz
    end

    @testset "PureKLU Factorization" begin
        A1 = sparse(A / 1)
        b1 = rand(n)
        x1 = zero(b)
        A2 = sparse(A / 2)
        b2 = rand(n)
        x2 = zero(b)

        prob1 = LinearProblem(A1, b1; u0 = x1)
        prob2 = LinearProblem(A2, b2; u0 = x2)
        test_interface(PureKLUFactorization(), prob1, prob2)
        test_interface(PureKLUFactorization(reuse_symbolic = false), prob1, prob2)
        test_interface(PureKLUFactorization(use_fma = false), prob1, prob2)
        test_interface(PureKLUFactorization(fully_preallocated = true), prob1, prob2)

        # Test that refactoring with a changed pattern is checked and handled.
        cache = SciMLBase.init(prob1, PureKLUFactorization(); cache_kwargs...) # initialize cache
        y = solve!(cache)
        cache.A = A2
        @test A2 * solve!(cache) ≈ b1
        X = sprand(n, n, 0.8)
        cache.A = X
        @test X * solve!(cache) ≈ b1

        # Partial pivoting tolerance option
        @test PureKLUFactorization().tol == 0.001 # default value
        function rowperm(tol)
            A = sparse([1.0e-3 1.0; 1.0 1.0])
            b = [1.0, 2.0]
            alg = PureKLUFactorization(; tol)
            cache = SciMLBase.init(LinearProblem(A, b), alg)
            solve!(cache)
            return cache.cacheval.p # row permutation vector
        end
        @test rowperm(0.0) == [1, 2] # below threshold: pivot on diagonal entry
        @test rowperm(1.0e-4) == [1, 2] # below threshold: pivot on diagonal entry
        @test rowperm(1.0e-3) == [1, 2] # at threshold: pivot on diagonal entry
        @test rowperm(1.0e-2) == [2, 1] # above threshold: pivot on largest entry
        @test rowperm(1.0) == [2, 1] # above threshold: pivot on largest entry
    end

    @testset "RFLUFactorization multi-RHS" begin
        # The extension routes matrix right-hand sides through TriangularSolve;
        # results must match the stdlib path exactly in behaviour.
        Random.seed!(7)
        for n in (8, 40), nrhs in (1, 3)
            Ar = rand(n, n) + n * I
            Br = rand(n, nrhs)
            sol = solve(LinearProblem(copy(Ar), copy(Br)), RFLUFactorization())
            @test sol.u ≈ Ar \ Br
            # cache reuse path
            cache = SciMLBase.init(LinearProblem(copy(Ar), copy(Br)), RFLUFactorization())
            @test solve!(cache).u ≈ Ar \ Br
            A2 = Ar + I
            cache.A = copy(A2)
            @test solve!(cache).u ≈ A2 \ Br
        end
    end

    @testset "RFLUFactorization backsolve routing (TriangularSolve, both pivots)" begin
        # Correctness for vector and matrix right-hand sides under both pivot
        # modes.  pivot = Val(false) crashed before the routing fix:
        # RecursiveFactorization returns the caller-supplied ipiv unwritten, and
        # the old backsolve handed it to LAPACK.getrs! / _ipiv_rows! (segfault /
        # BoundsError on garbage pivots).  n = 300 covers the region where
        # TriangularSolve < 0.2.5 used to defer vectors to BLAS: with 0.2.5 the
        # vector legs stay on its native kernels at every size.
        Random.seed!(11)
        for pivot in (Val(true), Val(false)), n2 in (8, 40, 300)
            Ar = rand(n2, n2) + 2n2 * I
            br = rand(n2)
            Br = rand(n2, 3)
            alg2 = RFLUFactorization(pivot, Val(true))
            @test solve(LinearProblem(copy(Ar), copy(br)), alg2).u ≈ Ar \ br
            @test solve(LinearProblem(copy(Ar), copy(Br)), alg2).u ≈ Ar \ Br
            # a single-column matrix takes the same TriangularSolve route
            B1 = Br[:, 1:1]
            @test solve(LinearProblem(copy(Ar), copy(B1)), alg2).u ≈ Ar \ B1
        end

        # ComplexF64 has no TriangularSolve kernels; it keeps the stdlib path
        # and must stay correct under both pivot modes.
        for pivot in (Val(true), Val(false))
            Ac = rand(ComplexF64, 20, 20) + 40I
            bc = rand(ComplexF64, 20)
            Bc = rand(ComplexF64, 20, 2)
            algc = RFLUFactorization(pivot, Val(true))
            @test solve(LinearProblem(copy(Ac), copy(bc)), algc).u ≈ Ac \ bc
            @test solve(LinearProblem(copy(Ac), copy(Bc)), algc).u ≈ Ac \ Bc
        end

        # Dispatch audit: never-BLAS enforcement.  The extension's TS-routing
        # methods must be selected for Float64/Float32 strided inputs, and
        # TriangularSolve must resolve both backsolve legs to its native
        # kernels — never to its LinearAlgebra (BLAS) catch-all.  These
        # assertions fail if a TriangularSolve or extension restructuring
        # silently reintroduces a BLAS fallback.
        ext = Base.get_extension(LinearSolve, :LinearSolveRecursiveFactorizationExt)
        @test ext !== nothing
        if ext !== nothing
            for T in (Float64, Float32)
                MT = Matrix{T}
                # matrix legs and the vector legs the vector path hands
                # TriangularSolve directly (native vector kernels need
                # TriangularSolve >= 0.2.5)
                for BT in (MT, Vector{T})
                    @test ext._ts_native_backsolve(UnitLowerTriangular{T, MT}, BT)
                    @test ext._ts_native_backsolve(UpperTriangular{T, MT}, BT)
                end
                LUT = LU{T, MT, Vector{LinearAlgebra.BlasInt}}
                m_vec = which(
                    ext._rf_ldiv!, Tuple{Vector{T}, LUT, Vector{T}, Val{true}, Val{true}}
                )
                m_mat = which(ext._rf_ldiv!, Tuple{MT, LUT, MT, Val{true}, Val{true}})
                CMT = Matrix{ComplexF64}
                CLUT = LU{ComplexF64, CMT, Vector{LinearAlgebra.BlasInt}}
                m_vec_fb = which(
                    ext._rf_ldiv!,
                    Tuple{Vector{ComplexF64}, CLUT, Vector{ComplexF64}, Val{true}, Val{true}}
                )
                m_mat_fb = which(ext._rf_ldiv!, Tuple{CMT, CLUT, CMT, Val{true}, Val{true}})
                # the strided real methods are the TS-routing ones, not the
                # stdlib fallbacks the complex types resolve to
                @test m_vec !== m_vec_fb
                @test m_mat !== m_mat_fb
            end
        end

        # Factorization cells (dispatch-audited in RecursiveFactorization's own
        # test suite): RFLU's factorization is RecursiveFactorization.lu!,
        # whose Float32/Float64 path runs on BLAS-free recursive kernels with
        # TriangularSolve panel solves.  Complex eltypes factor correctly (see
        # above) but their panel solves fall back to LAPACK, so that cell must
        # stay reachable only by explicitly requesting RFLUFactorization: the
        # default algorithm must not route complex matrices to it.
        @test LinearSolve.userecursivefactorization(rand(2, 2))
        algc64 = LinearSolve.defaultalg(
            rand(ComplexF64, 100, 100), rand(ComplexF64, 100),
            LinearSolve.OperatorAssumptions(true)
        )
        @test algc64.alg !== LinearSolve.DefaultAlgorithmChoice.RFLUFactorization
    end

    @testset "SupernodalLU Factorization" begin
        A1 = sparse(A / 1)
        b1 = rand(n)
        x1 = zero(b)
        A2 = sparse(A / 2)
        b2 = rand(n)
        x2 = zero(b)

        prob1 = LinearProblem(A1, b1; u0 = x1)
        prob2 = LinearProblem(A2, b2; u0 = x2)
        test_interface(SupernodalLUFactorization(), prob1, prob2)
        test_interface(SupernodalLUFactorization(reuse_symbolic = false), prob1, prob2)
        test_interface(SupernodalLUFactorization(ordering = :nd), prob1, prob2)
        test_interface(SupernodalLUFactorization(matching = false), prob1, prob2)

        # Test that refactoring with a changed pattern is checked and handled.
        cache = SciMLBase.init(prob1, SupernodalLUFactorization(); cache_kwargs...)
        y = solve!(cache)
        cache.A = A2
        @test A2 * solve!(cache) ≈ b1
        X = sprand(n, n, 0.8)
        cache.A = X
        @test X * solve!(cache) ≈ b1

        # numerically singular systems surface as Infeasible, not NaN Success
        Z = spzeros(4, 4)
        Z[1, 1] = 1.0
        zsol = solve(LinearProblem(Z, ones(4)), SupernodalLUFactorization())
        @test zsol.retcode == ReturnCode.Infeasible
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

    @testset "CliqueTrees Factorization (Float64)" begin
        A1 = sparse(A / 1)
        b1 = rand(n)
        x1 = zero(b)
        A2 = sparse(A / 2)
        b2 = rand(n)
        x2 = zero(b)

        prob1 = LinearProblem(A1, b1; u0 = x1)
        prob2 = LinearProblem(A2, b2; u0 = x2)
        test_interface(CliqueTreesFactorization(), prob1, prob2)
    end

    @testset "CliqueTrees Factorization (Float64x1)" begin
        A1 = sparse(A / 1) .|> Float64x1
        b1 = rand(n) .|> Float64x1
        x1 = zero(b) .|> Float64x1
        A2 = sparse(A / 2) .|> Float64x1
        b2 = rand(n) .|> Float64x1
        x2 = zero(b) .|> Float64x1

        prob1 = LinearProblem(A1, b1; u0 = x1)
        prob2 = LinearProblem(A2, b2; u0 = x2)
        test_interface(CliqueTreesFactorization(), prob1, prob2)
    end

    @testset "CliqueTrees Factorization (Float64x2)" begin
        A1 = sparse(A / 1) .|> Float64x2
        b1 = rand(n) .|> Float64x2
        x1 = zero(b) .|> Float64x2
        A2 = sparse(A / 2) .|> Float64x2
        b2 = rand(n) .|> Float64x2
        x2 = zero(b) .|> Float64x2

        prob1 = LinearProblem(A1, b1; u0 = x1)
        prob2 = LinearProblem(A2, b2; u0 = x2)
        test_interface(CliqueTreesFactorization(), prob1, prob2)
    end

    @testset "CliqueTrees Factorization (Dual64)" begin
        A1 = sparse(A / 1) .|> Dual64
        b1 = rand(n) .|> Dual64
        x1 = zero(b) .|> Dual64
        A2 = sparse(A / 2) .|> Dual64
        b2 = rand(n) .|> Dual64
        x2 = zero(b) .|> Dual64

        prob1 = LinearProblem(A1, b1; u0 = x1)
        prob2 = LinearProblem(A2, b2; u0 = x2)
        test_interface(CliqueTreesFactorization(), prob1, prob2)
    end

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

    @testset "SymTridiagonal with LDLtFactorization" begin
        # Test that LDLtFactorization works correctly with SymTridiagonal
        # and that the default algorithm correctly selects it
        k = 100
        ρ = 0.95
        A_tri = SymTridiagonal(ones(k) .+ ρ^2, -ρ * ones(k - 1))
        b = rand(k)

        # Test with explicit LDLtFactorization
        prob_tri = LinearProblem(A_tri, b)
        sol = solve(prob_tri, LDLtFactorization())
        @test A_tri * sol.u ≈ b

        # Test that default algorithm uses LDLtFactorization for SymTridiagonal
        default_alg = LinearSolve.defaultalg(A_tri, b, OperatorAssumptions(true))
        @test default_alg isa LinearSolve.DefaultLinearSolver
        @test default_alg.alg == LinearSolve.DefaultAlgorithmChoice.LDLtFactorization

        # Test that the factorization is cached and reused
        cache = init(prob_tri, LDLtFactorization())
        sol1 = solve!(cache)
        @test A_tri * sol1.u ≈ b
        @test !cache.isfresh  # Cache should not be fresh after first solve

        # Solve again with same matrix to ensure cache is reused
        cache.b = rand(k)  # Change RHS
        sol2 = solve!(cache)
        @test A_tri * sol2.u ≈ cache.b
        @test !cache.isfresh  # Cache should still not be fresh
    end

    @testset "Tridiagonal cache not mutated (issue #825)" begin
        # Test that solving with Tridiagonal does not mutate cache.A
        # See https://github.com/SciML/LinearSolve.jl/issues/825
        k = 6
        lower = ones(k - 1)
        diag = -2 * ones(k)
        upper = ones(k - 1)
        A_tri = Tridiagonal(lower, diag, upper)
        b = rand(k)

        # Store original matrix values for comparison
        A_orig = Tridiagonal(copy(lower), copy(diag), copy(upper))

        # Test that default algorithm uses DirectLdiv! for Tridiagonal on Julia 1.11+
        default_alg = LinearSolve.defaultalg(A_tri, b, OperatorAssumptions(true))
        @static if VERSION >= v"1.11"
            @test default_alg isa DirectLdiv!
        else
            @test default_alg isa LinearSolve.DefaultLinearSolver
            @test default_alg.alg == LinearSolve.DefaultAlgorithmChoice.LUFactorization
        end

        # Test with default algorithm
        prob_tri = LinearProblem(A_tri, b)
        cache = init(prob_tri)

        # Verify solution is correct
        sol1 = solve!(cache)
        @test A_orig * sol1.u ≈ b

        # Verify cache.A is not mutated
        @test cache.A ≈ A_orig

        # Verify multiple solves give correct answers
        b2 = rand(k)
        cache.b = b2
        sol2 = solve!(cache)
        @test A_orig * sol2.u ≈ b2

        # Cache.A should still be unchanged
        @test cache.A ≈ A_orig

        # Verify solve! allocates minimally after first solve (warm-up)
        # The small allocation (48 bytes) is from the return type construction,
        # same as other factorization methods like LUFactorization
        @static if VERSION >= v"1.11"
            # Warm up
            for _ in 1:3
                solve!(cache)
            end
            # Test minimal allocations (same as LUFactorization)
            allocs = @allocated solve!(cache)
            @test allocs <= 64  # Allow small overhead from return type
        end
    end

    test_algs = [
        LUFactorization(),
        QRFactorization(),
        SVDFactorization(),
        RFLUFactorization(),
        LinearSolve.defaultalg(prob1.A, prob1.b),
    ]

    if LinearSolve.usemkl
        push!(test_algs, MKLLUFactorization())
    end

    # Test OpenBLAS if available
    if LinearSolve.useopenblas
        push!(test_algs, OpenBLASLUFactorization())
    end

    # Test BLIS if extension is available
    if Base.get_extension(LinearSolve, :LinearSolveBLISExt) !== nothing
        push!(test_algs, LinearSolve.BLISLUFactorization())
    end

    @testset "Concrete Factorizations" begin
        for alg in test_algs
            @testset "$alg" begin
                test_interface(alg, prob1, prob2)
                test_interface(alg, prob3, prob4)
            end
        end
        if LinearSolve.appleaccelerate_isavailable()
            test_interface(AppleAccelerateLUFactorization(), prob1, prob2)
            test_interface(AppleAccelerateLUFactorization(), prob3, prob4)
        end
    end

    @testset "Generic Factorizations" begin
        for fact_alg in (
                lu, lu!,
                qr, qr!,
                cholesky,
                # cholesky!,
                # ldlt, ldlt!,
                bunchkaufman, bunchkaufman!,
                lq, lq!,
                svd, svd!,
                LinearAlgebra.factorize,
            )
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

    # Every way of re-entering `solve!` has to rebuild the cacheval's initial
    # residual, not just the `cache.b = ...` setproperty hook.
    @testset "Simple GMRES resolve: restart = $restart, blocksize = $blocksize" for
        restart in (true, false), blocksize in (0, 2)

        nr = 6
        Ar = [
            float(i == j ? 10 + i : 0.3 * (i + j)) *
                (blocksize == 0 || (i - 1) ÷ blocksize == (j - 1) ÷ blocksize)
                for i in 1:nr, j in 1:nr
        ]
        br = float.(1:nr)
        br2 = [0.3i + 1 for i in 1:nr]
        alg = SimpleGMRES(; restart, blocksize)

        @testset "$desc" for (desc, update_b!, expected) in (
                ("b mutated in place", (c, b) -> (c.b .= b), br2),
                ("b replaced", (c, b) -> (c.b = copy(b)), br2),
                ("b unchanged", (c, b) -> nothing, br),
            )
            cache = init(
                LinearProblem(copy(Ar), copy(br)), alg; abstol = 1.0e-12, reltol = 1.0e-12
            )
            @test solve!(cache).u ≈ Ar \ br rtol = 1.0e-13
            update_b!(cache, br2)
            # Tighter than the requested tolerance on purpose: a resolve off
            # stale state still lands near the answer, just orders short of it.
            @test solve!(cache).u ≈ Ar \ expected rtol = 1.0e-13
        end
    end

    # The Arnoldi loop builds its Krylov space from `Pl \ (b - Ax)`, so the
    # starting residual has to be preconditioned the same way. `_init_cacheval`
    # applied `Pl` instead of `Pl \ `, which seeded the first pass with the wrong
    # vector: the solve still reported `Success`, off by a factor of ~4000.
    @testset "Simple GMRES preconditioned: restart = $restart, blocksize = $blocksize" for
        restart in (true, false), blocksize in (0, 2)

        np = 6
        Ap = [
            float(i == j ? 10 + 5i : 0.3 * (i + j)) *
                (blocksize == 0 || (i - 1) ÷ blocksize == (j - 1) ÷ blocksize)
                for i in 1:np, j in 1:np
        ]
        bp = float.(1:np)
        alg = SimpleGMRES(; restart, blocksize)

        for (desc, Pl, Pr) in (
                ("Pl", Diagonal(diag(Ap)), I),
                ("Pr", I, Diagonal(diag(Ap))),
                ("Pl and Pr", Diagonal(sqrt.(diag(Ap))), Diagonal(sqrt.(diag(Ap)))),
            )
            @testset "$desc" begin
                sol = solve(
                    LinearProblem(copy(Ap), copy(bp)), alg;
                    Pl, Pr, abstol = 1.0e-12, reltol = 1.0e-12, maxiters = 100
                )
                @test SciMLBase.successful_retcode(sol)
                @test sol.u ≈ Ap \ bp rtol = 1.0e-8
            end
        end
    end

    @testset "KrylovJL" begin
        kwargs = (; gmres_restart = 5)
        precs = (A, p = nothing) -> (Diagonal(inv.(diag(A))), I)
        algorithms = (
            ("Default", KrylovJL(; kwargs...)),
            ("CG", KrylovJL_CG(; kwargs...)),
            ("GMRES", KrylovJL_GMRES(; kwargs...)),
            ("FGMRES", KrylovJL_FGMRES(; kwargs...)),
            ("GMRES_prec", KrylovJL_GMRES(; precs, ldiv = false, kwargs...)),
            ("FGMRES_prec", KrylovJL_FGMRES(; precs, ldiv = false, kwargs...)),
            # ("BICGSTAB",KrylovJL_BICGSTAB(; kwargs...)),
            ("MINRES", KrylovJL_MINRES(; kwargs...)),
            ("MINARES", KrylovJL_MINARES(; kwargs...)),
        )
        for (name, algorithm) in algorithms
            @testset "$name" begin
                test_interface(algorithm, prob1, prob2)
                test_interface(algorithm, prob3, prob4)
                test_tolerance_update(algorithm, prob5, u5)
            end
        end
    end

    @testset "Reuse precs" begin
        num_precs_calls = 0

        function countingprecs(A, p = nothing)
            num_precs_calls += 1
            (Diagonal(inv.(diag(A))), I)
        end

        n = 10
        A = spdiagm(-1 => -ones(n - 1), 0 => fill(10.0, n), 1 => -ones(n - 1))
        b = rand(n)
        p = LinearProblem(A, b)
        cache = init(p, KrylovJL_CG(precs = countingprecs, ldiv = false))
        x0 = copy(solve!(cache))
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
            for alg in (
                    ("Default", IterativeSolversJL(; kwargs...)),
                    ("CG", IterativeSolversJL_CG(; kwargs...)),
                    ("GMRES", IterativeSolversJL_GMRES(; kwargs...)),
                    ("IDRS", IterativeSolversJL_IDRS(; kwargs...)),
                    ("IDRS(2)", IterativeSolversJL_IDRS(; idrs_s = 2, kwargs...)),
                    ("MINRES", IterativeSolversJL_MINRES(; kwargs...)),
                    # BICGSTAB stays out: IterativeSolvers' own bicgstabl breaks
                    # down on the identity `prob1` here and throws
                    # "matrix contains Infs or NaNs" out of LAPACK, which is an
                    # upstream numerical issue rather than a wiring problem on
                    # this side.
                    # ("BICGSTAB",IterativeSolversJL_BICGSTAB(; kwargs...)),
                )
                @testset "$(alg[1])" begin
                    test_interface(alg[2], prob1, prob2)
                    test_interface(alg[2], prob3, prob4)
                    test_tolerance_update(alg[2], prob5, u5)
                end
            end

            @testset "tolerances as algorithm kwargs (#24)" begin
                # `idrs_iterable!` takes abstol/reltol/maxiter positionally and
                # accepts only `smoothing`/`verbose` as keywords, so forwarding
                # them from `alg.kwargs` used to raise a MethodError. MINRES read
                # `.residual`, which its iterable calls `resnorm`, and threw a
                # FieldError on every solve.
                # Note these solves report `ReturnCode.Default` rather than
                # `Success`: this extension never passes a retcode to
                # `build_linear_solution`. That is pre-existing and separate from
                # what is tested here, so assert on the solution itself.
                for alg in (
                        IterativeSolversJL_IDRS(abstol = 1.0e-10, reltol = 1.0e-10),
                        IterativeSolversJL_MINRES(abstol = 1.0e-10, reltol = 1.0e-10),
                        IterativeSolversJL_CG(abstol = 1.0e-10, reltol = 1.0e-10),
                    )
                    sol = solve(LinearProblem(A5, b5), alg)
                    @test sol.u ≈ u5 rtol = 1.0e-6
                end
            end

            @testset "maxiters on the algorithm (#175)" begin
                # LinearSolve spells it `maxiters`, IterativeSolvers `maxiter`.
                # Passing the LinearSolve spelling on the algorithm used to
                # forward an unknown keyword and raise a MethodError.
                for f in (
                        IterativeSolversJL_CG, IterativeSolversJL_GMRES,
                        IterativeSolversJL_IDRS, IterativeSolversJL_MINRES,
                        IterativeSolversJL_BICGSTAB,
                    )
                    @test solve(LinearProblem(A5, b5), f(maxiters = 200)).u ≈ u5 rtol = 1.0e-6
                end

                # The algorithm-level value caps the iteration count, and an
                # explicit `maxiter` still wins if both are given.
                slow = LinearProblem(Symmetric(Matrix(A5) + 0.5I), b5)
                for m in (2, 5)
                    @test solve(slow, IterativeSolversJL_CG(maxiters = m)).iters <= m
                end
                @test solve(slow, IterativeSolversJL_CG(maxiter = 3, maxiters = 50)).iters == 3
            end
        end
    end

    if VERSION > v"1.9-"
        @testset "KrylovKit" begin
            kwargs = (; gmres_restart = 5)
            for alg in (
                    ("Default", KrylovKitJL(; kwargs...)),
                    ("CG", KrylovKitJL_CG(; kwargs...)),
                    ("GMRES", KrylovKitJL_GMRES(; kwargs...)),
                )
                @testset "$(alg[1])" begin
                    test_interface(alg[2], prob1, prob2)
                    test_interface(alg[2], prob3, prob4)
                    test_tolerance_update(alg[2], prob5, u5)
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
            @test abs(norm(A * sol2.u .- b) - norm(A * sol.u .- b)) < 1.0e-12

            # Regression test for https://github.com/SciML/LinearSolve.jl/issues/936
            # CHOLMODFactorization must handle Float32 sparse matrices without
            # tripping the cacheval's type assertion (the cache used to be
            # initialized with a Factor{Float64} regardless of the input eltype).
            for T in (Float32, Float64)
                A32 = T.(sprand(50, 50, 0.1))
                A32 = A32 * A32' + 10I
                A32 = T.(A32)
                b32 = rand(T, 50)

                prob32 = LinearProblem(A32, b32)
                sol32 = solve(prob32, CHOLMODFactorization())
                @test eltype(sol32.u) === T
                @test norm(A32 * sol32.u - b32) < sqrt(eps(T)) * 100

                prob32s = LinearProblem(Symmetric(A32), b32)
                sol32s = solve(prob32s, CHOLMODFactorization())
                @test eltype(sol32s.u) === T
                @test norm(A32 * sol32s.u - b32) < sqrt(eps(T)) * 100
            end
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

    @testset "Operators with has_concretization" begin
        n = 4
        Random.seed!(42)
        A_sparse = sprand(n, n, 0.8) + I
        b = rand(n)

        # Create a MatrixOperator wrapping the sparse matrix
        A_op = MatrixOperator(A_sparse)

        prob_matrix = LinearProblem(A_sparse, b)
        prob_operator = LinearProblem(A_op, b)

        # Test KLU with operator
        sol_matrix = solve(prob_matrix, KLUFactorization())
        sol_operator = solve(prob_operator, KLUFactorization())
        @test sol_matrix.u ≈ sol_operator.u

        # Test UMFPACK with operator
        sol_matrix = solve(prob_matrix, UMFPACKFactorization())
        sol_operator = solve(prob_operator, UMFPACKFactorization())
        @test sol_matrix.u ≈ sol_operator.u

        # Test WOperator with sparse Jacobian
        n_w = 8
        M = sparse(I(n_w) * 1.0)
        gamma = 1 / 2.0
        J = sprand(n_w, n_w, 0.5) + sparse(I(n_w) * 10.0)  # Make it diagonally dominant
        u = rand(n_w)
        b_w = rand(n_w)

        W = WOperator{true}(M, gamma, J, u)
        W_matrix = convert(AbstractMatrix, W)

        prob_woperator = LinearProblem(W, b_w)
        prob_wmatrix = LinearProblem(W_matrix, b_w)

        # Test KLU with WOperator
        sol_woperator = solve(prob_woperator, KLUFactorization())
        sol_wmatrix = solve(prob_wmatrix, KLUFactorization())
        @test sol_woperator.u ≈ sol_wmatrix.u

        # Test UMFPACK with WOperator
        sol_woperator = solve(prob_woperator, UMFPACKFactorization())
        sol_wmatrix = solve(prob_wmatrix, UMFPACKFactorization())
        @test sol_woperator.u ≈ sol_wmatrix.u
    end

    @testset "Solve Function" begin
        A1 = rand(n) |> Diagonal
        b1 = rand(n)
        x1 = zero(b1)
        A2 = rand(n) |> Diagonal
        b2 = rand(n)
        x2 = zero(b1)

        @testset "LinearSolveFunction" begin
            function sol_func(
                    A, b, u, p, newA, Pl, Pr, solverdata; verbose = true,
                    kwargs...
                )
                if verbose == true
                    println("out-of-place solve")
                end
                u .= A \ b
            end

            function sol_func!(
                    A, b, u, p, newA, Pl, Pr, solverdata; verbose = true,
                    kwargs...
                )
                if verbose == true
                    println("in-place solve")
                end
                ldiv!(u, A, b)
            end

            prob1 = LinearProblem(A1, b1; u0 = x1)
            prob2 = LinearProblem(A1, b1; u0 = x1)

            for alg in (
                    LinearSolveFunction(sol_func),
                    LinearSolveFunction(sol_func!),
                )
                test_interface(alg, prob1, prob2)
            end
        end

        @testset "DirectLdiv!" begin
            function get_operator(A, u; add_inverse = true)
                function f(v, u, p, t)
                    println("using FunctionOperator OOP mul")
                    A * v
                end
                function f(w, v, u, p, t)
                    println("using FunctionOperator IIP mul")
                    mul!(w, A, v)
                end

                function fi(v, u, p, t)
                    println("using FunctionOperator OOP div")
                    A \ v
                end
                function fi(w, v, u, p, t)
                    println("using FunctionOperator IIP div")
                    ldiv!(w, A, v)
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

    @testset "Sparse matrix (check pattern_changed)" begin
        n = 4
        A = spdiagm(1 => ones(n - 1), 0 => fill(2.0, n), -1 => ones(n - 1))
        b = rand(n)
        linprob = @inferred LinearProblem(A, b)
        alg = @inferred LUFactorization()
        linsolve = @inferred init(linprob, alg)
        linres = @inferred solve!(linsolve)
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

# https://github.com/SciML/LinearSolve.jl/issues/869
# Test that memory kwarg works for GMRES (doesn't error)
@testset "Krylov.jl memory kwarg (issue #869)" begin
    A = sprand(100, 100, 0.1) + 10I  # Well-conditioned matrix
    b = rand(100)

    # Test GMRES with memory kwarg - should not error and should converge
    # Previously, passing memory kwarg would cause a MethodError because
    # memory was incorrectly passed to krylov_solve! instead of workspace creation
    prob = LinearProblem(A, b)
    linsolve = init(prob, KrylovJL_GMRES(memory = 30))
    sol = solve!(linsolve)
    @test sol.retcode == ReturnCode.Success
    @test norm(A * sol.u - b) < 1.0e-6

    # Test with different memory values to ensure it's actually being used
    prob2 = LinearProblem(A, b)
    linsolve2 = init(prob2, KrylovJL_GMRES(memory = 10))
    sol2 = solve!(linsolve2)
    @test sol2.u isa Vector
end

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

@testset "BlockDiagonal blockwise factorizations (#203)" begin
    BDExt = Base.get_extension(LinearSolve, :LinearSolveBlockDiagonalsExt)
    BDFact = BDExt.BlockDiagonalFactorization

    for bsizes in ([3, 3, 3, 3], [2, 3, 4])
        A = BlockDiagonal([rand(n, n) + n * I for n in bsizes])
        b = rand(size(A, 1))
        xref = Matrix(A) \ b

        # LU and QR factorize block by block instead of handing the whole
        # BlockDiagonal to the generic scalar LinearAlgebra path.
        for alg in (
                LUFactorization(), LUFactorization(LinearAlgebra.NoPivot()),
                QRFactorization(LinearAlgebra.NoPivot()),
                QRFactorization(LinearAlgebra.ColumnNorm()),
            )
            cache = init(LinearProblem(A, copy(b)), alg)
            @test cache.cacheval isa BDFact
            sol = solve!(cache)
            @test SciMLBase.successful_retcode(sol)
            @test sol.u ≈ xref

            # Re-solving with a new b reuses the stored per-block factorizations.
            b2 = rand(size(A, 1))
            cache.b = b2
            @test solve!(cache).u ≈ Matrix(A) \ b2
        end

        # The algorithms that do not specialize keep working unchanged.
        for alg in (nothing, GenericFactorization(), SimpleGMRES(), KrylovJL_GMRES())
            s = alg === nothing ? solve(LinearProblem(A, copy(b))) :
                solve(LinearProblem(A, copy(b)), alg)
            @test s.u ≈ xref rtol = 1.0e-6
        end

        # Batched right hand sides go through the same per-block solves.
        B = rand(size(A, 1), 3)
        @test solve(LinearProblem(A, copy(B)), LUFactorization()).u ≈ Matrix(A) \ B
    end

    # Element types other than Float64 take the same path.
    A32 = BlockDiagonal([rand(Float32, 3, 3) + 3I for _ in 1:3])
    b32 = rand(Float32, size(A32, 1))
    sol32 = solve(LinearProblem(A32, copy(b32)), LUFactorization())
    @test eltype(sol32.u) === Float32
    @test sol32.u ≈ Matrix(A32) \ b32 rtol = 1.0f-3

    Ac = BlockDiagonal([rand(ComplexF64, 3, 3) + 3I for _ in 1:3])
    bc = rand(ComplexF64, size(Ac, 1))
    @test solve(LinearProblem(Ac, copy(bc)), LUFactorization()).u ≈ Matrix(Ac) \ bc

    # A singular block reports failure rather than crashing.
    Asing = BlockDiagonal([zeros(3, 3), rand(3, 3) + 3I])
    sol_sing = solve(LinearProblem(Asing, rand(6)), LUFactorization())
    @test sol_sing.retcode === ReturnCode.Failure

    # Rectangular blocks do not decompose into independent square subsystems,
    # so they keep the generic dense representation.
    Arect = BlockDiagonal([rand(2, 3), rand(3, 2)])
    cache_rect = init(LinearProblem(Arect, rand(5)), LUFactorization())
    @test !(cache_rect.cacheval isa BDFact)

    # residualsafety stays on the blockwise representation. The blocks are
    # factorized from a copy so the residual check still sees an unfactored A.
    Asafe = BlockDiagonal([rand(3, 3) + 3I for _ in 1:3])
    bsafe = rand(size(Asafe, 1))
    Asafe_before = Matrix(Asafe)
    cache_safe = init(LinearProblem(Asafe, copy(bsafe)), LUFactorization(residualsafety = true))
    @test cache_safe.cacheval isa BDFact
    sol_safe = solve!(cache_safe)
    @test SciMLBase.successful_retcode(sol_safe)
    @test sol_safe.u ≈ Matrix(Asafe) \ bsafe
    @test Matrix(cache_safe.A) ≈ Asafe_before

    # The default algorithm picks the direct blockwise solve rather than falling
    # through to the matrix-free KrylovJL_GMRES arm.
    Adef = BlockDiagonal([rand(4, 4) + 4I for _ in 1:5])
    bdef = rand(size(Adef, 1))
    @test LinearSolve.defaultalg(Adef, bdef).alg ===
        LinearSolve.DefaultAlgorithmChoice.LUFactorization
    sol_def = solve(LinearProblem(Adef, copy(bdef)))
    @test SciMLBase.successful_retcode(sol_def)
    @test sol_def.u ≈ Matrix(Adef) \ bdef

    # Rectangular blocks are not independent subsystems, so the default there
    # stays on the generic operator handling.
    @test LinearSolve.defaultalg(BlockDiagonal([rand(2, 3), rand(3, 2)]), rand(5)).alg ===
        LinearSolve.DefaultAlgorithmChoice.KrylovJL_GMRES
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
    @time "solve MySparseMatrixCSC" u = solve(pr)
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

    pr = LinearProblem(B, b)
    solver = UMFPACKFactorization()
    cache = init(pr, solver)
    u = solve!(cache)
    @test norm(u - u0, Inf) < 1.0e-8
    reinit!(cache; A = B1, b = b1)
    u = solve!(cache)
    @test norm(u - u0, Inf) < 1.0e-8

    pr = LinearProblem(B, b)
    solver = KLUFactorization()

    # Regression test for #737: KLU should work with AbstractSparseMatrixCSC wrappers
    sol = solve(pr, solver)
    @test norm(sol.u - u0, Inf) < 1.0e-8

    # Repeat direct solve to exercise cache-init/reuse paths through solve(prob, alg)
    sol = solve(pr, solver)
    @test norm(sol.u - u0, Inf) < 1.0e-8

    cache = init(pr, solver)
    u = solve!(cache)
    @test norm(u - u0, Inf) < 1.0e-8
    reinit!(cache; A = B1, b = b1)
    u = solve!(cache)
    @test norm(u - u0, Inf) < 1.0e-8
end

@testset "ParallelSolves" begin
    n = 1000
    @info "ParallelSolves: Threads.nthreads()=$(Threads.nthreads())"
    A_sparse = 10I - sprand(n, n, 0.01)
    B = [rand(n), rand(n)]
    U = [A_sparse \ B[i] for i in 1:2]
    sol = similar(U)

    Threads.@threads for i in 1:2
        sol[i] = solve(LinearProblem(A_sparse, B[i]), UMFPACKFactorization())
    end

    for i in 1:2
        @test sol[i] ≈ U[i]
    end

    Threads.@threads for i in 1:2
        sol[i] = solve(LinearProblem(A_sparse, B[i]), KLUFactorization())
    end
    for i in 1:2
        @test sol[i] ≈ U[i]
    end

    Threads.@threads for i in 1:2
        sol[i] = solve(LinearProblem(A_sparse, B[i]), SparspakFactorization())
    end
    for i in 1:2
        @test sol[i] ≈ U[i]
    end
end

@static if isdefined(@__MODULE__, :AlgebraicMultigrid)
    @testset "AlgebraicMultigridJL" begin
        n = 100
        A_amg = spdiagm(-1 => -ones(n - 1), 0 => 2 * ones(n), 1 => -ones(n - 1))
        b_amg = rand(n)
        prob_amg = LinearProblem(A_amg, b_amg)

        # Default (Ruge-Stuben)
        sol_amg = solve(prob_amg, AlgebraicMultigridJL())
        @test norm(A_amg * sol_amg.u - b_amg) < 1.0e-6

        # Smoothed Aggregation
        sol_amg = solve(prob_amg, AlgebraicMultigridJL(AlgebraicMultigrid.SmoothedAggregationAMG()))
        @test norm(A_amg * sol_amg.u - b_amg) < 1.0e-6

        # With tighter tolerance
        sol_amg = solve(prob_amg, AlgebraicMultigridJL(), reltol = 1.0e-8)
        @test norm(A_amg * sol_amg.u - b_amg) < 1.0e-8

        # Non-square matrix should throw. `needs_square_A(::AlgebraicMultigridJL)`
        # is true, so this is now rejected at `init` with an `ArgumentError`
        # naming the least-squares alternatives, rather than reaching the
        # solver and tripping its `AssertionError`.
        A_rect = sparse([1.0 1.0 0.0; 0.0 1.0 1.0])
        b_rect = [1.0, 1.0]
        @test_throws ArgumentError solve(LinearProblem(A_rect, b_rect), AlgebraicMultigridJL())
    end
end

@static if isdefined(@__MODULE__, :LinearSolvePyAMG)
    @testset "PyAMG" begin
        n = 100
        A_pyamg = spdiagm(-1 => -ones(n - 1), 0 => 2 * ones(n), 1 => -ones(n - 1))
        b_pyamg = rand(n)
        prob_pyamg = LinearProblem(A_pyamg, b_pyamg)

        # Ruge-Stuben (default)
        sol_pyamg = solve(prob_pyamg, PyAMG())
        @test norm(A_pyamg * sol_pyamg.u - b_pyamg) < 1.0e-6

        # Smoothed Aggregation
        sol_pyamg = solve(prob_pyamg, PyAMG_SmoothedAggregation())
        @test norm(A_pyamg * sol_pyamg.u - b_pyamg) < 1.0e-6

        # CG acceleration
        sol_pyamg = solve(prob_pyamg, PyAMG(accel = "cg"), reltol = 1.0e-8)
        @test norm(A_pyamg * sol_pyamg.u - b_pyamg) < 1.0e-8

        # GMRES acceleration
        sol_pyamg = solve(prob_pyamg, PyAMG(accel = "gmres"), reltol = 1.0e-6)
        @test norm(A_pyamg * sol_pyamg.u - b_pyamg) < 1.0e-6

        # Re-solve with different b, same A
        b_pyamg2 = rand(n)
        cache_pyamg = init(prob_pyamg, PyAMG(accel = "cg"))
        solve!(cache_pyamg)
        reinit!(cache_pyamg; b = b_pyamg2)
        sol_pyamg2 = solve!(cache_pyamg)
        @test norm(A_pyamg * sol_pyamg2.u - b_pyamg2) < 1.0e-6
    end
end

# Integer-eltype problems are promoted to float at `init`, matching `\` (division
# does not stay in the integers). Previously every such input threw an opaque
# `InexactError` from inside `ldiv!` or a `MethodError` from the QR/Krylov wrappers.
# Regression test for https://github.com/SciML/LinearSolve.jl/issues/206
@testset "Integer eltype promotion (#206)" begin
    # The MWE from the issue: float A, integer b.
    A206 = [
        1.0 0.0 -1.0 0.0
        0.0 -3152.28 0.0 3152.28
        0.388658 0.921382 -1.76854 -1.45868
        0.921382 -0.388658 1.45868 1.76854
    ]
    b206 = [0, 0, 1, 0]
    res206 = A206 \ b206

    @testset "float A, integer b" begin
        for alg in (
                nothing, LUFactorization(), GenericLUFactorization(),
                QRFactorization(), KrylovJL_GMRES(),
            )
            prob = LinearProblem(copy(A206), copy(b206))
            sol = alg === nothing ? solve(prob) : solve(prob, alg)
            @test sol.retcode === ReturnCode.Success
            @test eltype(sol.u) == Float64
            @test sol.u ≈ res206
        end
        # A float tolerance with an integer `b` used to throw
        # `InexactError: Int64(1.0e-8)` from the init-time tolerance conversion.
        @test solve(LinearProblem(copy(A206), copy(b206)), reltol = 1.0e-8).u ≈ res206
    end

    @testset "integer A" begin
        Ai = [4 1 0 0; 1 4 1 0; 0 1 4 1; 0 0 1 4]
        @test solve(LinearProblem(copy(Ai), copy(b206))).u ≈ Ai \ b206
        # Large enough to leave the small-matrix default path.
        Random.seed!(206)
        Al = rand(1:9, 50, 50) + 200I
        bl = rand(1:9, 50)
        @test solve(LinearProblem(copy(Al), copy(bl))).u ≈ Al \ bl
        # Sparse and structured containers keep their structure through `float`.
        @test solve(LinearProblem(sparse(Ai), copy(b206))).u ≈ Float64.(Ai) \ b206
        @test solve(LinearProblem(Diagonal([2, 4, 5, 8]), copy(b206))).u ≈
            Diagonal([2, 4, 5, 8]) \ b206
        @test solve(
            LinearProblem(Tridiagonal([1, 1, 1], [4, 4, 4, 4], [1, 1, 1]), copy(b206))
        ).u ≈ Tridiagonal([1, 1, 1], [4, 4, 4, 4], [1, 1, 1]) \ b206
    end

    @testset "other integer-like eltypes" begin
        # Bool is an Integer; Complex{Int} and BigInt promote to ComplexF64/BigFloat.
        @test solve(LinearProblem(copy(A206), [false, false, true, false])).u ≈ res206
        solc = solve(LinearProblem(copy(A206), Complex{Int}[0, 0, 1, 0]))
        @test eltype(solc.u) == ComplexF64
        @test solc.u ≈ res206
        Abig = big.([4 1 0 0; 1 4 1 0; 0 1 4 1; 0 0 1 4])
        solb = solve(LinearProblem(Abig, big.(b206)))
        @test eltype(solb.u) == BigFloat
        @test solb.u ≈ Abig \ big.(b206)
    end

    @testset "integer u0 and batched integer b" begin
        # `u0` must be on the problem: a `u0` kwarg to `solve` lands in `__init`'s
        # trailing kwargs and is silently ignored.
        sol = solve(LinearProblem(copy(A206), copy(b206); u0 = [0, 0, 0, 0]))
        @test eltype(sol.u) == Float64
        @test sol.u ≈ res206
        B = [0 1; 0 2; 1 3; 0 4]
        solB = solve(LinearProblem(copy(A206), copy(B)))
        @test solB.u ≈ A206 \ Float64.(B)
    end

    @testset "wrapped and abstractly-typed integer A" begin
        # Adjoint/Transpose/BitMatrix are not `DenseMatrix`, so these only work if
        # promotion happens before `defaultalg` sees the type: choosing on the
        # unpromoted wrapper while the cache holds the promoted dense matrix left
        # the Krylov workspace slot typed `Nothing` and threw a `TypeError`.
        Ai = [4 1 0 0; 1 4 1 0; 0 1 4 1; 0 0 1 4]
        @test solve(LinearProblem(adjoint(copy(Ai)), copy(b206))).u ≈ Ai' \ b206
        @test solve(LinearProblem(transpose(copy(Ai)), copy(b206))).u ≈
            transpose(Ai) \ b206
        Abit = Bool[1 0 0 0; 1 1 0 0; 0 1 1 0; 0 0 1 1]
        @test solve(LinearProblem(copy(Abit), copy(b206))).u ≈ Abit \ b206
        # Symmetric{Int} with a float b: `A` is promoted while `b` is not; the
        # default solver's `A_backup` must be typed on the promoted matrix or the
        # first solve's safety backup throws `InexactError` from integer storage.
        As = Symmetric([4 1 0 0; 1 4 1 0; 0 1 4 1; 0 0 1 4])
        bf = [0.0, 0.0, 1.0, 0.0]
        @test solve(LinearProblem(As, copy(bf))).u ≈ As \ bf
        # Abstractly-typed integer arrays cannot go through `float(::AbstractArray)`;
        # they promote to the Float64 default instead of crashing.
        solabs = solve(LinearProblem(copy(A206), Integer[0, 0, 1, 0]))
        @test eltype(solabs.u) == Float64
        @test solabs.u ≈ res206
    end

    @testset "mixed exact/integer matches \\ via joint promotion" begin
        # An integer operand against a Rational one promotes to Rational, so the
        # solve stays exact, matching `\` -- not to Float64, which would have
        # returned float-accuracy values (or worse, dressed them as Rationals).
        Ar = Rational{Int}[4 1 0 0; 1 4 1 0; 0 1 4 1; 0 0 1 4]
        solra = solve(LinearProblem(copy(Ar), copy(b206)))
        @test eltype(solra.u) == Rational{Int}
        @test solra.u == Ar \ b206
        Ai = [4 1 0 0; 1 4 1 0; 0 1 4 1; 0 0 1 4]
        brat = Rational{Int}[0, 0, 1, 0]
        solrb = solve(LinearProblem(copy(Ai), copy(brat)))
        @test eltype(solrb.u) == Rational{Int}
        # Deliberate divergence from `\` here: `factorize(::Matrix{Int})` sends
        # Base to Float64 for this orientation, while the joint promotion keeps
        # the exact Rational solve (value-equal to floating accuracy).
        @test solrb.u == Rational{Int}.(Ai) \ brat
        @test solrb.u ≈ Ai \ brat
        # An integer operand against Float32/BigFloat takes that type, not Float64.
        @test eltype(solve(LinearProblem(Float32.(A206), copy(b206))).u) == Float32
        @test eltype(solve(LinearProblem(big.(A206), copy(b206))).u) == BigFloat
    end

    @testset "cache reuse after integer init" begin
        # The promoted cache's `A_backup` must accept any float update: assigning a
        # fractional `A` after an all-integer init used to throw `InexactError`
        # from the Int-typed backup during the safety copy.
        Ai = [4 1 0 0; 1 4 1 0; 0 1 4 1; 0 0 1 4]
        cache = init(LinearProblem(copy(Ai), copy(b206)))
        @test solve!(cache).u ≈ Ai \ b206
        Afrac = [0.5 0.1 0.0 0.0; 0.1 0.5 0.1 0.0; 0.0 0.1 0.5 0.1; 0.0 0.0 0.1 0.5]
        cache.A = copy(Afrac)
        @test solve!(cache).u ≈ Afrac \ b206
    end

    @testset "AbstractSolveFunction receives the user's arrays" begin
        # Custom solve functions define their own semantics (exact integer solves,
        # GF(2) arithmetic on Bool arrays); promotion must not touch their inputs.
        received = Ref{Any}(nothing)
        function record_solve!(A, b, u, p, isfresh, Pl, Pr, cacheval; kwargs...)
            received[] = (typeof(A), typeof(b))
            u .= b .÷ 2
            return u
        end
        Ai = [2 0; 0 2]
        sol = solve(LinearProblem(copy(Ai), [4, 4]), LinearSolveFunction(record_solve!))
        @test received[] == (Matrix{Int}, Vector{Int})
        @test sol.u == [2, 2]
        @test eltype(sol.u) == Int
    end

    @testset "cache reuse converts integer updates" begin
        cache = init(LinearProblem(copy(A206), copy(b206)))
        @test solve!(cache).u ≈ res206
        # `cache.b`/`cache.A` are float-typed fields; assigning integer arrays
        # converts through `setproperty!` rather than reintroducing integer storage.
        cache.b = [1, 0, 0, 0]
        @test solve!(cache).u ≈ A206 \ [1.0, 0, 0, 0]
    end

    @testset "Rational stays exact (not promoted)" begin
        Ar = Rational{Int}[4 1 0 0; 1 4 1 0; 0 1 4 1; 0 0 1 4]
        br = Rational{Int}[0, 0, 1, 0]
        sol = solve(LinearProblem(copy(Ar), copy(br)))
        @test eltype(sol.u) == Rational{Int}
        @test sol.u == Ar \ br
    end
end
