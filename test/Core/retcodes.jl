using LinearSolve, LinearAlgebra, RecursiveFactorization, StaticArrays, Test

alglist = (
    LUFactorization,
    QRFactorization,
    KrylovJL_GMRES,
    GenericLUFactorization,
    RFLUFactorization,
    SVDFactorization,
    NormalCholeskyFactorization,
    KrylovJL_CRAIGMR,
    KrylovJL_LSMR,
)

@testset "Success" begin
    for alg in alglist
        A = [2.0 1.0; -1.0 1.0]
        b = [-1.0, 1.0]
        prob = LinearProblem(A, b)
        linsolve = init(prob, alg())
        sol = solve!(linsolve)
        @test SciMLBase.successful_retcode(sol.retcode)
    end
end

lualgs = (
    LUFactorization(),
    QRFactorization(),
    GenericLUFactorization(),
    LinearSolve.DefaultLinearSolver(
        LinearSolve.DefaultAlgorithmChoice.LUFactorization; safetyfallback = false
    ),
    RFLUFactorization(),
    NormalCholeskyFactorization(),
)
@testset "Failure" begin
    for alg in lualgs
        @show alg
        A = [1.0 1.0; 1.0 1.0]
        b = [-1.0, 1.0]
        prob = LinearProblem(A, b)
        linsolve = init(prob, alg)
        sol = solve!(linsolve)
        if alg isa NormalCholeskyFactorization
            # This is a known and documented incorrectness in NormalCholeskyFactorization
            # due to numerical instability in its method that is fundamental.
            @test SciMLBase.successful_retcode(sol.retcode)
        else
            @test !SciMLBase.successful_retcode(sol.retcode)
        end
    end
end

rankdeficientalgs = (
    QRFactorization(LinearAlgebra.ColumnNorm()),
    KrylovJL_GMRES(),
    SVDFactorization(),
    KrylovJL_CRAIGMR(),
    KrylovJL_LSMR(),
    LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.LUFactorization),
)

@testset "Rank Deficient Success" begin
    for alg in rankdeficientalgs
        @show alg
        A = [1.0 1.0; 1.0 1.0]
        b = [-1.0, 1.0]
        prob = LinearProblem(A, b)
        linsolve = init(prob, alg)
        sol = solve!(linsolve)
        @test SciMLBase.successful_retcode(sol.retcode)
    end
end

staticarrayalgs = (
    DirectLdiv!(),
    LUFactorization(),
    CholeskyFactorization(),
    NormalCholeskyFactorization(),
    SVDFactorization(),
)
@testset "StaticArray Success" begin
    A = Float64[1 2 3; 4 3.5 1.7; 5.2 1.8 9.7]
    A = A * A'
    b = Float64[2, 5, 8]
    prob1 = LinearProblem(SMatrix{3, 3}(A), SVector{3}(b))
    sol = solve(prob1)
    @test SciMLBase.successful_retcode(sol.retcode)

    for alg in staticarrayalgs
        sol = solve(prob1, alg)
        @test SciMLBase.successful_retcode(sol.retcode)
    end

    @test_broken sol = solve(prob1, QRFactorization()) # Needs StaticArrays `qr` fix
end

@testset "StaticArray Singular" begin
    A2 = SA[0.0 0.0; 1.0 -1.0]
    A4 = SMatrix{4, 4}(
        [
            1.0 2.0 3.0 4.0
            2.0 4.0 6.0 8.0
            0.0 1.0 0.0 1.0
            1.0 0.0 1.0 0.0
        ]
    )
    for (A, b, B) in (
            (A2, SA[1.0, 2.0], SMatrix{2, 2}(1.0I)),
            (A4, SA[1.0, 2.0, 3.0, 4.0], SMatrix{4, 4}(1.0I)),
        )
        for rhs in (b, B)
            # Default alg rescues singular LU with an SVD least-squares solve,
            # returning the finite min-norm pseudo-solution (like the dense
            # default's pivoted-QR rescue).
            sol = solve(LinearProblem(A, rhs))
            @test SciMLBase.successful_retcode(sol.retcode)
            @test all(isfinite, sol.u)
            @test sol.u isa (rhs isa SVector ? SVector : SMatrix)
            @test sol.u ≈ pinv(Matrix(A)) * (rhs isa SVector ? Vector(rhs) : Matrix(rhs))

            # Explicit LUFactorization reports Failure without throwing,
            # matching the dense LUFactorization behavior on singular input.
            sole = solve(LinearProblem(A, rhs), LUFactorization())
            @test sole.retcode == ReturnCode.Failure
            @test all(isfinite, sole.u)
        end
    end

    # Nonsingular static solves match `\` (up to FMA contraction differences
    # in StaticArrays' inlined small-size formulas, per review the check is
    # allowed to inline rather than preserving bit-identity)
    A2n = SA[2.0 1.0; 1.0 3.0]
    A4n = SMatrix{4, 4}(
        [
            4.0 1.0 0.0 2.0
            1.0 5.0 1.0 0.0
            0.0 1.0 6.0 1.0
            2.0 0.0 1.0 7.0
        ]
    )
    for (A, b, B) in (
            (A2n, SA[1.0, 2.0], SA[1.0 0.5; 2.0 1.5]),
            (
                A4n, SA[1.0, 2.0, 3.0, 4.0],
                SMatrix{4, 2}([1.0 0.5; 2.0 1.5; 3.0 2.5; 4.0 3.5]),
            ),
        )
        for rhs in (b, B)
            @test solve(LinearProblem(A, rhs)).u ≈ A \ rhs rtol = 4 * eps()
            @test solve(LinearProblem(A, rhs), LUFactorization()).u === lu(A) \ rhs
        end
    end

    # GESVFactorization on static problems: the static analog of the dense
    # gesv driver — one-shot solve, Failure through the retcode on singular,
    # no SVD rescue.
    for (A, b) in ((A2n, SA[1.0, 2.0]), (A4n, SVector{4}(1.0:4.0)))
        sol = solve(LinearProblem(A, b), GESVFactorization())
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ A \ b rtol = 4 * eps()
    end
    B2 = SMatrix{2, 2}(1.0I)
    @test solve(LinearProblem(A2n, B2), GESVFactorization()).u ≈ A2n \ B2
    As2 = SA[0.0 0.0; 1.0 -1.0]
    As4 = SMatrix{4, 4}([1.0 2.0 3.0 4.0; 2.0 4.0 6.0 8.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0])
    for (As, b) in ((As2, SA[1.0, 2.0]), (As4, SVector{4}(1.0:4.0)))
        sol = solve(LinearProblem(As, b), GESVFactorization())
        @test sol.retcode == ReturnCode.Failure
        @test all(iszero, sol.u)
    end
    @test_throws ArgumentError solve(
        LinearProblem(SMatrix{3, 2}(ones(3, 2)), SA[1.0, 2.0, 3.0]), GESVFactorization()
    )

    # Non-square static problems are unchanged by the singular handling
    Ans = SMatrix{3, 2}([1.0 2.0; 3.0 4.0; 5.0 6.0])
    bns = SA[1.0, 2.0, 3.0]
    solns = solve(LinearProblem(Ans, bns))
    @test SciMLBase.successful_retcode(solns.retcode)
    @test solns.u ≈ Matrix(Ans) \ Vector(bns)
end

@testset "MMatrix/MVector" begin
    A0 = [
        4.0 1.0 0.0 2.0
        1.0 5.0 1.0 0.0
        0.0 1.0 6.0 1.0
        2.0 0.0 1.0 7.0
    ]
    MA = MMatrix{4, 4}(A0)
    mb = MVector{4}([1.0, 2.0, 3.0, 4.0])
    mb2 = MVector{4}([4.0, 3.0, 2.0, 1.0])
    prob = LinearProblem(MA, mb)

    @test LinearSolve.defaultalg(MA, mb, LinearSolve.OperatorAssumptions(true)) isa
        LUFactorization
    @test LinearSolve.defaultalg(
        MMatrix{3, 2}(ones(3, 2)), MVector{3}(ones(3)),
        LinearSolve.OperatorAssumptions(false)
    ) isa SVDFactorization

    for alg in (DirectLdiv!(), GESVFactorization())
        s = solve(prob, alg)
        @test s.u isa MVector{4, Float64}
        @test s.retcode == ReturnCode.Success
        @test s.u ≈ A0 \ Vector(mb)
    end

    for alg in (
            nothing,
            LUFactorization(),
            LUFactorization(; pivot = NoPivot()),
            LUFactorization(; residualsafety = true),
            GenericLUFactorization(),
            GenericLUFactorization(NoPivot()),
            QRFactorization(),
            QRFactorization(ColumnNorm()),
            CholeskyFactorization(),
            NormalCholeskyFactorization(),
            SVDFactorization(),
            KrylovJL_GMRES(),
        )
        s = solve(prob, alg)
        @test s.u isa MVector{4, Float64}
        @test SciMLBase.successful_retcode(s.retcode)
        @test norm(A0 * Vector(s.u) - Vector(mb)) < 1.0e-10

        cache = init(LinearProblem(copy(MA), copy(mb)), alg)
        sc = solve!(cache)
        @test sc.u isa MVector{4, Float64}
        @test SciMLBase.successful_retcode(sc.retcode)
        @test norm(A0 * Vector(sc.u) - Vector(mb)) < 1.0e-10

        cache.b = copy(mb2)
        sc2 = solve!(cache)
        @test sc2.u isa MVector{4, Float64}
        @test norm(A0 * Vector(sc2.u) - Vector(mb2)) < 1.0e-10
    end

    # Batched MMatrix RHS
    MB = MMatrix{4, 2}([1.0 0.5; 2.0 1.5; 3.0 2.5; 4.0 3.5])
    for alg in (
            nothing, LUFactorization(), GenericLUFactorization(),
            QRFactorization(), SVDFactorization(),
        )
        sb = solve(LinearProblem(MA, MB), alg)
        @test sb.u isa MMatrix{4, 2, Float64}
        @test norm(A0 * Matrix(sb.u) - Matrix(MB)) < 1.0e-10

        cb = init(LinearProblem(copy(MA), copy(MB)), alg)
        sb2 = solve!(cb)
        @test sb2.u isa MMatrix{4, 2, Float64}
        @test norm(A0 * Matrix(sb2.u) - Matrix(MB)) < 1.0e-10
    end

    # Singular MMatrix: default rescues via SVD; LU/GenericLU/GESV report Failure
    Asing4 = MMatrix{4, 4}(
        [1.0 2.0 3.0 4.0; 2.0 4.0 6.0 8.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]
    )
    sol_sing = solve(LinearProblem(Asing4, mb))
    @test SciMLBase.successful_retcode(sol_sing.retcode)
    @test sol_sing.u isa MVector{4, Float64}
    @test all(isfinite, sol_sing.u)
    @test sol_sing.u ≈ pinv(Matrix(Asing4)) * Vector(mb)

    for alg in (LUFactorization(), GenericLUFactorization(), GESVFactorization())
        sole = solve(LinearProblem(Asing4, mb), alg)
        @test sole.retcode == ReturnCode.Failure
        @test sole.u isa MVector{4, Float64}
        @test all(isfinite, sole.u)
    end

    # Non-square MMatrix
    for (m, n) in ((3, 2), (2, 3))
        Ans_m = MMatrix{m, n}(reshape(Float64.(1:(m * n)), m, n))
        bns_m = MVector{m}(Float64.(1:m))
        solns_m = solve(LinearProblem(Ans_m, bns_m))
        @test SciMLBase.successful_retcode(solns_m.retcode)
        @test solns_m.u isa MVector{n, Float64}
        @test solns_m.u ≈ pinv(Matrix(Ans_m)) * Vector(bns_m)

        cns_m = init(LinearProblem(copy(Ans_m), copy(bns_m)))
        solns_mc = solve!(cns_m)
        @test solns_mc.u isa MVector{n, Float64}
        @test solns_mc.u ≈ pinv(Matrix(Ans_m)) * Vector(bns_m)
    end

    # Integer, Float32, and ComplexF64 MMatrix/MVector
    MAi = MMatrix{2, 2}([2 1; 1 3])
    mbi = MVector{2}([1, 2])
    for alg in (nothing, LUFactorization(), GenericLUFactorization())
        si = solve(LinearProblem(MAi, mbi), alg)
        @test si.u isa MVector{2, Float64}
        @test si.u ≈ [0.2, 0.6]
    end
    for T in (Float32, ComplexF64)
        MAT = MMatrix{4, 4, T}(A0)
        mbT = MVector{4, T}(mb)
        for alg in (nothing, LUFactorization(), GenericLUFactorization())
            sT = solve(LinearProblem(MAT, mbT), alg)
            @test sT.u isa MVector{4, T}
            @test norm(MAT * sT.u - mbT) < (T === Float32 ? 1.0f-5 : 1.0e-11)
        end
    end

    # alias_A = false preserves MA; alias_A = true solves in place; SMatrix + GenericLU works
    for alg in (LUFactorization(), GenericLUFactorization())
        MA_keep = copy(MA)
        ck = init(LinearProblem(MA_keep, copy(mb)), alg)
        @test solve!(ck).u ≈ A0 \ Vector(mb)
        @test MA_keep == MA

        MA_mut = copy(MA)
        cm = init(
            LinearProblem(MA_mut, copy(mb)), alg;
            alias = LinearAliasSpecifier(alias_A = true)
        )
        @test solve!(cm).u ≈ A0 \ Vector(mb)
    end
    SA = SMatrix{4, 4}(A0)
    sb_s = SVector{4}(mb)
    sg = solve(LinearProblem(SA, sb_s), GenericLUFactorization())
    @test sg.u isa SVector{4, Float64}
    @test sg.u ≈ SA \ sb_s
end
