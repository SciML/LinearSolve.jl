using HYPRE
using HYPRE.LibHYPRE: HYPRE_BigInt,
    HYPRE_Complex, HYPRE_IJMatrixGetValues,
    HYPRE_IJVectorGetValues, HYPRE_Int
using LinearAlgebra
using LinearSolve
using MPI
using Random: MersenneTwister
using SparseArrays
using Test

MPI.Init()
HYPRE.Init()

# Convert from HYPREArrays to Julia arrays
function to_array(A::HYPREMatrix)
    i = (A.ilower):(A.iupper)
    j = (A.jlower):(A.jupper)
    nrows = HYPRE_Int(length(i))
    ncols = fill(HYPRE_Int(length(j)), length(i))
    rows = convert(Vector{HYPRE_BigInt}, i)
    cols = convert(Vector{HYPRE_BigInt}, repeat(j, length(i)))
    values = Vector{HYPRE_Complex}(undef, length(i) * length(j))
    HYPRE_IJMatrixGetValues(A.ijmatrix, nrows, ncols, rows, cols, values)
    return sparse(permutedims(reshape(values, (length(j), length(i)))))
end
function to_array(b::HYPREVector)
    i = (b.ilower):(b.iupper)
    nvalues = HYPRE_Int(length(i))
    indices = convert(Vector{HYPRE_BigInt}, i)
    values = Vector{HYPRE_Complex}(undef, length(i))
    HYPRE_IJVectorGetValues(b.ijvector, nvalues, indices, values)
    return values
end
to_array(x) = x

function failure_prob()
    n = 200
    # This SPD 1D Laplacian does not converge in a single PCG iteration, so with
    # maxiters = 1 we should get a non-success retcode rather than an exception.
    A = spdiagm(-1 => -ones(n - 1), 0 => 2.0 .* ones(n), 1 => -ones(n - 1))
    b = ones(n)
    return LinearProblem(A, b)
end

is_parasails_precond(Pl) = Pl isa HYPRE.ParaSails || (Pl isa DataType && Pl <: HYPRE.ParaSails)

function generate_probs(alg; Pl = LinearAlgebra.I)
    rng = MersenneTwister(1234)
    n = 100
    if alg.solver isa HYPRE.BoomerAMG || alg.solver === HYPRE.BoomerAMG
        # BoomerAMG needs a "nice" matrix so construct a simple FEM-like matrix.
        # Ironically this matrix doesn't play nice with the other solvers...
        I, J, V = Int[], Int[], Float64[]
        for i in 1:99
            k = (1 + rand(rng)) * [1.0 -1.0; -1.0 1.0]
            append!(V, k)
            append!(I, [i, i + 1, i, i + 1]) # rows
            append!(J, [i, i, i + 1, i + 1]) # cols
        end
        A = sparse(I, J, V)
        A[:, 1] .= 0
        A[1, :] .= 0
        A[:, end] .= 0
        A[end, :] .= 0
        A[1, 1] = 2
        A[end, end] = 2
    elseif is_parasails_precond(Pl)
        # ParaSails setup is fragile on the random sparse SPD family below on some
        # platforms. Use a deterministic strictly diagonally dominant SPD tridiagonal.
        A = spdiagm(-1 => -ones(n - 1), 0 => 4.0 .* ones(n), 1 => -ones(n - 1))
    else
        A = sprand(rng, n, n, 0.01) + 3 * LinearAlgebra.I
        A = A'A
    end
    A1 = A / 1
    @test isposdef(A1)
    b1 = rand(rng, n)
    x1 = zero(b1)
    prob1 = LinearProblem(A1, b1; u0 = x1)
    A2 = A / 2
    @test isposdef(A2)
    b2 = rand(rng, n)
    prob2 = LinearProblem(A2, b2)
    # HYPREArrays
    prob3 = LinearProblem(HYPREMatrix(A1), HYPREVector(b1); u0 = HYPREVector(x1))
    prob4 = LinearProblem(HYPREMatrix(A2), HYPREVector(b2))
    return prob1, prob2, prob3, prob4
end

function test_interface(alg; kw...)
    prob1, prob2, prob3, prob4 = generate_probs(alg; Pl = get(kw, :Pl, LinearAlgebra.I))

    atol = 1.0e-6
    rtol = 1.0e-6
    cache_kwargs = (; abstol = atol, reltol = rtol, maxiters = 50)
    cache_kwargs = merge(cache_kwargs, kw)

    # prob1, prob3 with initial guess, prob2, prob4 without
    for prob in (prob1, prob2, prob3, prob4)
        A, b = to_array(prob.A), to_array(prob.b)

        # Solve prob directly (without cache)
        y = solve(prob, alg; cache_kwargs...)
        @test A * to_array(y.u) ≈ b atol = atol rtol = rtol
        @test y.iters > 0
        @test y.resid < rtol

        # Solve with cache
        cache = SciMLBase.init(prob, alg; cache_kwargs...)
        @test cache.isfresh == cache.cacheval.isfresh_A ==
            cache.cacheval.isfresh_b == cache.cacheval.isfresh_u == true
        y = solve!(cache)
        cache = y.cache
        @test cache.isfresh == cache.cacheval.isfresh_A ==
            cache.cacheval.isfresh_b == cache.cacheval.isfresh_u == false
        @test A * to_array(y.u) ≈ b atol = atol rtol = rtol

        # Update A
        cache.A = A
        @test cache.isfresh == cache.cacheval.isfresh_A == true
        @test cache.cacheval.isfresh_b == cache.cacheval.isfresh_u == false
        y = solve!(cache; cache_kwargs...)
        cache = y.cache
        @test cache.isfresh == cache.cacheval.isfresh_A ==
            cache.cacheval.isfresh_b == cache.cacheval.isfresh_u == false
        @test A * to_array(y.u) ≈ b atol = atol rtol = rtol

        # Update b
        b2 = 2 * to_array(b)
        if b isa HYPREVector
            b2 = HYPREVector(b2)
        end
        cache.b = b2
        @test cache.cacheval.isfresh_b
        @test cache.cacheval.isfresh_A == cache.cacheval.isfresh_u == false
        y = solve!(cache; cache_kwargs...)
        cache = y.cache
        @test cache.isfresh == cache.cacheval.isfresh_A ==
            cache.cacheval.isfresh_b == cache.cacheval.isfresh_u == false
        @test A * to_array(y.u) ≈ to_array(b2) atol = atol rtol = rtol
    end
    return
end

function test_retcode_failure()
    prob = failure_prob()
    sol = solve(
        prob, HYPREAlgorithm(HYPRE.PCG);
        abstol = 1.0e-12, reltol = 1.0e-12, maxiters = 1
    )
    @test sol.retcode == SciMLBase.ReturnCode.MaxIters
    @test sol.iters == 1
    return @test sol.resid > sol.cache.reltol
end

const comm = MPI.COMM_WORLD

# HYPRE.BiCGSTAB
test_interface(HYPREAlgorithm(HYPRE.BiCGSTAB))
test_interface(HYPREAlgorithm(HYPRE.BiCGSTAB), Pl = HYPRE.BoomerAMG)
test_interface(HYPREAlgorithm(HYPRE.BiCGSTAB(comm)))
test_interface(HYPREAlgorithm(HYPRE.BiCGSTAB(comm)), Pl = HYPRE.BoomerAMG())
# HYPRE.BoomerAMG
test_interface(HYPREAlgorithm(HYPRE.BoomerAMG))
test_interface(HYPREAlgorithm(HYPRE.BoomerAMG()))
# HYPRE.FlexGMRES
test_interface(HYPREAlgorithm(HYPRE.FlexGMRES))
test_interface(HYPREAlgorithm(HYPRE.FlexGMRES), Pl = HYPRE.BoomerAMG)
test_interface(HYPREAlgorithm(HYPRE.FlexGMRES(comm)))
test_interface(HYPREAlgorithm(HYPRE.FlexGMRES(comm)), Pl = HYPRE.BoomerAMG())
# HYPRE.GMRES
test_interface(HYPREAlgorithm(HYPRE.GMRES))
test_interface(HYPREAlgorithm(HYPRE.GMRES), Pl = HYPRE.BoomerAMG)
test_interface(HYPREAlgorithm(HYPRE.GMRES(comm)), Pl = HYPRE.BoomerAMG())
# HYPRE.Hybrid
test_interface(HYPREAlgorithm(HYPRE.Hybrid))
test_interface(HYPREAlgorithm(HYPRE.Hybrid), Pl = HYPRE.BoomerAMG)
test_interface(HYPREAlgorithm(HYPRE.Hybrid()))
test_interface(HYPREAlgorithm(HYPRE.Hybrid()), Pl = HYPRE.BoomerAMG())
# HYPRE.ILU
test_interface(HYPREAlgorithm(HYPRE.ILU))
test_interface(HYPREAlgorithm(HYPRE.ILU), Pl = HYPRE.BoomerAMG)
test_interface(HYPREAlgorithm(HYPRE.ILU()))
test_interface(HYPREAlgorithm(HYPRE.ILU()), Pl = HYPRE.BoomerAMG)
# HYPRE.ParaSails
# Use instantiated ParaSails preconditioners with explicit settings and a deterministic
# test matrix. Avoid forcing the SPD-specific ParaSails mode since that path depends on
# LAPACK symbols being available through HYPRE on every CI image.
test_interface(HYPREAlgorithm(HYPRE.PCG), Pl = HYPRE.ParaSails(; Params = (0.1, 1), Filter = 0.05))
test_interface(HYPREAlgorithm(HYPRE.PCG()), Pl = HYPRE.ParaSails(; Params = (0.1, 1), Filter = 0.05))
# HYPRE.PCG
test_interface(HYPREAlgorithm(HYPRE.PCG))
test_interface(HYPREAlgorithm(HYPRE.PCG), Pl = HYPRE.BoomerAMG)
test_interface(HYPREAlgorithm(HYPRE.PCG(comm)))
test_interface(HYPREAlgorithm(HYPRE.PCG(comm)), Pl = HYPRE.BoomerAMG())
test_retcode_failure()

# Test MPI execution
mpitestfile = joinpath(@__DIR__, "hypretests_mpi.jl")
r = run(ignorestatus(`$(mpiexec()) -n 2 $(Base.julia_cmd()) $(mpitestfile)`))
@test r.exitcode == 0
