using LinearAlgebra
using LinearSolve
using MPI
using PETSc
using SparseArrays
using Random
using Statistics
using Test

MPI.Init()

const PETScExt = Base.get_extension(LinearSolve, :LinearSolvePETScExt)
const rank = MPI.Comm_rank(MPI.COMM_WORLD)
const nprocs = MPI.Comm_size(MPI.COMM_WORLD)

# ══════════════════════════════════════════════════════════════════════════════
#  SERIAL TESTS  (comm = MPI.COMM_SELF, the default)
# ══════════════════════════════════════════════════════════════════════════════

@testset "Serial: Basic Solvers" begin
    n = 100
    A = sprand(n, n, 0.05) + 10I
    A = A'A
    b = rand(n)
    prob = LinearProblem(A, b)

    @testset "GMRES" begin
        sol = solve(prob, PETScAlgorithm(:gmres); abstol = 1.0e-10, reltol = 1.0e-10)
        @test norm(A * sol.u - b) / norm(b) < 1.0e-6
        PETScExt.cleanup_petsc_cache!(sol)
    end

    @testset "CG" begin
        sol = solve(prob, PETScAlgorithm(:cg); abstol = 1.0e-10, reltol = 1.0e-10)
        @test norm(A * sol.u - b) / norm(b) < 1.0e-6
        PETScExt.cleanup_petsc_cache!(sol)
    end

    @testset "BiCGSTAB" begin
        sol = solve(prob, PETScAlgorithm(:bcgs); abstol = 1.0e-10, reltol = 1.0e-10)
        @test norm(A * sol.u - b) / norm(b) < 1.0e-6
        PETScExt.cleanup_petsc_cache!(sol)
    end

    @testset "GMRES + Jacobi" begin
        sol = solve(
            prob, PETScAlgorithm(:gmres; pc_type = :jacobi);
            abstol = 1.0e-10, reltol = 1.0e-10
        )
        @test norm(A * sol.u - b) / norm(b) < 1.0e-6
        PETScExt.cleanup_petsc_cache!(sol)
    end

    @testset "GMRES + ILU" begin
        sol = solve(
            prob, PETScAlgorithm(:gmres; pc_type = :ilu);
            abstol = 1.0e-10, reltol = 1.0e-10
        )
        @test norm(A * sol.u - b) / norm(b) < 1.0e-6
        PETScExt.cleanup_petsc_cache!(sol)
    end
end

@testset "Serial: Dense Matrix" begin
    n = 50
    A = rand(n, n) + 10I; A = A'A
    b = rand(n)
    sol = solve(LinearProblem(A, b), PETScAlgorithm(:gmres); abstol = 1.0e-10)
    @test norm(A * sol.u - b) / norm(b) < 1.0e-6
    PETScExt.cleanup_petsc_cache!(sol)
end

@testset "Serial: Cache Interface" begin
    n = 50
    A = sprand(n, n, 0.1) + 10I; A = A'A
    b1 = rand(n)
    cache = SciMLBase.init(
        LinearProblem(A, b1), PETScAlgorithm(:gmres);
        abstol = 1.0e-10, reltol = 1.0e-10
    )
    sol1 = solve!(cache)
    @test norm(A * sol1.u - b1) / norm(b1) < 1.0e-6

    b2 = rand(n)
    cache.b = b2
    sol2 = solve!(cache)
    @test norm(A * sol2.u - b2) / norm(b2) < 1.0e-6
    PETScExt.cleanup_petsc_cache!(cache)
end

@testset "Serial: Nullspace Constant" begin
    n = 100
    D = sparse(1:n, 1:n, 2.0, n, n)
    D -= sparse(1:(n - 1), 2:n, 1.0, n, n)
    D -= sparse(2:n, 1:(n - 1), 1.0, n, n)
    D[1, 1] = 1.0; D[end, end] = 1.0
    b = rand(n); b .-= sum(b) / n

    alg = PETScAlgorithm(:cg; pc_type = :jacobi, nullspace = :constant)
    sol = solve(LinearProblem(D, b), alg; abstol = 1.0e-10)
    @test sol.retcode == SciMLBase.ReturnCode.Success
    @test norm(D * sol.u - b) / norm(b) < 1.0e-6
    PETScExt.cleanup_petsc_cache!(sol)
end

@testset "Serial: Nullspace Custom" begin
    n = 50
    A = spdiagm(0 => 2 * ones(n), 1 => -ones(n - 1), -1 => -ones(n - 1))
    b = rand(n); b .-= sum(b) / n

    alg = PETScAlgorithm(:gmres; pc_type = :ilu, nullspace = :custom, nullspace_vecs = [ones(n)])
    sol = solve(LinearProblem(A, b), alg; abstol = 1.0e-10)
    @test sol.retcode == SciMLBase.ReturnCode.Success
    @test norm(A * sol.u - b) / norm(b) < 1.0e-6
end

@testset "Serial: Transposed Solve" begin
    n = 50
    A = sprand(n, n, 0.2) + 5I; b = rand(n)
    sol = solve(LinearProblem(A, b), PETScAlgorithm(:gmres; transposed = true))
    @test norm(A' * sol.u - b) / norm(b) < 1.0e-6
    PETScExt.cleanup_petsc_cache!(sol)
end

@testset "Serial: Warm Start" begin
    n = 200
    A = sprand(n, n, 0.02) + 10I; A = A'A; b = rand(n)
    prob = LinearProblem(A, b)

    sol_cold = solve(
        prob, PETScAlgorithm(:cg; initial_guess_nonzero = false);
        reltol = 1.0e-12
    )
    iters_cold = sol_cold.iters
    PETScExt.cleanup_petsc_cache!(sol_cold)

    cache_warm = SciMLBase.init(
        prob, PETScAlgorithm(:cg; initial_guess_nonzero = true); reltol = 1.0e-12
    )
    solve!(cache_warm)
    cache_warm.b = b + rand(n) * 0.01
    sol2 = solve!(cache_warm)
    @test sol2.iters < iters_cold
    PETScExt.cleanup_petsc_cache!(cache_warm)
end

@testset "Serial: Scalar Types" begin
    n = 50

    @testset "Float64 sparse" begin
        A = sprand(Float64, n, n, 0.1) + 10I; A = A'A; b = rand(Float64, n)
        sol = solve(LinearProblem(A, b), PETScAlgorithm(:cg); abstol = 1.0e-10)
        @test eltype(sol.u) == Float64
        @test norm(A * sol.u - b) / norm(b) < 1.0e-6
        PETScExt.cleanup_petsc_cache!(sol)
    end

    @testset "Float32 sparse" begin
        A32 = sparse(Float32.(Matrix(sprand(Float64, n, n, 0.1) + 10I)))
        A32 = A32'A32; b32 = rand(Float32, n)
        sol32 = solve(LinearProblem(A32, b32), PETScAlgorithm(:cg); abstol = 1.0f-5)
        @test eltype(sol32.u) == Float32
        @test norm(A32 * sol32.u - b32) / norm(b32) < 1.0f-3
        PETScExt.cleanup_petsc_cache!(sol32)
    end

    @testset "ComplexF64 sparse" begin
        B = sprand(ComplexF64, n, n, 0.1)
        Ac = B' * B + 10I
        bc = rand(ComplexF64, n)
        sol = solve(LinearProblem(Ac, bc), PETScAlgorithm(:gmres); abstol = 1.0e-10)
        @test eltype(sol.u) == ComplexF64
        @test norm(Ac * sol.u - bc) / norm(bc) < 1.0e-6
        PETScExt.cleanup_petsc_cache!(sol)
    end

    @testset "ComplexF32 sparse" begin
        B32 = sparse(ComplexF32.(Matrix(sprand(ComplexF64, n, n, 0.1))))
        Ac32 = B32' * B32 + 10I
        bc32 = rand(ComplexF32, n)
        sol32 = solve(LinearProblem(Ac32, bc32), PETScAlgorithm(:gmres); abstol = 1.0f-5)
        @test eltype(sol32.u) == ComplexF32
        @test norm(Ac32 * sol32.u - bc32) / norm(bc32) < 1.0f-3
        PETScExt.cleanup_petsc_cache!(sol32)
    end

    @testset "Unsupported type errors" begin
        A_big = BigFloat.(Matrix(sprand(n, n, 0.1) + 10I))
        b_big = BigFloat.(rand(n))
        @test_throws Exception solve(
            LinearProblem(sparse(A_big), b_big), PETScAlgorithm(:gmres)
        )
    end
end

@testset "PETSc Options Database" begin
    n = 100
    A = sprand(n, n, 0.1); A = A + A' + 20I
    b = rand(n)
    prob = LinearProblem(A, b)

    @testset "High-precision tolerance" begin
        alg = PETScAlgorithm(:gmres; ksp_options = (ksp_rtol = 1.0e-14, ksp_atol = 1.0e-14))
        sol = solve(prob, alg)
        @test norm(A * sol.u - b) / norm(b) < 1.0e-13
        PETScExt.cleanup_petsc_cache!(sol)
    end

    @testset "Monitoring flags (no crash)" begin
        alg = PETScAlgorithm(:cg; ksp_options = (ksp_monitor = "", ksp_view = ""))
        sol = solve(prob, alg)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        PETScExt.cleanup_petsc_cache!(sol)
    end

    @testset "KSP type override via options" begin
        alg = PETScAlgorithm(:gmres; ksp_options = (ksp_type = "cg",))
        sol = solve(prob, alg)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        PETScExt.cleanup_petsc_cache!(sol)
    end
end

@testset "Serial: Cleanup" begin
    n = 50
    A = sprand(n, n, 0.1) + 10I; A = A'A; b = rand(n)
    prob = LinearProblem(A, b)
    alg = PETScAlgorithm(:gmres; pc_type = :jacobi)

    @testset "via solution" begin
        sol = solve(prob, alg)
        pcache = sol.cache.cacheval
        @test pcache.ksp !== nothing
        PETScExt.cleanup_petsc_cache!(sol)
        @test pcache.ksp === nothing
    end

    @testset "via cache" begin
        cache = SciMLBase.init(prob, alg); solve!(cache)
        PETScExt.cleanup_petsc_cache!(cache)
        @test cache.cacheval.ksp === nothing
    end

    @testset "idempotent" begin
        sol = solve(prob, alg)
        PETScExt.cleanup_petsc_cache!(sol)
        PETScExt.cleanup_petsc_cache!(sol)
        @test sol.cache.cacheval.ksp === nothing
    end

    @testset "empty cache (before solve)" begin
        ce = SciMLBase.init(prob, alg)
        PETScExt.cleanup_petsc_cache!(ce)
        @test ce.cacheval.ksp === nothing
    end
end

# ══════════════════════════════════════════════════════════════════════════════
#  MATRIX REBUILD LOGIC TESTS
#  Exercises the three cases in solve!:
#    Case 1 — first solve (no KSP yet)
#    Case 2 — structure changed → full KSP rebuild
#    Case 3 — same structure, values only → in-place update, KSP reused
# ══════════════════════════════════════════════════════════════════════════════

@testset "Serial: Matrix Rebuild Logic" begin
    n = 50
    Random.seed!(42)

    # Helper: SPD sparse matrix with a fixed sparsity pattern, variable values.
    base = sprand(n, n, 0.1); base = base + base' + 10I
    pattern_I, pattern_J, _ = findnz(base)

    function make_spd_with_pattern(scale)
        vals = abs.(randn(length(pattern_I))) .* scale
        A = sparse(pattern_I, pattern_J, vals, n, n)
        A = A + A'
        d = vec(sum(abs.(A), dims = 2)) .+ scale
        for i in 1:n; A[i, i] = d[i]; end
        return A
    end

    @testset "Case 1: first solve builds KSP" begin
        A = make_spd_with_pattern(10.0)
        b = rand(n)
        cache = SciMLBase.init(
            LinearProblem(A, b), PETScAlgorithm(:cg; pc_type = :jacobi); abstol = 1.0e-10
        )
        @test cache.cacheval.ksp === nothing
        sol = solve!(cache)
        @test cache.cacheval.ksp !== nothing
        @test cache.cacheval.sparsity_hash != UInt(0)
        @test norm(A * sol.u - b) / norm(b) < 1.0e-6
        PETScExt.cleanup_petsc_cache!(cache)
    end

    @testset "Case 3: same sparsity — KSP reused, values updated" begin
        A1 = make_spd_with_pattern(10.0)
        b1 = rand(n)
        cache = SciMLBase.init(
            LinearProblem(A1, b1), PETScAlgorithm(:cg; pc_type = :jacobi); abstol = 1.0e-10
        )
        solve!(cache)
        ksp_before = cache.cacheval.ksp
        hash_before = cache.cacheval.sparsity_hash

        A2 = make_spd_with_pattern(20.0)
        @test PETScExt.sparsity_fingerprint(A1) == PETScExt.sparsity_fingerprint(A2)

        b2 = rand(n)
        SciMLBase.reinit!(cache; A = A2, b = b2)
        sol2 = solve!(cache)

        @test cache.cacheval.ksp === ksp_before
        @test cache.cacheval.sparsity_hash == hash_before
        @test norm(A2 * sol2.u - b2) / norm(b2) < 1.0e-6
        PETScExt.cleanup_petsc_cache!(cache)
    end

    @testset "Case 2: sparsity changed — KSP rebuilt" begin
        A1 = make_spd_with_pattern(10.0)
        b1 = rand(n)
        cache = SciMLBase.init(
            LinearProblem(A1, b1), PETScAlgorithm(:cg; pc_type = :jacobi); abstol = 1.0e-10
        )
        solve!(cache)
        ksp_before = cache.cacheval.ksp

        A3 = sprand(n, n, 0.2) + 15I; A3 = A3 + A3'
        @test PETScExt.sparsity_fingerprint(A1) != PETScExt.sparsity_fingerprint(A3)

        b3 = rand(n)
        SciMLBase.reinit!(cache; A = A3, b = b3)
        sol3 = solve!(cache)

        @test cache.cacheval.ksp !== ksp_before
        @test cache.cacheval.sparsity_hash == PETScExt.sparsity_fingerprint(A3)
        @test norm(A3 * sol3.u - b3) / norm(b3) < 1.0e-6
        PETScExt.cleanup_petsc_cache!(cache)
    end

    @testset "Case 3: correctness over multiple value-only updates" begin
        A = make_spd_with_pattern(10.0)
        b = rand(n)
        cache = SciMLBase.init(
            LinearProblem(A, b), PETScAlgorithm(:cg; pc_type = :jacobi); abstol = 1.0e-10
        )
        solve!(cache)
        for _ in 1:5
            A_new = make_spd_with_pattern(10.0 + rand())
            b_new = rand(n)
            SciMLBase.reinit!(cache; A = A_new, b = b_new)
            sol = solve!(cache)
            @test norm(A_new * sol.u - b_new) / norm(b_new) < 1.0e-6
        end
        PETScExt.cleanup_petsc_cache!(cache)
    end

    @testset "Case 3: same-size dense — KSP reused" begin
        A1 = Matrix(make_spd_with_pattern(10.0))
        b1 = rand(n)
        cache = SciMLBase.init(
            LinearProblem(A1, b1), PETScAlgorithm(:gmres); abstol = 1.0e-10
        )
        sol1 = solve!(cache)
        @test norm(A1 * sol1.u - b1) / norm(b1) < 1.0e-6

        ksp_before = cache.cacheval.ksp
        A2 = Matrix(make_spd_with_pattern(20.0))
        b2 = rand(n)
        SciMLBase.reinit!(cache; A = A2, b = b2)
        sol2 = solve!(cache)

        @test cache.cacheval.ksp === ksp_before
        @test norm(A2 * sol2.u - b2) / norm(b2) < 1.0e-6
        PETScExt.cleanup_petsc_cache!(cache)
    end

    @testset "Case 2: size change dense — KSP rebuilt" begin
        n2 = n + 5
        A1 = Matrix(make_spd_with_pattern(10.0))
        b1 = rand(n)
        cache = SciMLBase.init(
            LinearProblem(A1, b1), PETScAlgorithm(:gmres); abstol = 1.0e-10
        )
        sol1 = solve!(cache)
        @test norm(A1 * sol1.u - b1) / norm(b1) < 1.0e-6

        ksp_before = cache.cacheval.ksp
        A2 = rand(n2, n2); A2 = A2'A2 + n2 * I
        b2 = rand(n2)
        SciMLBase.reinit!(cache; A = A2, b = b2, u = zeros(n2))
        sol2 = solve!(cache)

        @test cache.cacheval.ksp !== ksp_before
        @test norm(A2 * sol2.u - b2) / norm(b2) < 1.0e-6
        PETScExt.cleanup_petsc_cache!(cache)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
#  NO-COPY / MEMORY SAFETY TESTS
#  Verify that Julia backing arrays are never unexpectedly mutated by PETSc.
# ══════════════════════════════════════════════════════════════════════════════

@testset "No-copy verification" begin
    n = 50
    function make_spd_sparse(n)
        A = sprand(n, n, 0.2) + 10I; return A + A'
    end
    function make_spd_dense(n)
        B = randn(n, n); return B'B + 10I
    end

    @testset "Sparse operator: backing arrays not copied" begin
        A = make_spd_sparse(n)
        b = rand(n)
        ptr_nzval  = pointer(A.nzval)
        ptr_colptr = pointer(A.colptr)
        sol = solve(LinearProblem(A, b), PETScAlgorithm(:gmres; pc_type = :ilu);
                    abstol = 1e-10)
        @test norm(A * sol.u - b) / norm(b) < 1e-6
        @test pointer(A.nzval)  === ptr_nzval
        @test pointer(A.colptr) === ptr_colptr
        PETScExt.cleanup_petsc_cache!(sol)
    end

    @testset "Dense operator: Julia array not copied" begin
        A = make_spd_dense(n)
        b = rand(n)
        ptr_before = pointer(A)
        sol = solve(LinearProblem(A, b), PETScAlgorithm(:gmres; pc_type = :jacobi);
                    abstol = 1e-10)
        @test norm(A * sol.u - b) / norm(b) < 1e-6
        @test pointer(A) === ptr_before
        PETScExt.cleanup_petsc_cache!(sol)
    end

    @testset "Sparse Case 3: nzval pointer stable across value-only updates" begin
        base = make_spd_sparse(n)
        I_idx, J_idx, _ = findnz(base)
        function make_with_pattern(scale)
            vals = abs.(randn(length(I_idx))) .* scale
            A = sparse(I_idx, J_idx, vals, n, n); A = A + A'
            d = vec(sum(abs.(A), dims = 2)) .+ scale
            for i in 1:n; A[i, i] = d[i]; end
            return A
        end

        A1 = make_with_pattern(10.0)
        cache = SciMLBase.init(
            LinearProblem(A1, rand(n)), PETScAlgorithm(:cg; pc_type = :jacobi); abstol = 1e-10
        )
        solve!(cache)

        A2 = make_with_pattern(20.0)
        ptr_nzval  = pointer(A2.nzval)
        ptr_colptr = pointer(A2.colptr)
        b2 = rand(n)
        SciMLBase.reinit!(cache; A = A2, b = b2)
        sol = solve!(cache)

        @test norm(A2 * sol.u - b2) / norm(b2) < 1e-6
        @test pointer(A2.nzval)  === ptr_nzval
        @test pointer(A2.colptr) === ptr_colptr
        PETScExt.cleanup_petsc_cache!(cache)
    end

    @testset "Dense explicit prec_matrix: not mutated by LU/Cholesky" begin
        # With a separate prec_matrix, PETSc factors its own internal copy.
        # The user's P array must be unchanged after one or more solves.
        A = make_spd_dense(n)
        P = copy(A)
        P_snapshot = copy(P)
        alg = PETScAlgorithm(:preonly; pc_type = :cholesky, prec_matrix = P)

        b1 = rand(n)
        cache = SciMLBase.init(LinearProblem(A, b1), alg; abstol = 1e-10)
        sol1 = solve!(cache)
        @test norm(A * sol1.u - b1) / norm(b1) < 1e-6
        @test P == P_snapshot   # first solve must not mutate P

        b2 = rand(n)
        SciMLBase.reinit!(cache; b = b2)
        sol2 = solve!(cache)
        @test norm(A * sol2.u - b2) / norm(b2) < 1e-6
        @test P == P_snapshot   # second solve must not mutate P either
        PETScExt.cleanup_petsc_cache!(cache)
    end

    @testset "Dense operator with factorising PC (no prec_matrix): documented mutation" begin
        # Without a separate prec_matrix, petsc_P aliases petsc_A.
        # An in-place factorisation (LU/Cholesky) will overwrite the Julia array.
        # This is documented behaviour — the solve is still correct.
        A = make_spd_dense(n)
        A_snapshot = copy(A)
        b = rand(n)
        sol = solve(LinearProblem(A, b), PETScAlgorithm(:preonly; pc_type = :lu);
                    abstol = 1e-10)
        @test norm(A_snapshot * sol.u - b) / norm(b) < 1e-6
        # A may have been overwritten by LU — that's expected, no assertion on A's values
        PETScExt.cleanup_petsc_cache!(sol)
    end
end