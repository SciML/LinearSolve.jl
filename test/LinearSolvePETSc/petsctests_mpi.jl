using LinearAlgebra
using LinearSolve
using MPI
using PartitionedArrays
using PartitionedArrays: PSparseMatrix, PVector, MPIArray, with_mpi,
    uniform_partition, partition, own_to_local, own_length, tuple_of_arrays,
    own_values
using PETSc
using SparseArrays
using SparseMatricesCSR
using Random
using Test
import SciMLBase

Random.seed!(1234)

MPI.Init()

const PETScExt = Base.get_extension(LinearSolve, :LinearSolvePETScExt)

# ══════════════════════════════════════════════════════════════════════════════
#  PARTITIONED ARRAYS (PSparseMatrix / PVector) TESTS
#
#  These tests exercise LinearSolvePETScMPIExt — the extension that handles
#  PSparseMatrix/PVector from PartitionedArrays.jl via MPIArray backends.
#
#  The `with_mpi` callback creates the distributed environment and works
#  correctly after a prior MPI.Init() call.
#
#  Two local-matrix variants are tested:
#    • SplitMatrix (CSC-backed, from psparse COO build) — conservative path
#    • SparseMatrixCSR{0}        (CSR-backed, built directly)  — fast path
#
#  Run with: mpiexecjl -n <P> julia test/petsctests_mpi.jl
# ══════════════════════════════════════════════════════════════════════════════

const PAExt = Base.get_extension(LinearSolve, :LinearSolvePETScMPIExt)

@testset "PSparseMatrix: extension loaded" begin
    @test PAExt !== nothing
end

@testset "SparseMatrixCSC: multi-rank assembly plumbing" begin
    n = 12
    A = spdiagm(-1 => -ones(n - 1), 0 => 4.0 .* ones(n), 1 => -ones(n - 1))
    b = ones(n)
    x_ref = A \ b

    function init_replicated_cache(; prec_matrix = nothing)
        alg = PETScAlgorithm(:gmres; comm = MPI.COMM_WORLD, pc_type = :jacobi, prec_matrix)
        cache = SciMLBase.init(
            LinearProblem(A, b), alg;
            abstol = 1.0e-10, reltol = 1.0e-10
        )
        pcache = cache.cacheval
        petsclib = PETScExt.get_petsclib(eltype(A))
        PETSc.initialized(petsclib) || PETSc.initialize(petsclib)
        pcache.petsclib = petsclib
        pcache.initialized = true
        return cache, alg, petsclib, pcache
    end

    @testset "cache sentinels and ownership range" begin
        cache, alg, petsclib, pcache = init_replicated_cache()
        @test pcache.rstart == -1
        @test pcache.rend == -1
        @test pcache.comm === nothing

        PETScExt.build_ksp!(pcache, petsclib, cache, alg)

        @test pcache.comm == MPI.COMM_WORLD
        @test pcache.petsc_A !== nothing
        @test pcache.ksp !== nothing
        @test pcache.rstart >= 0
        @test pcache.rend >= pcache.rstart

        owned_rows = pcache.rend - pcache.rstart
        @test MPI.Allreduce(owned_rows, +, MPI.COMM_WORLD) == n
        PETScExt.cleanup_petsc_cache!(cache)
    end

    @testset "separate preconditioner follows MPI assembly path" begin
        P = copy(A)
        P[1, 1] += 1
        cache, alg, petsclib, pcache = init_replicated_cache(; prec_matrix = P)

        PETScExt.build_ksp!(pcache, petsclib, cache, alg)

        @test pcache.petsc_P !== nothing
        @test pcache.petsc_P !== pcache.petsc_A
        @test pcache.rstart >= 0
        @test pcache.rend >= pcache.rstart
        PETScExt.cleanup_petsc_cache!(cache)
    end

    @testset "basic 2-rank GMRES solve replicates full solution" begin
        lincache = SciMLBase.init(
            LinearProblem(A, b),
            PETScAlgorithm(:gmres; comm = MPI.COMM_WORLD);
            abstol = 1.0e-10,
            reltol = 1.0e-10
        )
        sol = solve!(lincache)
        pcache = lincache.cacheval

        @test sol.retcode == SciMLBase.ReturnCode.Success
        @test norm(A * sol.u - b) / norm(b) < 1.0e-10
        @test sol.u ≈ x_ref atol = 1.0e-10
        @test pcache.rstart >= 0
        @test pcache.rend >= pcache.rstart
        PETScExt.cleanup_petsc_cache!(lincache)
    end

    @testset "cache reuse with new rhs preserves full-solution contract" begin
        cache = SciMLBase.init(
            LinearProblem(A, b),
            PETScAlgorithm(:gmres; comm = MPI.COMM_WORLD);
            abstol = 1.0e-10,
            reltol = 1.0e-10
        )
        sol1 = solve!(cache)
        @test sol1.retcode == SciMLBase.ReturnCode.Success
        @test sol1.u ≈ x_ref atol = 1.0e-10

        b2 = collect(1.0:n)
        x_ref2 = A \ b2
        cache.b = b2
        sol2 = solve!(cache)

        @test sol2.retcode == SciMLBase.ReturnCode.Success
        @test norm(A * sol2.u - b2) / norm(b2) < 1.0e-10
        @test sol2.u ≈ x_ref2 atol = 1.0e-10
        PETScExt.cleanup_petsc_cache!(cache)
    end

    @testset "maxiters failure sets retcode" begin
        lincache = SciMLBase.init(
            LinearProblem(A, b),
            PETScAlgorithm(:gmres; comm = MPI.COMM_WORLD);
            abstol = 1.0e-16,
            reltol = 1.0e-16,
            maxiters = 1
        )
        sol = solve!(lincache)

        @test sol.retcode == SciMLBase.ReturnCode.Failure
        @test sol.iters == 1
        @test norm(A * sol.u - b) / norm(b) > lincache.reltol
        PETScExt.cleanup_petsc_cache!(lincache)
    end
end

# ── Helper: uniform_partition over MPI.COMM_WORLD ────────────────────────────

function mpi_row_partition(distribute, n)
    parts = distribute(LinearIndices((MPI.Comm_size(MPI.COMM_WORLD),)))
    return uniform_partition(parts, n)
end

# ── Helper: uniform_partition with a fixed number of parts (DebugArray) ──────
# Use nparts=1 so the single "rank" owns all DOFs, exactly mirroring a
# one-process MPI run.  DebugArray's _local_part returns .items[1], which
# with a single part covers the full local system.

function debug_row_partition(distribute, n)
    parts = distribute(LinearIndices((1,)))
    return uniform_partition(parts, n)
end

# ── Helper: build a diagonal PSparseMatrix (SplitMatrix local) ───────────────
# A[i,i] = scale * i,  b[i] = scale * i  →  exact solution u[i] = 1

function build_splitmat_diag(row_partition, scale = 1.0)
    I_v, J_v, V_v = map(row_partition) do rng
        collect(Int, rng), collect(Int, rng), scale .* Float64.(rng)
    end |> tuple_of_arrays
    A = psparse(I_v, J_v, V_v, row_partition, row_partition) |> fetch
    b = PVector(map(rng -> scale .* Float64.(rng), row_partition), row_partition)
    u = PVector(map(rng -> zeros(length(rng)), row_partition), row_partition)
    return A, b, u
end

# ── Helper: build a diagonal PSparseMatrix (SparseMatrixCSR{0} local) ────────

function build_csr_diag(row_partition, scale = 1.0)
    csr_part = map(row_partition) do rng
        m = length(rng)
        sp = sparse(1:m, 1:m, scale .* Float64.(rng), m, m)
        convert(SparseMatrixCSR{0, Float64, Int64}, sp)
    end
    A = PSparseMatrix(csr_part, row_partition, row_partition, true)
    b = PVector(map(rng -> scale .* Float64.(rng), row_partition), row_partition)
    u = PVector(map(rng -> zeros(length(rng)), row_partition), row_partition)
    return A, b, u
end

# ── Helper: assert all owned DOFs of a PVector are ≈ expected_val ────────────

function assert_owned_approx(u::PVector, expected_val; atol = 1.0e-8)
    return map(partition(u), partition(axes(u, 1))) do local_u, row_idx
        for j in own_to_local(row_idx)
            @test local_u[j] ≈ expected_val atol = atol
        end
    end
end

# ── SplitMatrix (COO / psparse) path ─────────────────────────────────────────

@testset "PSparseMatrix SplitMatrix: GMRES basic solve" begin
    n = 8
    with_mpi() do distribute
        rp = mpi_row_partition(distribute, n)
        A, b, u = build_splitmat_diag(rp)
        cache = SciMLBase.init(LinearProblem(A, b; u0 = u), PETScAlgorithm(:gmres); abstol = 1.0e-12)
        sol = solve!(cache)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol.u, 1.0)
        PETScExt.cleanup_petsc_cache!(cache)
    end
end

@testset "PSparseMatrix SplitMatrix: CG basic solve" begin
    n = 8
    with_mpi() do distribute
        rp = mpi_row_partition(distribute, n)
        A, b, u = build_splitmat_diag(rp)
        cache = SciMLBase.init(LinearProblem(A, b; u0 = u), PETScAlgorithm(:cg); abstol = 1.0e-12)
        sol = solve!(cache)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol.u, 1.0)
        PETScExt.cleanup_petsc_cache!(cache)
    end
end

@testset "PSparseMatrix SplitMatrix: GMRES + Jacobi preconditioner" begin
    n = 8
    with_mpi() do distribute
        rp = mpi_row_partition(distribute, n)
        A, b, u = build_splitmat_diag(rp)
        cache = SciMLBase.init(
            LinearProblem(A, b; u0 = u),
            PETScAlgorithm(:gmres; pc_type = :jacobi);
            abstol = 1.0e-12
        )
        sol = solve!(cache)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol.u, 1.0)
        PETScExt.cleanup_petsc_cache!(cache)
    end
end

@testset "PSparseMatrix SplitMatrix: explicit prec_matrix" begin
    n = 8
    with_mpi() do distribute
        rp = mpi_row_partition(distribute, n)
        A, b, u = build_splitmat_diag(rp)
        P, _, _ = build_splitmat_diag(rp)
        alg = PETScAlgorithm(:gmres; pc_type = :jacobi, prec_matrix = P)
        cache = SciMLBase.init(LinearProblem(A, b; u0 = u), alg; abstol = 1.0e-12)
        sol = solve!(cache)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol.u, 1.0)
        PETScExt.cleanup_petsc_cache!(cache)
    end
end

@testset "PSparseMatrix SplitMatrix: tridiagonal GMRES + Jacobi" begin
    # Diagonally dominant tridiagonal: A[i,i]=4, A[i,i±1]=-1
    n = 12
    with_mpi() do distribute
        rp = mpi_row_partition(distribute, n)

        I_v, J_v, V_v = map(rp) do rng
            Is, Js, Vs = Int[], Int[], Float64[]
            for i in rng
                push!(Is, i); push!(Js, i); push!(Vs, 4.0)
                i > 1 && (push!(Is, i); push!(Js, i - 1); push!(Vs, -1.0))
                i < n && (push!(Is, i); push!(Js, i + 1); push!(Vs, -1.0))
            end
            Is, Js, Vs
        end |> tuple_of_arrays
        A = psparse(I_v, J_v, V_v, rp, rp) |> fetch
        b = PVector(map(rng -> ones(length(rng)), rp), rp)
        u = PVector(map(rng -> zeros(length(rng)), rp), rp)

        alg = PETScAlgorithm(:gmres; pc_type = :jacobi)
        cache = SciMLBase.init(LinearProblem(A, b; u0 = u), alg; abstol = 1.0e-10, reltol = 1.0e-10)
        sol = solve!(cache)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        PETScExt.cleanup_petsc_cache!(cache)
    end
end

@testset "PSparseMatrix SplitMatrix: maxiters failure sets retcode" begin
    n = 12
    with_mpi() do distribute
        rp = mpi_row_partition(distribute, n)

        I_v, J_v, V_v = map(rp) do rng
            Is, Js, Vs = Int[], Int[], Float64[]
            for i in rng
                push!(Is, i); push!(Js, i); push!(Vs, 4.0)
                i > 1 && (push!(Is, i); push!(Js, i - 1); push!(Vs, -1.0))
                i < n && (push!(Is, i); push!(Js, i + 1); push!(Vs, -1.0))
            end
            Is, Js, Vs
        end |> tuple_of_arrays
        A = psparse(I_v, J_v, V_v, rp, rp) |> fetch
        b = PVector(map(rng -> ones(length(rng)), rp), rp)
        u = PVector(map(rng -> zeros(length(rng)), rp), rp)

        cache = SciMLBase.init(
            LinearProblem(A, b; u0 = u),
            PETScAlgorithm(:gmres);
            abstol = 1.0e-16,
            reltol = 1.0e-16,
            maxiters = 1
        )
        sol = solve!(cache)
        @test sol.retcode == SciMLBase.ReturnCode.Failure
        @test sol.iters == 1
        PETScExt.cleanup_petsc_cache!(cache)
    end
end

@testset "PSparseMatrix SplitMatrix: residual safety failure" begin
    n = 12
    with_mpi() do distribute
        rp = mpi_row_partition(distribute, n)

        I_v, J_v, V_v = map(rp) do rng
            Is, Js, Vs = Int[], Int[], Float64[]
            for i in rng
                push!(Is, i); push!(Js, i); push!(Vs, 4.0)
                i > 1 && (push!(Is, i); push!(Js, i - 1); push!(Vs, -1.0))
                i < n && (push!(Is, i); push!(Js, i + 1); push!(Vs, -1.0))
            end
            Is, Js, Vs
        end |> tuple_of_arrays
        A = psparse(I_v, J_v, V_v, rp, rp) |> fetch
        b = PVector(map(rng -> ones(length(rng)), rp), rp)
        u = PVector(map(rng -> zeros(length(rng)), rp), rp)

        cache = SciMLBase.init(
            LinearProblem(A, b; u0 = u),
            PETScAlgorithm(:gmres; ksp_options = (ksp_rtol = 0.9,));
            abstol = 0.0,
            reltol = 1.0e-12,
            maxiters = 100
        )
        sol = solve!(cache)
        @test sol.retcode == SciMLBase.ReturnCode.APosterioriSafetyFailure
        PETScExt.cleanup_petsc_cache!(cache)
    end
end

@testset "PSparseMatrix SplitMatrix: reinit! (Case 2 — pattern change)" begin
    # Case 2: sparsity pattern changes between solves → KSP must be rebuilt.
    n = 8
    with_mpi() do distribute
        rp = mpi_row_partition(distribute, n)

        # First solve: diagonal pattern
        A1, b1, u1 = build_splitmat_diag(rp, 1.0)
        cache = SciMLBase.init(
            LinearProblem(A1, b1; u0 = u1),
            PETScAlgorithm(:gmres); abstol = 1.0e-12
        )
        sol1 = solve!(cache)
        @test sol1.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol1.u, 1.0)

        # Second solve: add off-diagonal entries → new pattern
        I_v2, J_v2, V_v2 = map(rp) do rng
            Is, Js, Vs = Int[], Int[], Float64[]
            for i in rng
                push!(Is, i); push!(Js, i); push!(Vs, 5.0)
                i < n && (push!(Is, i); push!(Js, i + 1); push!(Vs, -1.0))
            end
            Is, Js, Vs
        end |> tuple_of_arrays
        A2 = psparse(I_v2, J_v2, V_v2, rp, rp) |> fetch
        b2 = PVector(map(rng -> ones(length(rng)), rp), rp)

        reinit!(cache; A = A2, b = b2, reuse_precs = false)
        sol2 = solve!(cache)
        @test sol2.retcode == SciMLBase.ReturnCode.Success

        PETScExt.cleanup_petsc_cache!(cache)
    end
end

@testset "PSparseMatrix SplitMatrix: reinit! (Case 3 — values only, SplitMatrix conservative)" begin
    # SplitMatrix local type always triggers a conservative KSP rebuild on
    # reinit!.  The solve must still give the correct answer.
    n = 8
    with_mpi() do distribute
        rp = mpi_row_partition(distribute, n)

        A1, b1, u1 = build_splitmat_diag(rp, 1.0)
        cache = SciMLBase.init(
            LinearProblem(A1, b1; u0 = u1),
            PETScAlgorithm(:gmres); abstol = 1.0e-12
        )
        sol1 = solve!(cache)
        @test sol1.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol1.u, 1.0)

        # Scale all values by 3 — same pattern, different values
        A2, b2, _ = build_splitmat_diag(rp, 3.0)
        reinit!(cache; A = A2, b = b2, reuse_precs = false)
        sol2 = solve!(cache)
        @test sol2.retcode == SciMLBase.ReturnCode.Success
        # Exact solution is still all-ones (3i / 3i = 1)
        assert_owned_approx(sol2.u, 1.0)

        PETScExt.cleanup_petsc_cache!(cache)
    end
end

# ── SparseMatrixCSR{0} local-matrix path (fast Case 3) ───────────────────────

@testset "PSparseMatrix CSR{0}: GMRES basic solve" begin
    n = 8
    with_mpi() do distribute
        rp = mpi_row_partition(distribute, n)
        A, b, u = build_csr_diag(rp)
        cache = SciMLBase.init(LinearProblem(A, b; u0 = u), PETScAlgorithm(:gmres); abstol = 1.0e-12)
        sol = solve!(cache)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol.u, 1.0)
        PETScExt.cleanup_petsc_cache!(cache)
    end
end

@testset "PSparseMatrix CSR{0}: GMRES + Jacobi preconditioner" begin
    n = 8
    with_mpi() do distribute
        rp = mpi_row_partition(distribute, n)
        A, b, u = build_csr_diag(rp)
        cache = SciMLBase.init(
            LinearProblem(A, b; u0 = u),
            PETScAlgorithm(:gmres; pc_type = :jacobi);
            abstol = 1.0e-12
        )
        sol = solve!(cache)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol.u, 1.0)
        PETScExt.cleanup_petsc_cache!(cache)
    end
end

@testset "PSparseMatrix CSR{0}: explicit prec_matrix" begin
    n = 8
    with_mpi() do distribute
        rp = mpi_row_partition(distribute, n)
        A, b, u = build_csr_diag(rp)
        P, _, _ = build_csr_diag(rp)
        alg = PETScAlgorithm(:gmres; pc_type = :jacobi, prec_matrix = P)
        cache = SciMLBase.init(LinearProblem(A, b; u0 = u), alg; abstol = 1.0e-12)
        sol = solve!(cache)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol.u, 1.0)
        PETScExt.cleanup_petsc_cache!(cache)
    end
end

@testset "PSparseMatrix CSR{0}: reinit! Case 3 — KSP reused, values updated" begin
    # For CSR{0} local mats the fast pattern-check path is taken, so a
    # same-pattern reinit! should reuse the existing KSP object.
    n = 8
    with_mpi() do distribute
        rp = mpi_row_partition(distribute, n)

        A1, b1, u1 = build_csr_diag(rp, 1.0)
        cache = SciMLBase.init(
            LinearProblem(A1, b1; u0 = u1),
            PETScAlgorithm(:gmres); abstol = 1.0e-12
        )
        sol1 = solve!(cache)
        @test sol1.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol1.u, 1.0)
        ksp_before = cache.cacheval.ksp

        # Same sparsity pattern, scaled values → MatUpdateMPIAIJWithArray path
        A2, b2, _ = build_csr_diag(rp, 2.0)
        reinit!(cache; A = A2, b = b2, reuse_precs = false)
        sol2 = solve!(cache)
        @test sol2.retcode == SciMLBase.ReturnCode.Success
        # Exact solution is still all-ones (2i / 2i = 1)
        assert_owned_approx(sol2.u, 1.0)
        @test cache.cacheval.ksp === ksp_before  # KSP reused, not rebuilt

        PETScExt.cleanup_petsc_cache!(cache)
    end
end

@testset "PSparseMatrix CSR{0}: reinit! Case 2 — pattern change rebuilds KSP" begin
    n = 8
    with_mpi() do distribute
        rp = mpi_row_partition(distribute, n)

        A1, b1, u1 = build_csr_diag(rp, 1.0)
        cache = SciMLBase.init(
            LinearProblem(A1, b1; u0 = u1),
            PETScAlgorithm(:gmres); abstol = 1.0e-12
        )
        solve!(cache)
        ksp_before = cache.cacheval.ksp

        # Add an off-diagonal entry → pattern change
        csr_part2 = map(rp) do rng
            m = length(rng)
            i_idx = vcat(1:m, 1:(m - 1))
            j_idx = vcat(1:m, 2:m)
            v = vcat(5.0 .* ones(m), -1.0 .* ones(m - 1))
            sp = sparse(i_idx, j_idx, v, m, m)
            convert(SparseMatrixCSR{0, Float64, Int64}, sp)
        end
        A2 = PSparseMatrix(csr_part2, rp, rp, true)
        b2 = PVector(map(rng -> ones(length(rng)), rp), rp)

        reinit!(cache; A = A2, b = b2, reuse_precs = false)
        sol2 = solve!(cache)
        @test sol2.retcode == SciMLBase.ReturnCode.Success
        @test cache.cacheval.ksp !== ksp_before  # KSP was rebuilt

        PETScExt.cleanup_petsc_cache!(cache)
    end
end

# ── DebugArray backend (single-process simulation via with_debug) ─────────────
#
# with_debug() passes `DebugArray` as the distribute function, which emulates
# the MPIArray interface sequentially.  The extension routes these through
# MPI.COMM_SELF so PETSc sees a valid single-rank communicator.

@testset "PSparseMatrix DebugArray SplitMatrix: GMRES basic solve" begin
    n = 8
    with_debug() do distribute
        rp = debug_row_partition(distribute, n)
        A, b, u = build_splitmat_diag(rp)
        cache = SciMLBase.init(LinearProblem(A, b; u0 = u), PETScAlgorithm(:gmres); abstol = 1.0e-12)
        sol = solve!(cache)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol.u, 1.0)
        PETScExt.cleanup_petsc_cache!(cache)
    end
end

@testset "PSparseMatrix DebugArray SplitMatrix: CG basic solve" begin
    n = 8
    with_debug() do distribute
        rp = debug_row_partition(distribute, n)
        A, b, u = build_splitmat_diag(rp)
        cache = SciMLBase.init(LinearProblem(A, b; u0 = u), PETScAlgorithm(:cg); abstol = 1.0e-12)
        sol = solve!(cache)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol.u, 1.0)
        PETScExt.cleanup_petsc_cache!(cache)
    end
end

@testset "PSparseMatrix DebugArray SplitMatrix: GMRES + Jacobi preconditioner" begin
    n = 8
    with_debug() do distribute
        rp = debug_row_partition(distribute, n)
        A, b, u = build_splitmat_diag(rp)
        cache = SciMLBase.init(
            LinearProblem(A, b; u0 = u),
            PETScAlgorithm(:gmres; pc_type = :jacobi);
            abstol = 1.0e-12
        )
        sol = solve!(cache)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol.u, 1.0)
        PETScExt.cleanup_petsc_cache!(cache)
    end
end

@testset "PSparseMatrix DebugArray SplitMatrix: explicit prec_matrix" begin
    n = 8
    with_debug() do distribute
        rp = debug_row_partition(distribute, n)
        A, b, u = build_splitmat_diag(rp)
        P, _, _ = build_splitmat_diag(rp)
        alg = PETScAlgorithm(:gmres; pc_type = :jacobi, prec_matrix = P)
        cache = SciMLBase.init(LinearProblem(A, b; u0 = u), alg; abstol = 1.0e-12)
        sol = solve!(cache)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol.u, 1.0)
        PETScExt.cleanup_petsc_cache!(cache)
    end
end

@testset "PSparseMatrix DebugArray SplitMatrix: reinit! (Case 2 — pattern change)" begin
    n = 8
    with_debug() do distribute
        rp = debug_row_partition(distribute, n)

        A1, b1, u1 = build_splitmat_diag(rp, 1.0)
        cache = SciMLBase.init(
            LinearProblem(A1, b1; u0 = u1),
            PETScAlgorithm(:gmres); abstol = 1.0e-12
        )
        sol1 = solve!(cache)
        @test sol1.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol1.u, 1.0)

        # New matrix with off-diagonal entries → pattern change
        I_v2, J_v2, V_v2 = map(rp) do rng
            Is, Js, Vs = Int[], Int[], Float64[]
            for i in rng
                push!(Is, i); push!(Js, i); push!(Vs, 5.0)
                i < n && (push!(Is, i); push!(Js, i + 1); push!(Vs, -1.0))
            end
            Is, Js, Vs
        end |> tuple_of_arrays
        A2 = psparse(I_v2, J_v2, V_v2, rp, rp) |> fetch
        b2 = PVector(map(rng -> ones(length(rng)), rp), rp)

        reinit!(cache; A = A2, b = b2, reuse_precs = false)
        sol2 = solve!(cache)
        @test sol2.retcode == SciMLBase.ReturnCode.Success

        PETScExt.cleanup_petsc_cache!(cache)
    end
end

@testset "PSparseMatrix DebugArray SplitMatrix: reinit! (Case 3 — values only)" begin
    n = 8
    with_debug() do distribute
        rp = debug_row_partition(distribute, n)

        A1, b1, u1 = build_splitmat_diag(rp, 1.0)
        cache = SciMLBase.init(
            LinearProblem(A1, b1; u0 = u1),
            PETScAlgorithm(:gmres); abstol = 1.0e-12
        )
        sol1 = solve!(cache)
        @test sol1.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol1.u, 1.0)

        # Same pattern, scaled values
        A2, b2, _ = build_splitmat_diag(rp, 3.0)
        reinit!(cache; A = A2, b = b2, reuse_precs = false)
        sol2 = solve!(cache)
        @test sol2.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol2.u, 1.0)

        PETScExt.cleanup_petsc_cache!(cache)
    end
end

@testset "PSparseMatrix DebugArray CSR{0}: GMRES basic solve" begin
    n = 8
    with_debug() do distribute
        rp = debug_row_partition(distribute, n)
        A, b, u = build_csr_diag(rp)
        cache = SciMLBase.init(LinearProblem(A, b; u0 = u), PETScAlgorithm(:gmres); abstol = 1.0e-12)
        sol = solve!(cache)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol.u, 1.0)
        PETScExt.cleanup_petsc_cache!(cache)
    end
end

@testset "PSparseMatrix DebugArray CSR{0}: GMRES + Jacobi preconditioner" begin
    n = 8
    with_debug() do distribute
        rp = debug_row_partition(distribute, n)
        A, b, u = build_csr_diag(rp)
        cache = SciMLBase.init(
            LinearProblem(A, b; u0 = u),
            PETScAlgorithm(:gmres; pc_type = :jacobi);
            abstol = 1.0e-12
        )
        sol = solve!(cache)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol.u, 1.0)
        PETScExt.cleanup_petsc_cache!(cache)
    end
end

@testset "PSparseMatrix DebugArray CSR{0}: explicit prec_matrix" begin
    n = 8
    with_debug() do distribute
        rp = debug_row_partition(distribute, n)
        A, b, u = build_csr_diag(rp)
        P, _, _ = build_csr_diag(rp)
        alg = PETScAlgorithm(:gmres; pc_type = :jacobi, prec_matrix = P)
        cache = SciMLBase.init(LinearProblem(A, b; u0 = u), alg; abstol = 1.0e-12)
        sol = solve!(cache)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol.u, 1.0)
        PETScExt.cleanup_petsc_cache!(cache)
    end
end

@testset "PSparseMatrix DebugArray CSR{0}: reinit! Case 3 — KSP reused" begin
    n = 8
    with_debug() do distribute
        rp = debug_row_partition(distribute, n)

        A1, b1, u1 = build_csr_diag(rp, 1.0)
        cache = SciMLBase.init(
            LinearProblem(A1, b1; u0 = u1),
            PETScAlgorithm(:gmres); abstol = 1.0e-12
        )
        sol1 = solve!(cache)
        @test sol1.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol1.u, 1.0)
        ksp_before = cache.cacheval.ksp

        # Same pattern, scaled values → MatUpdateMPIAIJWithArray path
        A2, b2, _ = build_csr_diag(rp, 2.0)
        reinit!(cache; A = A2, b = b2, reuse_precs = false)
        sol2 = solve!(cache)
        @test sol2.retcode == SciMLBase.ReturnCode.Success
        assert_owned_approx(sol2.u, 1.0)
        @test cache.cacheval.ksp === ksp_before  # KSP reused, not rebuilt

        PETScExt.cleanup_petsc_cache!(cache)
    end
end

@testset "PSparseMatrix DebugArray CSR{0}: reinit! Case 2 — pattern change rebuilds KSP" begin
    n = 8
    with_debug() do distribute
        rp = debug_row_partition(distribute, n)

        A1, b1, u1 = build_csr_diag(rp, 1.0)
        cache = SciMLBase.init(
            LinearProblem(A1, b1; u0 = u1),
            PETScAlgorithm(:gmres); abstol = 1.0e-12
        )
        solve!(cache)
        ksp_before = cache.cacheval.ksp

        # Add off-diagonal → pattern change
        csr_part2 = map(rp) do rng
            m = length(rng)
            i_idx = vcat(1:m, 1:(m - 1))
            j_idx = vcat(1:m, 2:m)
            v = vcat(5.0 .* ones(m), -1.0 .* ones(m - 1))
            sp = sparse(i_idx, j_idx, v, m, m)
            convert(SparseMatrixCSR{0, Float64, Int64}, sp)
        end
        A2 = PSparseMatrix(csr_part2, rp, rp, true)
        b2 = PVector(map(rng -> ones(length(rng)), rp), rp)

        reinit!(cache; A = A2, b = b2, reuse_precs = false)
        sol2 = solve!(cache)
        @test sol2.retcode == SciMLBase.ReturnCode.Success
        @test cache.cacheval.ksp !== ksp_before  # KSP was rebuilt

        PETScExt.cleanup_petsc_cache!(cache)
    end
end

@testset "PSparseMatrix: cleanup_petsc_cache!" begin
    n = 4
    with_mpi() do distribute
        rp = mpi_row_partition(distribute, n)
        A, b, u = build_splitmat_diag(rp)
        cache = SciMLBase.init(LinearProblem(A, b; u0 = u), PETScAlgorithm(:gmres); abstol = 1.0e-12)
        sol = solve!(cache)
        @test sol.retcode == SciMLBase.ReturnCode.Success
        # cleanup should not throw
        PETScExt.cleanup_petsc_cache!(cache)
        @test cache.cacheval.ksp === nothing
    end
end
