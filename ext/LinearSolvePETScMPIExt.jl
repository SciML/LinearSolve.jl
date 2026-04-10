module LinearSolvePETScMPIExt

# MPI-parallel PETSc extension for LinearSolve.jl.
#
# Each MPI rank runs the same Julia code simultaneously.  When the system
# matrix is a `PSparseMatrix` (from PartitionedArrays.jl) this extension
# intercepts the key hooks exposed by `LinearSolvePETScExt` to build a
# distributed PETSc KSP solve:
#
#   • to_petsc_mat    → MatCreateMPIAIJWithArrays  (owned rows per rank)
#   • ensure_vecs!    → VecCreateMPI               (owned DOF local sizes)
#   • run_ksp!        → VecGetArray I/O + KSPSolve
#   • store/check_sparse_pattern!, update_sparse_values! for PSparseMatrix
#
# Two backends are supported:
#   • MPIArray  — true MPI-parallel solve; `comm` comes from the partition.
#   • DebugArray — single-process fallback (e.g. unit tests with with_debug());
#                 uses MPI.COMM_SELF so PETSc still sees a valid communicator.
#
# After the solve, only the OWNED degrees-of-freedom of the solution PVector
# are updated.  Ghost values remain stale and must be synchronised by the
# caller via `PartitionedArrays.consistent!` if needed.

using PartitionedArrays: PartitionedArrays, PSparseMatrix, PVector,
    MPIArray, DebugArray, partition,
    own_length, ghost_length,
    own_to_local, local_to_global, own_to_global, ghost_to_global

using SparseMatricesCSR: SparseMatrixCSR, getrowptr, getcolval

using PETSc
import PETSc: MPI, LibPETSc

using LinearSolve: PETScAlgorithm, LinearSolve

const PETScExt = Base.get_extension(LinearSolve, :LinearSolvePETScExt)

# ── Local-partition accessor ──────────────────────────────────────────────────
#
# MPIArray stores each rank's local data in a `.item` field (accessed
# independently on every rank during an MPI run).
# DebugArray (single-process simulation) stores all parts in `.items`; with
# a single part we just take the first element.

_local_part(a::MPIArray)   = a.item
_local_part(a::DebugArray) = a.items[1]

# ── Comm resolution ───────────────────────────────────────────────────────────
#
# Extract the MPI communicator from the PSparseMatrix's array partition.
# Returns COMM_SELF for DebugArray (single-process tests).

function PETScExt.get_comm(alg::PETScAlgorithm, A::PSparseMatrix)
    p = partition(A)
    return p isa MPIArray ? p.comm : MPI.COMM_SELF
end

# ── Sparse dispatch trait ─────────────────────────────────────────────────────

PETScExt.is_sparse_petsc(::PSparseMatrix) = true

# ── Local CSR construction ────────────────────────────────────────────────────
#
# Build the three arrays required by MatCreateMPIAIJWithArrays:
#   rowptr  — 0-based row offsets for the owned rows on this rank
#   colind  — 0-based GLOBAL column indices for every NZ in those rows
#   nzval   — scalar values in CSR order
#
# Only the first `m_own` rows of the local matrix are passed to PETSc.  Ghost
# rows (when present) are dropped — PETSc handles off-rank contributions
# through its internal assembly.
#
# The conversion to SparseMatrixCSR{0,PetscScalar,PetscInt} is inlined here
# so that the CSR{0} fast-path (common with GridapFEM matrices) skips the
# intermediate allocation on the colind loop.

function _build_mpi_csr_arrays(
        local_mat, row_idx, col_idx,
        ::Type{PetscInt}, ::Type{PetscScalar}
    ) where {PetscInt, PetscScalar}

    m_own = own_length(row_idx)

    # Convert the local sparse matrix to CSR{0, PetscScalar, PetscInt}.
    # For a SparseMatrixCSR{0,PetscScalar,PetscInt} this is a no-op.
    csr = convert(SparseMatrixCSR{0, PetscScalar, PetscInt}, local_mat)

    # Row pointer slice for owned rows only.
    rowptr = Vector{PetscInt}(getrowptr(csr)[1:(m_own + 1)])
    n_nz   = Int(rowptr[end])

    # Global (0-based) column index translation.
    # local_to_global(col_idx)[j_local_1based] gives the 1-based global column.
    # csr.colval stores 0-based local column indices.
    cols_l2g = local_to_global(col_idx)
    colind = Vector{PetscInt}(undef, n_nz)
    src_cv = getcolval(csr)
    @inbounds for k in 1:n_nz
        colind[k] = PetscInt(cols_l2g[src_cv[k] + 1] - 1)
    end

    nzval = Vector{PetscScalar}(csr.nzval[1:n_nz])

    return rowptr, colind, nzval, m_own, n_nz
end

# ── Matrix construction ───────────────────────────────────────────────────────
#
# Called during KSP build (Case 1 / Case 2).
# If `pcache` is not nothing (system-matrix path), we stash all MPI-specific
# data in pcache.mpi_data so that:
#   • the GC anchor for rowptr/colind/nzval is maintained
#     (PETSc may borrow these arrays internally)
#   • subsequent Case 3 value-only updates can find the nzval buffer
#   • pattern-change detection has a fingerprint to compare against
#
# When `pcache` is nothing (preconditioner-matrix path) we simply build the
# matrix without storing metadata — Case 3 for a PSparseMatrix prec is not
# yet supported and will trigger a full KSP rebuild instead.

function PETScExt.to_petsc_mat(petsclib, A::PSparseMatrix, pcache)
    part = partition(A)
    comm = part isa MPIArray ? part.comm : MPI.COMM_SELF

    PetscInt    = PETSc.inttype(petsclib)
    PetscScalar = PETSc.scalartype(petsclib)

    M, N = size(A)

    local_mat = _local_part(partition(A))
    row_idx   = _local_part(partition(axes(A, 1)))
    col_idx   = _local_part(partition(axes(A, 2)))

    rowptr, colind, nzval, m_own, n_nz =
        _build_mpi_csr_arrays(local_mat, row_idx, col_idx, PetscInt, PetscScalar)

    n_own_cols = own_length(col_idx)

    mat = LibPETSc.MatCreateMPIAIJWithArrays(
        petsclib, comm,
        PetscInt(m_own), PetscInt(n_own_cols),
        PetscInt(M), PetscInt(N),
        rowptr, colind, nzval
    )
    PETSc.assemble!(mat)

    # Store GC anchors + metadata in the cache when available.
    # We keep the rawlocal CSR structure (before global-col translation) for
    # cheap pattern-change detection in check_pattern_changed.
    if pcache !== nothing
        csr_det = convert(SparseMatrixCSR{0, PetscScalar, PetscInt}, _local_part(partition(A)))
        local_rowptr = Vector{Int}(getrowptr(csr_det)[1:(m_own + 1)])
        local_colval = Vector{Int}(getcolval(csr_det)[1:n_nz])

        pcache.mpi_data = (
            rowptr      = rowptr,       # GC anchor (may be borrowed by PETSc)
            colind      = colind,       # GC anchor
            nzval       = nzval,        # buffer reused for Case 3 updates
            m_own       = m_own,
            n_nz        = n_nz,
            global_size = (M, N),
            local_rowptr = local_rowptr, # pre-global-translation, for detection
            local_colval = local_colval,
            comm        = comm,
        )
    end

    return mat
end

# ── Sparsity pattern storage ──────────────────────────────────────────────────
#
# Called immediately after to_petsc_mat in build_ksp!.
# mpi_data is already populated by to_petsc_mat above, so this function
# just needs to copy the rowptr/colval fingerprint fields into prev_colptr /
# prev_rowval for the shared check_pattern_changed fast path.
# (prev_colptr / prev_rowval are checked to be `=== nothing` as the "never
# built" sentinel in the default serial logic; we reuse them here.)

function PETScExt.store_sparse_pattern!(pcache, A::PSparseMatrix)
    data = pcache.mpi_data
    data === nothing && return
    pcache.prev_colptr = data.local_rowptr
    pcache.prev_rowval = data.local_colval
    return nothing
end

# ── Sparsity pattern change detection ─────────────────────────────────────────

function PETScExt.check_pattern_changed(pcache, A::PSparseMatrix)
    # First solve: mpi_data not yet set.
    pcache.mpi_data === nothing && return true

    data = pcache.mpi_data

    # Fast global-size check.
    size(A) != data.global_size && return true

    local_mat = _local_part(partition(A))
    row_idx   = _local_part(partition(axes(A, 1)))
    m_own     = own_length(row_idx)
    m_own != data.m_own && return true

    # Detailed structure check for CSR{0} local matrices.
    if local_mat isa SparseMatrixCSR
        rp = getrowptr(local_mat)
        cv = getcolval(local_mat)
        old_rp = data.local_rowptr
        old_cv = data.local_colval

        # Identity shortcut: same array objects ⇒ same pattern.
        rp === old_rp && cv === old_cv && return false

        length(rp) < m_own + 1 && return true
        Int(rp[m_own + 1]) != data.n_nz && return true

        # Element-wise comparison of owned-row prefix.
        @views rp[1:(m_own + 1)] != old_rp && return true
        n_nz = data.n_nz
        length(cv) < n_nz && return true
        @views cv[1:n_nz] != old_cv && return true
        return false
    else
        # Conservative fallback for non-CSR local matrices: always rebuild.
        # Rebuilding the KSP is safe; only efficiency suffers.
        # On the common path (CSR{0} local mats) the fast check above is used.
        return true
    end
end

# ── In-place value update (Case 3) ───────────────────────────────────────────
#
# Only the system matrix (pcache.petsc_A) is supported — a separate
# PSparseMatrix preconditioner would need its own mpi_data slot, which is not
# yet implemented.
#
# Fast path for CSR{0} local matrices: copy nzval[1:n_nz] directly into the
# pre-allocated buffer, then call MatUpdateMPIAIJWithArray.
# General path: convert to CSR{0} first, then copy.

function PETScExt.update_sparse_values!(
        petsclib, PA, pcache, A::PSparseMatrix; assemble::Bool = true
    )
    if PA !== pcache.petsc_A
        error(
            "Case 3 value updates for a PSparseMatrix preconditioner matrix " *
            "are not yet supported. Use `alg.prec_matrix = nothing` and let " *
            "PETSc handle preconditioning internally."
        )
    end

    data = pcache.mpi_data
    data === nothing && error("mpi_data not initialised — this is an internal bug.")

    PetscScalar = PETSc.scalartype(petsclib)
    PetscInt    = PETSc.inttype(petsclib)

    local_mat = _local_part(partition(A))
    n_nz      = data.n_nz
    nzval_buf = data.nzval

    if local_mat isa SparseMatrixCSR{0}
        # Fast path: local nzval are already in PETSc row-major order for
        # owned rows — one copyto! from the first n_nz elements.
        copyto!(nzval_buf, view(local_mat.nzval, 1:n_nz))
    else
        # General path: convert to CSR{0,PetscScalar,PetscInt} first.
        csr = convert(SparseMatrixCSR{0, PetscScalar, PetscInt}, local_mat)
        copyto!(nzval_buf, view(csr.nzval, 1:n_nz))
    end

    LibPETSc.MatUpdateMPIAIJWithArray(petsclib, PA, nzval_buf)
    assemble && PETSc.assemble!(PA)
    return nothing
end

# ── Distributed vector creation ───────────────────────────────────────────────
#
# Create a pair of VECMPI vectors whose local (owned) size matches the PVector
# partition on this rank.  They are reused across solves (same pattern).
# `vec_n` stores the local owned size rather than the global size so that
# a change in the per-rank distribution is also detected.

function PETScExt.ensure_vecs!(pcache, petsclib, comm, b::PVector)
    row_idx = _local_part(partition(axes(b, 1)))
    n_own   = own_length(row_idx)
    N       = length(b)                # global

    if pcache.vec_n != n_own || pcache.petsc_x === nothing || pcache.petsc_b === nothing
        pcache.petsc_x !== nothing && PETSc.destroy(pcache.petsc_x)
        pcache.petsc_b !== nothing && PETSc.destroy(pcache.petsc_b)
        PetscInt       = PETSc.inttype(petsclib)
        pcache.petsc_x = LibPETSc.VecCreateMPI(
            petsclib, comm, PetscInt(n_own), PetscInt(N))
        pcache.petsc_b = LibPETSc.VecCreateMPI(
            petsclib, comm, PetscInt(n_own), PetscInt(N))
        pcache.vec_n   = n_own
    end
    return pcache.petsc_x, pcache.petsc_b
end

# ── Vector I/O helpers ────────────────────────────────────────────────────────
#
# Copy the owned DOFs of a PVector into / out of a VECMPI PETSc vector via
# VecGetArray / VecRestoreArray.  For standard OwnAndGhostIndices layout,
# own_to_local returns 1:n_own so the inner loop is equivalent to a copyto!
# and the compiler can vectorise it.
#
# Note: only owned DOFs are transferred.  Ghost DOFs of the solution PVector
# remain stale after the solve and must be refreshed by the caller if needed
# (e.g. via PartitionedArrays.consistent!).

function _pvec_to_petscvec!(petsclib, pv::PVector, petsc_v)
    local_vals = _local_part(partition(pv))
    row_idx    = _local_part(partition(axes(pv, 1)))
    own_idxs   = own_to_local(row_idx)

    buf = LibPETSc.VecGetArray(petsclib, petsc_v)
    try
        @inbounds for (i, j) in enumerate(own_idxs)
            buf[i] = local_vals[j]
        end
    finally
        LibPETSc.VecRestoreArray(petsclib, petsc_v, buf)
    end
    return nothing
end

function _petscvec_to_pvec!(petsclib, petsc_v, pv::PVector)
    local_vals = _local_part(partition(pv))
    row_idx    = _local_part(partition(axes(pv, 1)))
    own_idxs   = own_to_local(row_idx)

    buf = LibPETSc.VecGetArray(petsclib, petsc_v)
    try
        @inbounds for (i, j) in enumerate(own_idxs)
            local_vals[j] = buf[i]
        end
    finally
        LibPETSc.VecRestoreArray(petsclib, petsc_v, buf)
    end
    return nothing
end

# ── KSP solve with PVector I/O ────────────────────────────────────────────────
#
# Copies owned DOFs of b into petsc_b (RHS), owned DOFs of u into petsc_x
# (initial guess), runs KSPSolve, then copies the solution from petsc_x back
# into the owned DOFs of u.

function PETScExt.run_ksp!(pcache, petsclib, alg, b::PVector, u::PVector)
    _pvec_to_petscvec!(petsclib, b, pcache.petsc_b)
    _pvec_to_petscvec!(petsclib, u, pcache.petsc_x)  # initial guess

    if alg.transposed
        LibPETSc.KSPSolveTranspose(
            petsclib, pcache.ksp, pcache.petsc_b, pcache.petsc_x)
    else
        LibPETSc.KSPSolve(petsclib, pcache.ksp, pcache.petsc_b, pcache.petsc_x)
    end

    _petscvec_to_pvec!(petsclib, pcache.petsc_x, u)
    return nothing
end

end # module
