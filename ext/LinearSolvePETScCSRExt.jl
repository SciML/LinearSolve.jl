module LinearSolvePETScCSRExt

using SparseMatricesCSR: SparseMatrixCSR, getrowptr, getcolval
using PETSc
import PETSc: MPI, LibPETSc
using LinearSolve: PETScAlgorithm, LinearCache, LinearSolve

const PETScExt = Base.get_extension(LinearSolve, :LinearSolvePETScExt)

# ── Sparse trait ─────────────────────────────────────────────────────────────

PETScExt.is_sparse_petsc(::SparseMatrixCSR) = true

# ── Canonical-type conversion ────────────────────────────────────────────────
#
# PETSc's MatCreateSeqAIJWithArrays requires 0-based Int and the correct scalar
# type.  Following GridapPETSc.jl we convert A to SparseMatrixCSR{0,PetscScalar,
# PetscInt} (the "canonical" form) before handing arrays to PETSc.
#
# Cost breakdown:
#   Bi=0, types match → zero allocation; the original CSR is returned as-is.
#   Bi=0, index type differs → type-cast rowptr/colval only (no shift).
#   Bi≠0 (e.g. default Bi=1) → shift every index by -Bi and type-cast if needed.
#
# In all cases nzval is copied so that the user's Julia matrix is never mutated
# by the Case-3 in-place value update.

function _to_petsc_canonical(
        A::SparseMatrixCSR{Bi, Tv, Ti},
        ::Type{PetscInt}, ::Type{PetscScalar}) where {Bi, Tv, Ti, PetscInt, PetscScalar}
    m, n    = size(A)
    src_rp  = getrowptr(A)
    src_cv  = getcolval(A)
    src_nz  = A.nzval

    # ── index arrays ─────────────────────────────────────────────────────────
    if Bi == 0
        if Ti == PetscInt
            # Perfect-match path: borrow the original arrays — zero allocation.
            Tv == PetscScalar && return A
            rp = src_rp
            cv = src_cv
        else
            rp = PetscInt.(src_rp)
            cv = PetscInt.(src_cv)
        end
    else
        rp = PetscInt[v - Bi for v in src_rp]
        cv = PetscInt[v - Bi for v in src_cv]
    end

    # ── value array (always copied to avoid mutating the user's matrix) ──────
    nz = Tv == PetscScalar ? copy(src_nz) : PetscScalar.(src_nz)
    return SparseMatrixCSR{0}(m, n, rp, cv, nz)
end

# ── Matrix construction ───────────────────────────────────────────────────────
#
# Following GridapPETSc.jl:
#   1. Convert A to the canonical SparseMatrixCSR{0,PetscScalar,PetscInt}.
#   2. Pass its arrays directly to MatCreateSeqAIJWithArrays (zero-copy for
#      indices when types already match).
#   3. Store the canonical CSR object — not just the arrays — as the GC anchor
#      so that PETSc's borrowed pointers remain valid for the lifetime of PA.

function PETScExt.to_petsc_mat(petsclib, A::SparseMatrixCSR{Bi}) where {Bi}
    PetscInt    = PETSc.inttype(petsclib)
    PetscScalar = PETSc.scalartype(petsclib)

    canonical = _to_petsc_canonical(A, PetscInt, PetscScalar)
    m, n      = size(canonical)

    mat = LibPETSc.MatCreateSeqAIJWithArrays(
        petsclib, MPI.COMM_SELF,
        PetscInt(m), PetscInt(n),
        getrowptr(canonical), getcolval(canonical), canonical.nzval)

    # Store the whole canonical CSR as GC anchor (GridapPETSc pattern).
    # PETSc.destroy will pop! this automatically when the mat is destroyed.
    PETSc._MATSEQAIJ_WITHARRAYS_STORAGE[mat.ptr] = canonical
    return mat
end

# ── Sparsity pattern storage ─────────────────────────────────────────────────
#
# CSR is already in PETSc's native row-major order — no permutation needed.
# Reuse prev_colptr / prev_rowval to hold rowptr / colval for change detection.

function PETScExt.store_sparse_pattern!(pcache, A::SparseMatrixCSR)
    pcache.sparse_perm    = nothing
    pcache.sparse_scratch = nothing
    pcache.prev_colptr    = getrowptr(A)
    pcache.prev_rowval    = getcolval(A)
end

# ── Sparsity pattern change detection ────────────────────────────────────────

function PETScExt.check_pattern_changed(pcache, A::SparseMatrixCSR)
    pcache.prev_colptr === nothing && return true
    old_rp, old_cv = pcache.prev_colptr, pcache.prev_rowval
    new_rp, new_cv = getrowptr(A), getcolval(A)
    (length(old_rp) != length(new_rp) || length(old_cv) != length(new_cv)) && return true
    return old_rp != new_rp || old_cv != new_cv
end

# ── In-place value update (Case 3) ───────────────────────────────────────────
#
# CSR nzval is already in PETSc's row-major order, so a direct copyto! suffices
# — no scatter/permutation needed.  MatSeqAIJGetArray returns a pointer to the
# buffer that was passed to MatCreateSeqAIJWithArrays (the canonical nzval).

function PETScExt.update_sparse_values!(
        petsclib, PA, pcache, A::SparseMatrixCSR; assemble::Bool = true)
    vals = LibPETSc.MatSeqAIJGetArray(petsclib, PA)
    try
        copyto!(vals, A.nzval)
    finally
        LibPETSc.MatSeqAIJRestoreArray(petsclib, PA, vals)
    end
    assemble && PETSc.assemble!(PA)
    return nothing
end

end
