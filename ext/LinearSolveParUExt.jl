module LinearSolveParUExt

using LinearSolve
using LinearSolve: LinearVerbosity, OperatorAssumptions
using SparseArrays
using SparseArrays: AbstractSparseMatrixCSC, SparseMatrixCSC, nonzeros, rowvals, getcolptr
import SparseArrays.CHOLMOD as CHOLMOD
import LinearAlgebra
import SciMLBase
import SciMLBase: ReturnCode
using SciMLLogging: @SciMLMessage
import ParU_jll

const libparu = ParU_jll.libparu

# ParU return codes  (ParU_Info enum, see ParU.h)
const PARU_SUCCESS = Int32(0)
const PARU_OUT_OF_MEMORY = Int32(-1)
const PARU_INVALID = Int32(-2)
const PARU_SINGULAR = Int32(-3)
const PARU_TOO_LARGE = Int32(-4)

#-------------------------------------------------------------------------------
# ParUCache — holds the symbolic and numeric factorization objects plus the
# ParU control struct.  A finalizer ensures the ParU/CHOLMOD memory is freed.
#-------------------------------------------------------------------------------

mutable struct ParUCache
    sym::Ptr{Cvoid}       # ParU_C_Symbolic  (opaque C pointer, freed by ParU)
    num::Ptr{Cvoid}       # ParU_C_Numeric   (opaque C pointer, freed by ParU)
    control::Ptr{Cvoid}   # ParU_C_Control   (opaque C pointer, freed by ParU)
    # Stored sparsity pattern (1-based) for reuse-symbolic checks
    colptr::Vector{Int64}
    rowval::Vector{Int64}
    nnz::Int
    # Status of the most recent numeric factorization (mirrors PARU_* constants).
    # Checked before attempting to solve so that a null `num` pointer is never
    # passed to ParU_C_Solve_Axb after a failed factorization.
    last_factorize_info::Int32

    function ParUCache()
        @static if !Base.USE_GPL_LIBS
            error(
                "ParUFactorization requires GPL libraries (CHOLMOD/UMFPACK). " *
                    "Rebuild Julia with USE_GPL_LIBS=1"
            )
        end
        ctrl_ref = Ref{Ptr{Cvoid}}(C_NULL)
        info = ccall(
            (:ParU_C_InitControl, libparu), Int32,
            (Ref{Ptr{Cvoid}},),
            ctrl_ref
        )
        if info != PARU_SUCCESS
            error("ParU_C_InitControl failed with code $info")
        end
        cache = new(C_NULL, C_NULL, ctrl_ref[], Int64[], Int64[], 0, PARU_INVALID)
        finalizer(_paru_free_cache!, cache)
        return cache
    end
end

function _paru_free_cache!(cache::ParUCache)
    ctrl = cache.control
    ctrl == C_NULL && return

    if cache.num != C_NULL
        num_ref = Ref(cache.num)
        ccall(
            (:ParU_C_FreeNumeric, libparu), Int32,
            (Ref{Ptr{Cvoid}}, Ptr{Cvoid}), num_ref, ctrl
        )
        cache.num = C_NULL
    end
    if cache.sym != C_NULL
        sym_ref = Ref(cache.sym)
        ccall(
            (:ParU_C_FreeSymbolic, libparu), Int32,
            (Ref{Ptr{Cvoid}}, Ptr{Cvoid}), sym_ref, ctrl
        )
        cache.sym = C_NULL
    end
    ctrl_ref = Ref(ctrl)
    ccall(
        (:ParU_C_FreeControl, libparu), Int32,
        (Ref{Ptr{Cvoid}},), ctrl_ref
    )
    cache.control = C_NULL
    return
end

#-------------------------------------------------------------------------------
# _to_cholmod: convert to a CHOLMOD.Sparse{Float64, Int64} with stype=0
# (fully unsymmetric storage — required by ParU).  stype=0 suppresses the
# automatic symmetry detection that CHOLMOD would otherwise apply.
# The returned object must be GC.@preserve'd while its C pointer is used.
#-------------------------------------------------------------------------------

function _to_cholmod(A::AbstractSparseMatrixCSC)
    csc = SparseMatrixCSC{Float64, Int64}(
        size(A, 1), size(A, 2),
        Int64.(getcolptr(A)),
        Int64.(rowvals(A)),
        Float64.(nonzeros(A))
    )
    # stype=0: store both upper & lower triangles (general unsymmetric).
    # Using Sparse{Tv,Ti}(A, stype) avoids the automatic stype=-1 that
    # CHOLMOD.Sparse(csc) would set for symmetric matrices.
    return CHOLMOD.Sparse{Float64, Int64}(csc, 0)
end

# Return the raw cholmod_sparse pointer via the public Base.pointer API
# (includes a C_NULL guard) cast to Ptr{Cvoid} for ccall.
_paru_ptr(s::CHOLMOD.Sparse) = Ptr{Cvoid}(Base.pointer(s))

#-------------------------------------------------------------------------------
# Helpers for symbolic reuse: store and compare sparsity patterns
#-------------------------------------------------------------------------------

function _pattern_changed(cache::ParUCache, A::AbstractSparseMatrixCSC)
    cp = getcolptr(A)
    rv = rowvals(A)
    length(cache.colptr) == length(cp) && length(cache.rowval) == length(rv) || return true
    @inbounds for i in eachindex(cp)
        cache.colptr[i] == Int64(cp[i]) || return true
    end
    @inbounds for i in eachindex(rv)
        cache.rowval[i] == Int64(rv[i]) || return true
    end
    return false
end

function _store_pattern!(cache::ParUCache, A::AbstractSparseMatrixCSC)
    cache.colptr = Int64.(getcolptr(A))
    cache.rowval = Int64.(rowvals(A))
    return cache.nnz = length(nonzeros(A))
end

#-------------------------------------------------------------------------------
# init_cacheval
#-------------------------------------------------------------------------------

function LinearSolve.init_cacheval(
        alg::LinearSolve.ParUFactorization,
        A::AbstractSparseMatrixCSC{<:AbstractFloat}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return ParUCache()
end

# Generic fallback for type mismatches
function LinearSolve.init_cacheval(
        alg::LinearSolve.ParUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

#-------------------------------------------------------------------------------
# SciMLBase.solve!
#-------------------------------------------------------------------------------

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::LinearSolve.ParUFactorization;
        kwargs...
    )
    A = cache.A
    A = convert(AbstractMatrix, A)

    paru_cache = LinearSolve.@get_cacheval(cache, :ParUFactorization)

    if paru_cache === nothing
        error(
            "ParUFactorization currently only supports sparse Float64 matrices " *
                "(AbstractSparseMatrixCSC{<:AbstractFloat})"
        )
    end

    ctrl = paru_cache.control

    if cache.isfresh
        # Decide whether we need a fresh symbolic analysis
        do_symbolic = (
            !alg.reuse_symbolic ||
                paru_cache.sym == C_NULL ||
                length(nonzeros(A)) != paru_cache.nnz ||
                _pattern_changed(paru_cache, A)
        )

        # Always free the stale numeric factorization first
        if paru_cache.num != C_NULL
            num_ref = Ref(paru_cache.num)
            ccall(
                (:ParU_C_FreeNumeric, libparu), Int32,
                (Ref{Ptr{Cvoid}}, Ptr{Cvoid}), num_ref, ctrl
            )
            paru_cache.num = C_NULL
        end

        # --- Symbolic analysis (only when needed) ---
        if do_symbolic
            if paru_cache.sym != C_NULL
                sym_ref = Ref(paru_cache.sym)
                ccall(
                    (:ParU_C_FreeSymbolic, libparu), Int32,
                    (Ref{Ptr{Cvoid}}, Ptr{Cvoid}), sym_ref, ctrl
                )
                paru_cache.sym = C_NULL
            end

            chol_A = _to_cholmod(A)
            sym_ref = Ref{Ptr{Cvoid}}(C_NULL)
            info = GC.@preserve chol_A ccall(
                (:ParU_C_Analyze, libparu), Int32,
                (Ptr{Cvoid}, Ref{Ptr{Cvoid}}, Ptr{Cvoid}),
                _paru_ptr(chol_A), sym_ref, ctrl
            )
            if info != PARU_SUCCESS
                @SciMLMessage(
                    "ParU symbolic analysis failed (code $info)",
                    cache.verbose, :solver_failure
                )
                paru_cache.last_factorize_info = info
                cache.isfresh = false
                return SciMLBase.build_linear_solution(
                    alg, cache.u, nothing, cache; retcode = ReturnCode.Failure
                )
            end
            paru_cache.sym = sym_ref[]
            _store_pattern!(paru_cache, A)
        end

        # --- Numeric factorization ---
        chol_A_num = _to_cholmod(A)
        num_ref = Ref{Ptr{Cvoid}}(C_NULL)
        info = GC.@preserve chol_A_num ccall(
            (:ParU_C_Factorize, libparu), Int32,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ref{Ptr{Cvoid}}, Ptr{Cvoid}),
            _paru_ptr(chol_A_num), paru_cache.sym, num_ref, ctrl
        )
        if info == PARU_SINGULAR
            @SciMLMessage(
                "ParU factorization: matrix is numerically singular",
                cache.verbose, :solver_failure
            )
            paru_cache.last_factorize_info = PARU_SINGULAR
            cache.isfresh = false
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, cache; retcode = ReturnCode.Infeasible
            )
        elseif info != PARU_SUCCESS
            @SciMLMessage(
                "ParU numeric factorization failed (code $info)",
                cache.verbose, :solver_failure
            )
            paru_cache.last_factorize_info = info
            cache.isfresh = false
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, cache; retcode = ReturnCode.Failure
            )
        end
        paru_cache.num = num_ref[]
        paru_cache.last_factorize_info = PARU_SUCCESS
        cache.isfresh = false
    end

    # --- Solve Ax = b ---
    # Guard against null numeric pointer from a prior failed factorization.
    if paru_cache.last_factorize_info == PARU_SINGULAR
        @SciMLMessage(
            "ParU: cannot solve — prior factorization detected a singular matrix",
            cache.verbose, :solver_failure
        )
        return SciMLBase.build_linear_solution(
            alg, cache.u, nothing, cache; retcode = ReturnCode.Infeasible
        )
    elseif paru_cache.last_factorize_info != PARU_SUCCESS
        @SciMLMessage(
            "ParU: cannot solve — prior factorization failed (code $(paru_cache.last_factorize_info))",
            cache.verbose, :solver_failure
        )
        return SciMLBase.build_linear_solution(
            alg, cache.u, nothing, cache; retcode = ReturnCode.Failure
        )
    end

    # ParU_C_Solve_Axb(Sym, Num, b, x, Control) — separate input/output buffers
    b_vec = Vector{Float64}(cache.b)
    x_vec = similar(b_vec)
    info = ccall(
        (:ParU_C_Solve_Axb, libparu), Int32,
        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Cvoid}),
        paru_cache.sym, paru_cache.num, b_vec, x_vec, ctrl
    )
    if info != PARU_SUCCESS
        @SciMLMessage("ParU solve failed (code $info)", cache.verbose, :solver_failure)
        return SciMLBase.build_linear_solution(
            alg, cache.u, nothing, cache; retcode = ReturnCode.Failure
        )
    end

    cache.u .= x_vec
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache; retcode = ReturnCode.Success
    )
end

LinearSolve.needs_concrete_A(::LinearSolve.ParUFactorization) = true

end # module LinearSolveParUExt
