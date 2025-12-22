module LinearSolveEnzymeExt

using LinearSolve: LinearSolve, SciMLLinearSolveAlgorithm, init, solve!, LinearProblem,
                   LinearCache, AbstractKrylovSubspaceMethod, DefaultLinearSolver,
                   defaultalg_adjoint_eval, solve
using LinearSolve.LinearAlgebra
using EnzymeCore
using EnzymeCore: EnzymeRules
using SparseArrays: AbstractSparseMatrix, AbstractSparseMatrixCSC, SparseMatrixCSC

@inline EnzymeCore.EnzymeRules.inactive_type(::Type{<:LinearSolve.SciMLLinearSolveAlgorithm}) = true

# Helper functions for sparse-safe gradient accumulation
# These avoid broadcast operations that can change sparsity patterns
#
# Key insight: Enzyme.make_zero shares structural arrays (rowval, colptr) between
# primal and shadow sparse matrices. Broadcast operations like `dA .-= z * y'` can
# change the sparsity pattern, corrupting both shadow AND primal. We must operate
# directly on nzval to preserve the sparsity pattern.

using SparseArrays: nonzeros, rowvals, getcolptr, nzrange

"""
    _safe_add!(dst, src)

Add `src` to `dst` in a way that preserves the sparsity pattern of sparse matrices.
For sparse matrices with matching sparsity patterns (as with Enzyme shadows),
this operates directly on the nonzeros array.
"""
function _safe_add!(dst::AbstractSparseMatrixCSC, src::AbstractSparseMatrixCSC)
    nonzeros(dst) .+= nonzeros(src)
    return dst
end

function _safe_add!(dst::AbstractArray, src::AbstractArray)
    dst .+= src
    return dst
end

"""
    _safe_zero!(A)

Zero out `A` in a way that preserves the sparsity pattern of sparse matrices.
For sparse matrices, this operates directly on the nonzeros array.
"""
function _safe_zero!(A::AbstractSparseMatrixCSC)
    fill!(nonzeros(A), zero(eltype(A)))
    return A
end

function _safe_zero!(A::AbstractArray)
    fill!(A, zero(eltype(A)))
    return A
end

"""
    _sparse_outer_sub!(dA, z, y)

Compute `dA .-= z * transpose(y)` in a sparsity-preserving manner.

For sparse matrices, only accumulates gradients into existing non-zero positions.
This is mathematically correct for sparse matrix AD: gradients are only meaningful
at positions where the matrix can be modified.
"""
function _sparse_outer_sub!(dA::SparseMatrixCSC, z::AbstractVector, y::AbstractVector)
    rows = rowvals(dA)
    vals = nonzeros(dA)
    colptr = getcolptr(dA)

    # Non-allocating loop over CSC structure
    # This is efficient and cache-friendly (column-major order)
    @inbounds for col in 1:size(dA, 2)
        y_col = y[col]
        for idx in colptr[col]:(colptr[col + 1] - 1)
            vals[idx] -= z[rows[idx]] * y_col
        end
    end

    return dA
end

# GPU sparse matrices (CuSparseMatrixCSC, ROCSparseMatrixCSC, etc.)
# Use vectorized operations that work on GPU arrays
function _sparse_outer_sub!(dA::AbstractSparseMatrixCSC, z::AbstractVector, y::AbstractVector)
    rows = rowvals(dA)
    vals = nonzeros(dA)
    colptr = getcolptr(dA)
    n_cols = size(dA, 2)

    # Build column indices for each stored value (allocates O(nnz) memory)
    # This is needed for GPU-compatible vectorized operations
    col_indices = _expand_colptr_to_col_indices(rows, colptr, n_cols)

    # Vectorized update - works on GPU via broadcasting
    # vals[i] -= z[rows[i]] * y[col_indices[i]]
    vals .-= z[rows] .* y[col_indices]

    return dA
end

"""
    _expand_colptr_to_col_indices(rows, colptr, n_cols)

Convert CSC column pointer array to per-element column indices.
Returns a vector where element i contains the column index of the i-th stored value.
The output array is allocated on the same device as `rows`.
"""
function _expand_colptr_to_col_indices(rows::Vector, colptr::Vector, n_cols::Integer)
    # CPU path - use efficient loop
    nnz = length(rows)
    col_indices = Vector{eltype(colptr)}(undef, nnz)
    @inbounds for col in 1:n_cols
        for idx in colptr[col]:(colptr[col + 1] - 1)
            col_indices[idx] = col
        end
    end
    return col_indices
end

function _expand_colptr_to_col_indices(
        rows::AbstractVector, colptr::AbstractVector, n_cols::Integer)
    # GPU path - copy colptr to CPU, build indices, copy back
    # This avoids slow scalar indexing on GPU arrays
    colptr_cpu = collect(colptr)
    nnz = length(rows)

    # Build on CPU (fast)
    col_indices_cpu = Vector{eltype(colptr_cpu)}(undef, nnz)
    @inbounds for col in 1:n_cols
        for idx in colptr_cpu[col]:(colptr_cpu[col + 1] - 1)
            col_indices_cpu[idx] = col
        end
    end

    # Copy to GPU (matching rows array type)
    col_indices = similar(rows, eltype(colptr_cpu), nnz)
    copyto!(col_indices, col_indices_cpu)
    return col_indices
end

function _sparse_outer_sub!(dA::AbstractArray, z::AbstractVector, y::AbstractVector)
    dA .-= z * transpose(y)
    return dA
end

function EnzymeRules.forward(config::EnzymeRules.FwdConfigWidth{1},
        func::Const{typeof(LinearSolve.init)}, ::Type{RT}, prob::EnzymeCore.Annotation{LP},
        alg::Const; kwargs...) where {RT, LP <: LinearSolve.LinearProblem}
    @assert !(prob isa Const)
    res = func.val(prob.val, alg.val; kwargs...)
    if RT <: Const
        if EnzymeRules.needs_primal(config)
            return res
        else
            return nothing
        end
    end

    dres = func.val(prob.dval, alg.val; kwargs...)

    if dres.b == res.b
        _safe_zero!(dres.b)
    end
    if dres.A == res.A
        _safe_zero!(dres.A)
    end

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated(res, dres)
    elseif EnzymeRules.needs_shadow(config)
        return dres
    elseif EnzymeRules.needs_primal(config)
        return res
    else
        return nothing
    end
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfigWidth{1}, func::Const{typeof(LinearSolve.solve!)},
        ::Type{RT}, linsolve::EnzymeCore.Annotation{LP};
        kwargs...) where {RT, LP <: LinearSolve.LinearCache}
    @assert !(linsolve isa Const)

    res = func.val(linsolve.val; kwargs...)

    if RT <: Const
        if EnzymeRules.needs_primal(config)
            return res
        else
            return nothing
        end
    end
    if linsolve.val.alg isa LinearSolve.AbstractKrylovSubspaceMethod
        error("Algorithm $(_linsolve.alg) is currently not supported by Enzyme rules on LinearSolve.jl. Please open an issue on LinearSolve.jl detailing which algorithm is missing the adjoint handling")
    end

    res = deepcopy(res)  # Without this copy, the next solve will end up mutating the result

    b = linsolve.val.b
    linsolve.val.b = linsolve.dval.b - linsolve.dval.A * res.u
    dres = func.val(linsolve.val; kwargs...)
    linsolve.val.b = b

    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        return Duplicated(res, dres)
    elseif EnzymeRules.needs_shadow(config)
        return dres
    elseif EnzymeRules.needs_primal(config)
        return res
    else
        return nothing
    end
end

function EnzymeRules.augmented_primal(
        config, func::Const{typeof(LinearSolve.init)},
        ::Type{RT}, prob::EnzymeCore.Annotation{LP}, alg::Const;
        kwargs...) where {RT, LP <: LinearSolve.LinearProblem}
    res = func.val(prob.val, alg.val; kwargs...)
    dres = if EnzymeRules.width(config) == 1
        func.val(prob.dval, alg.val; kwargs...)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            func.val(prob.dval[i], alg.val; kwargs...)
        end
    end
    d_A = if EnzymeRules.width(config) == 1
        dres.A
    else
        (dval.A for dval in dres)
    end
    d_b = if EnzymeRules.width(config) == 1
        dres.b
    else
        (dval.b for dval in dres)
    end

    prob_d_A = if EnzymeRules.width(config) == 1
        prob.dval.A
    else
        (dval.A for dval in prob.dval)
    end
    prob_d_b = if EnzymeRules.width(config) == 1
        prob.dval.b
    else
        (dval.b for dval in prob.dval)
    end

    return EnzymeRules.AugmentedReturn(res, dres, (d_A, d_b, prob_d_A, prob_d_b))
end

function EnzymeRules.reverse(
        config, func::Const{typeof(LinearSolve.init)}, ::Type{RT},
        cache, prob::EnzymeCore.Annotation{LP}, alg::Const;
        kwargs...) where {RT, LP <: LinearSolve.LinearProblem}
    d_A, d_b, prob_d_A, prob_d_b = cache

    if EnzymeRules.width(config) == 1
        if d_A !== prob_d_A
            # Use sparse-safe addition to preserve sparsity pattern
            _safe_add!(prob_d_A, d_A)
            _safe_zero!(d_A)
        end
        if d_b !== prob_d_b
            _safe_add!(prob_d_b, d_b)
            _safe_zero!(d_b)
        end
    else
        for (_prob_d_A, _d_A, _prob_d_b, _d_b) in zip(prob_d_A, d_A, prob_d_b, d_b)
            if _d_A !== _prob_d_A
                _safe_add!(_prob_d_A, _d_A)
                _safe_zero!(_d_A)
            end
            if _d_b !== _prob_d_b
                _safe_add!(_prob_d_b, _d_b)
                _safe_zero!(_d_b)
            end
        end
    end

    return (nothing, nothing)
end

# y=inv(A) B
#   dA −= z y^T
#   dB += z, where  z = inv(A^T) dy
function EnzymeRules.augmented_primal(
        config, func::Const{typeof(LinearSolve.solve!)},
        ::Type{RT}, linsolve::EnzymeCore.Annotation{LP};
        kwargs...) where {RT, LP <: LinearSolve.LinearCache}
    res = func.val(linsolve.val; kwargs...)

    dres = if EnzymeRules.width(config) == 1
        deepcopy(res)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            deepcopy(res)
        end
    end

    if EnzymeRules.width(config) == 1
        dres.u .= 0
    else
        for dr in dres
            dr.u .= 0
        end
    end

    resvals = if EnzymeRules.width(config) == 1
        dres.u
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            dres[i].u
        end
    end

    dAs = if EnzymeRules.width(config) == 1
        (linsolve.dval.A,)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            linsolve.dval[i].A
        end
    end

    dbs = if EnzymeRules.width(config) == 1
        (linsolve.dval.b,)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            linsolve.dval[i].b
        end
    end

    cachesolve = deepcopy(linsolve.val)

    cache = (copy(res.u), resvals, cachesolve, dAs, dbs)

    _res = EnzymeRules.needs_primal(config) ? res : nothing
    _dres = EnzymeRules.needs_shadow(config) ? dres : nothing

    return EnzymeRules.AugmentedReturn(_res, _dres, cache)
end

function EnzymeRules.reverse(config, func::Const{typeof(LinearSolve.solve!)},
        ::Type{RT}, cache, linsolve::EnzymeCore.Annotation{LP};
        kwargs...) where {RT, LP <: LinearSolve.LinearCache}
    y, dys, _linsolve, dAs, dbs = cache

    @assert !(linsolve isa Const)
    @assert !(linsolve isa Active)

    if EnzymeRules.width(config) == 1
        dys = (dys,)
        dlinsolves = (linsolve.dval,)
        if (iszero(linsolve.dval.A) || iszero(linsolve.dval.b)) && !iszero(linsolve.dval.u)
            error("Adjoint case currently not handled. Instead of using `solve!(cache); s1 = copy(cache.u) ...`, use `sol = solve!(cache); s1 = copy(sol.u)`.")
        end
    else
        dlinsolves = linsolve.dval
        if any(x->(iszero(x.A) || iszero(x.b)) && !iszero(x.u), linsolve.dval)
            error("Adjoint case currently not handled. Instead of using `solve!(cache); s1 = copy(cache.u) ...`, use `sol = solve!(cache); s1 = copy(sol.u)`.")
        end
    end

    for (dA, db, dy, dy2) in zip(dAs, dbs, dys, dlinsolves)

        # Add the contribution from direct `linsolve.u` modifications
        dy .+= dy2.u

        z = if _linsolve.cacheval isa Factorization
            _linsolve.cacheval' \ dy
        elseif _linsolve.cacheval isa Tuple && _linsolve.cacheval[1] isa Factorization
            _linsolve.cacheval[1]' \ dy
        elseif _linsolve.alg isa LinearSolve.AbstractKrylovSubspaceMethod
            # Doesn't modify `A`, so it's safe to just reuse it
            invprob = LinearSolve.LinearProblem(transpose(_linsolve.A), dy)
            solve(invprob, _linsolve.alg;
                abstol = _linsolve.abstol,
                reltol = _linsolve.reltol,
                verbose = _linsolve.verbose)
        elseif _linsolve.alg isa LinearSolve.DefaultLinearSolver
            LinearSolve.defaultalg_adjoint_eval(_linsolve, dy)
        else
            error("Algorithm $(_linsolve.alg) is currently not supported by Enzyme rules on LinearSolve.jl. Please open an issue on LinearSolve.jl detailing which algorithm is missing the adjoint handling")
        end

        # Use sparse-safe outer product subtraction to preserve sparsity pattern
        _sparse_outer_sub!(dA, z, y)
        db .+= z
        dy .= eltype(dy)(0)
    end

    return (nothing,)
end

end
