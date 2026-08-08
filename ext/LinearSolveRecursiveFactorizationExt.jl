module LinearSolveRecursiveFactorizationExt

using LinearSolve: LinearSolve, RFLUFactorization, ButterflyFactorization,
    RF32MixedLUFactorization, LinearVerbosity
using ArrayInterface: ArrayInterface
using LinearAlgebra: LinearAlgebra, UnitLowerTriangular, UpperTriangular, ldiv!, mul!
using RecursiveFactorization: RecursiveFactorization
using TriangularSolve: TriangularSolve
using SciMLBase: SciMLBase, ReturnCode
using SciMLLogging: @SciMLMessage

LinearSolve.userecursivefactorization(A::Union{Nothing, AbstractMatrix}) = true

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::RFLUFactorization{P, T};
        kwargs...
    ) where {P, T}
    A = cache.A
    A = convert(AbstractMatrix, A)
    fact, ipiv = LinearSolve.@get_cacheval(cache, :RFLUFactorization)
    if cache.isfresh
        if length(ipiv) != min(size(A)...)
            ipiv = Vector{LinearAlgebra.BlasInt}(undef, min(size(A)...))
        end
        fact = RecursiveFactorization.lu!(A, ipiv, Val(P), Val(T), check = false)
        cache.cacheval = (fact, ipiv)
        if !LinearAlgebra.issuccess(fact)
            @SciMLMessage("Solver failed", cache.verbose, :solver_failure)
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, nothing; retcode = ReturnCode.Failure
            )
        end

        cache.isfresh = false
    end
    y = _rf_ldiv!(
        cache.u, LinearSolve.@get_cacheval(cache, :RFLUFactorization)[1], cache.b,
        Val(P), Val(T)
    )
    return SciMLBase.build_linear_solution(alg, y, nothing, nothing; retcode = ReturnCode.Success)
end

# Apply an RF factorization to a right-hand side.
#
# Policy: wherever TriangularSolve has a native kernel (Float32/Float64 with a
# strided right-hand side), both backsolve legs must run on TriangularSolve —
# never on a BLAS kernel (`getrs!`/`trsm`/`trsv`).  TriangularSolve's kernels
# only take strided *matrix* right-hand sides (its vector entry point defers to
# BLAS above a size cutoff), so vectors are presented as n×1 matrices via a
# zero-copy reshape.  Matrix right-hand sides, solve-only, 1 BLAS thread,
# nrhs = 8 (TriangularSolve vs BLAS trsm, measured for #1117/#1153):
#
#   U leg (upper ldiv!):            L leg (unit-lower ldiv!):
#   n     trsm        TS            trsm        TS
#   64      4.09 us    1.78 us        5.06 us    1.62 us
#   128    12.36 us    6.27 us       12.57 us    4.42 us
#   256    49.18 us   22.04 us       48.93 us   19.08 us
#   500   154.79 us   79.64 us      150.16 us   75.22 us
#
# A single right-hand side wins below ~n=128 (0.78-0.98x of getrs!) and costs
# up to ~2.3x (1 thread) / ~1.2x (threaded) around n=512-1000; the policy
# deliberately keeps it on TriangularSolve at every size.
#
# The `Pivot` flag must come from the algorithm, not from `fact.ipiv`:
# RecursiveFactorization's pivot-free `lu!` returns the caller-supplied ipiv
# vector without writing it (identity from RecursiveFactorization >= 0.2.29,
# undefined memory before), so a `pivot = Val(false)` factorization must never
# consume `fact.ipiv` — neither through `LAPACK.getrs!` nor `_ipiv_rows!`.
# Doing so crashed with garbage pivots (segfault in `dlaswp`).
function _rf_ldiv!(
        u::StridedVector{T}, fact::LinearAlgebra.LU{T, <:StridedMatrix{T}},
        b::AbstractVector{T}, ::Val{Pivot}, ::Val{Thread}
    ) where {T <: Union{Float32, Float64}, Pivot, Thread}
    # view-then-reshape keeps this allocation-free: both wrappers are immutable
    # and passed by value, unlike reshape(::Vector), which heap-allocates a
    # Matrix header (measured 48 bytes/solve, breaking the allocation-free
    # re-solve contract of test/Core/lu_refactorization.jl).
    um = reshape(view(u, :), length(u), 1)
    um isa StridedMatrix{T} || return _rf_stdlib_ldiv!(u, fact, b, Val(Pivot))
    u === b || copyto!(u, b)
    Pivot && LinearAlgebra._ipiv_rows!(fact, 1:length(fact.ipiv), u)
    F = fact.factors
    TriangularSolve.ldiv!(UnitLowerTriangular(F), um, Val(Thread))
    TriangularSolve.ldiv!(UpperTriangular(F), um, Val(Thread))
    return u
end

function _rf_ldiv!(
        U::AbstractMatrix{T}, fact::LinearAlgebra.LU{T, <:StridedMatrix{T}},
        B::AbstractMatrix{T}, ::Val{Pivot}, ::Val{Thread}
    ) where {T <: Union{Float32, Float64}, Pivot, Thread}
    U === B || copyto!(U, B)
    Pivot && LinearAlgebra._ipiv_rows!(fact, 1:length(fact.ipiv), U)
    F = fact.factors
    TriangularSolve.ldiv!(UnitLowerTriangular(F), U, Val(Thread))
    TriangularSolve.ldiv!(UpperTriangular(F), U, Val(Thread))
    return U
end

# Types TriangularSolve has no native kernel for (complex, non-strided,
# non-BLAS eltypes) keep the stdlib path.
@inline function _rf_ldiv!(
        u::AbstractVector, fact::LinearAlgebra.LU, b::AbstractVector,
        ::Val{Pivot}, ::Val
    ) where {Pivot}
    return _rf_stdlib_ldiv!(u, fact, b, Val(Pivot))
end
@inline function _rf_ldiv!(
        U::AbstractMatrix, fact::LinearAlgebra.LU, B::AbstractMatrix,
        ::Val{Pivot}, ::Val
    ) where {Pivot}
    return _rf_stdlib_ldiv!(U, fact, B, Val(Pivot))
end

@inline _rf_stdlib_ldiv!(u, fact, b, ::Val{true}) = ldiv!(u, fact, b)
function _rf_stdlib_ldiv!(u, fact, b, ::Val{false})
    u === b || copyto!(u, b)
    ldiv!(UpperTriangular(fact.factors), ldiv!(UnitLowerTriangular(fact.factors), u))
    return u
end

# Enforcement helper used by the test suite: true iff TriangularSolve resolves
# `ldiv!(::TA, ::TB, ::Val)` to one of its native kernel methods rather than
# its LinearAlgebra catch-all, i.e. the argument types above stay off BLAS.
function _ts_native_backsolve(::Type{TA}, ::Type{TB}) where {TA, TB}
    for V in (Val{false}, Val{true})
        catchall = which(TriangularSolve.ldiv!, Tuple{Any, Any, V})
        m = which(TriangularSolve.ldiv!, Tuple{TA, TB, V})
        (m !== catchall && m.module === TriangularSolve) || return false
    end
    return true
end

# Mixed precision RecursiveFactorization implementation

const PREALLOCATED_RF32_LU = begin
    A = rand(Float32, 0, 0)
    luinst = ArrayInterface.lu_instance(A)
    (luinst, Vector{LinearAlgebra.BlasInt}(undef, 0))
end

function LinearSolve.init_cacheval(
        alg::RF32MixedLUFactorization{P, T}, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::LinearSolve.OperatorAssumptions
    ) where {P, T}
    # Pre-allocate appropriate 32-bit arrays based on input type
    m, n = size(A)
    T32 = eltype(A) <: Complex ? ComplexF32 : Float32
    A_32 = similar(A, T32)
    b_32 = similar(b, T32)
    u_32 = similar(u, T32)
    luinst = ArrayInterface.lu_instance(rand(T32, 0, 0))
    ipiv = Vector{LinearAlgebra.BlasInt}(undef, min(m, n))
    # Return tuple with pre-allocated arrays
    return (luinst, ipiv, A_32, b_32, u_32)
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::RF32MixedLUFactorization{P, T};
        kwargs...
    ) where {P, T}
    A = cache.A
    A = convert(AbstractMatrix, A)

    if cache.isfresh
        # Get pre-allocated arrays from cacheval
        luinst, ipiv, A_32, b_32, u_32 = LinearSolve.@get_cacheval(cache, :RF32MixedLUFactorization)
        # Compute 32-bit type on demand and copy A
        T32 = eltype(A) <: Complex ? ComplexF32 : Float32
        A_32 .= T32.(A)

        # Ensure ipiv is the right size
        if length(ipiv) != min(size(A_32)...)
            resize!(ipiv, min(size(A_32)...))
        end

        fact = RecursiveFactorization.lu!(A_32, ipiv, Val(P), Val(T), check = false)
        cache.cacheval = (fact, ipiv, A_32, b_32, u_32)

        if !LinearAlgebra.issuccess(fact)
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, nothing; retcode = ReturnCode.Failure
            )
        end

        cache.isfresh = false
    end

    # Get the factorization and pre-allocated arrays from the cache
    fact_cached, ipiv, A_32, b_32, u_32 = LinearSolve.@get_cacheval(cache, :RF32MixedLUFactorization)

    # Compute types on demand for conversions
    T32 = eltype(cache.A) <: Complex ? ComplexF32 : Float32
    Torig = eltype(cache.u)

    # Copy b to pre-allocated 32-bit array
    b_32 .= T32.(cache.b)

    # Solve in 32-bit precision
    _rf_ldiv!(u_32, fact_cached, b_32, Val(P), Val(T))

    # Convert back to original precision
    cache.u .= Torig.(u_32)

    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, nothing; retcode = ReturnCode.Success
    )
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::ButterflyFactorization;
        kwargs...
    )
    cache_A = cache.A
    cache_A = convert(AbstractMatrix, cache_A)
    cache_b = cache.b
    M, N = size(cache_A)
    workspace = cache.cacheval[1]
    thread = alg.thread

    if cache.isfresh
        @assert M == N "A must be square"
        if (size(workspace.A, 1) != M)
            workspace = RecursiveFactorization.🦋workspace(cache_A, cache_b)
        end
        (; A, b, ws, U, V, out, tmp, n) = workspace
        RecursiveFactorization.🦋mul!(A, ws)
        F = RecursiveFactorization.lu!(A, Val(false), thread)
        cache.cacheval = (workspace, F)
        cache.isfresh = false
    end

    workspace, F = cache.cacheval
    (; A, b, ws, U, V, out, tmp, n) = workspace
    b[1:M] .= cache_b
    mul!(tmp, U', b)

    # TriangularSolve.ldiv!
    TriangularSolve.ldiv!(F, tmp, thread)

    mul!(b, V, tmp)
    out .= @view b[1:n]
    return SciMLBase.build_linear_solution(alg, out, nothing, nothing)
end

function LinearSolve.init_cacheval(
        alg::ButterflyFactorization, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::LinearSolve.OperatorAssumptions
    )
    return ws = RecursiveFactorization.🦋workspace(A, b), RecursiveFactorization.lu!(rand(1, 1), Val(false), alg.thread)
end

LinearSolve._custom_can_reuse_adjoint_factorization(
    ::ButterflyFactorization, ::Tuple
) = true

function LinearSolve._custom_adjoint_factorization_solve(
        ::ButterflyFactorization, cacheval::Tuple, A, b::AbstractVector
    )
    workspace, factorization = cacheval
    n = workspace.n
    T = promote_type(eltype(workspace.U), eltype(b))
    padded_rhs = zeros(T, size(workspace.U, 1))
    copyto!(view(padded_rhs, 1:n), b)
    transformed_rhs = adjoint(workspace.V) * padded_rhs
    upper_solution = adjoint(factorization.U) \ transformed_rhs
    factorization_solution = adjoint(factorization.L) \ upper_solution
    solution = workspace.U * factorization_solution
    return solution[1:n]
end

# ---- SupernodalLU panel triangular solves ---------------------------------
# The vendored supernodal sparse LU (src/SupernodalLU) applies two BLAS-3
# trsms per supernode against its just-factored diagonal block: the L21 panel
# on the right by U11, and the U12 panel on the left by unit-L11.  Route them
# through TriangularSolve, which RecursiveFactorization already depends on
# and uses for its own trsms — so when RFLU is the dense default, the sparse
# solver's panel work runs on the same kernels.  Measured: recovers the
# 2D-mesh refactorization gap left by the stdlib trsms.
const SNLU = LinearSolve.SupernodalLU
const SNLUTypes = Union{Float32, Float64}

function SNLU._panel_rdiv!(W::Matrix{Tv}, np::Int, len::Int) where {Tv <: SNLUTypes}
    len > np || return nothing
    TriangularSolve.rdiv!(
        view(W, (np + 1):len, 1:np), UpperTriangular(view(W, 1:np, 1:np)), Val(false)
    )
    return nothing
end

function SNLU._panel_ldiv!(W::Matrix{Tv}, np::Int, Z::Matrix{Tv}) where {Tv <: SNLUTypes}
    isempty(Z) && return nothing
    TriangularSolve.ldiv!(
        UnitLowerTriangular(view(W, 1:np, 1:np)), Z, Val(false)
    )
    return nothing
end

end
