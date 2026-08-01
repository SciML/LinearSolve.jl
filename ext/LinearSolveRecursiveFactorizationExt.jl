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
        cache.u, LinearSolve.@get_cacheval(cache, :RFLUFactorization)[1], cache.b, Val(T)
    )
    return SciMLBase.build_linear_solution(alg, y, nothing, nothing; retcode = ReturnCode.Success)
end

# Apply an RF factorization to a right-hand side.
#
# `RecursiveFactorization` already routes its own `lu!` through TriangularSolve,
# but the `ldiv!` that consumes the factorization only does so for the pivotless
# `NotIPIV` case (RecursiveFactorization/src/lu.jl); a pivoted `LU` falls back to
# LinearAlgebra, i.e. BLAS `trsv`/`trsm`.  For a matrix right-hand side that
# leaves a consistent ~1.6x on the table at every size we measured, because
# TriangularSolve's blocked kernels beat `trsm` here:
#
#   n     BLAS trsm   TriangularSolve
#   32     2.19 us      1.37 us  (1.60x)
#   64     6.27 us      3.76 us  (1.67x)
#   128   21.02 us     12.72 us  (1.65x)
#   256   86.08 us     54.07 us  (1.59x)
#   500  276.20 us    175.81 us  (1.58x)
#
# For a single vector right-hand side TriangularSolve has no advantage (it has
# no vector kernel, and reshaping to n x 1 measured 1.09x at n=128 but 0.88x by
# n=256), so vectors keep the stdlib path.
@inline function _rf_ldiv!(
        u::AbstractVector, fact::LinearAlgebra.LU, b::AbstractVector, ::Val
    )
    return ldiv!(u, fact, b)
end

function _rf_ldiv!(
        U::AbstractMatrix{T}, fact::LinearAlgebra.LU{T, <:StridedMatrix{T}},
        B::AbstractMatrix{T}, ::Val{Thread}
    ) where {T <: LinearAlgebra.BlasFloat, Thread}
    # A single column is the vector case in disguise: measured 0.87-0.90x at
    # n >= 256, so it keeps the stdlib path.
    size(B, 2) == 1 && return ldiv!(U, fact, B)
    U === B || copyto!(U, B)
    LinearAlgebra._ipiv_rows!(fact, 1:length(fact.ipiv), U)
    F = fact.factors
    TriangularSolve.ldiv!(UnitLowerTriangular(F), U, Val(Thread))
    TriangularSolve.ldiv!(UpperTriangular(F), U, Val(Thread))
    return U
end

# Non-strided or non-BLAS element types keep the stdlib path.
@inline function _rf_ldiv!(
        U::AbstractMatrix, fact::LinearAlgebra.LU, B::AbstractMatrix, ::Val
    )
    return ldiv!(U, fact, B)
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
    ldiv!(u_32, fact_cached, b_32)

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
