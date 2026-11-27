"""
    CertifiedLUQR(; pivot_tol = nothing, rank_tol = nothing, reference = :zero)

Solve square, dense CPU floating-point systems with LU followed, when necessary,
by column-pivoted QR. Unlike a least-squares solver, this algorithm returns
`ReturnCode.Success` only when
`norm(A*x - b) <= reltol * (norm(A)*norm(x) + norm(b))`.
Here `norm(A)` is the Frobenius norm; `abstol` is unused. An inconsistent system
returns `ReturnCode.Infeasible`, with the rejected approximation in `sol.u`.
Non-finite input or arithmetic returns `ReturnCode.Failure`.

LU uses the usual small-matrix generic/RecursiveFactorization size rules, then
LAPACK for larger matrices. Every LU pivot must exceed `pivot_tol * norm(A)`
(default `sqrt(eps(real(eltype(A))))`). Otherwise QR determines the numerical
rank: leading diagonals of `R` must exceed `rank_tol * abs(R[1,1])`
(default `size(A,1) * eps(real(eltype(A)))`). These are relative thresholds.

Rank-deficient consistent systems get the minimum-norm solution by default.
With `reference = :u0`, a copy of the problem's `u0` is saved at initialization
and the result is `u0 + A⁺*(b - A*u0)`, using the QR numerical rank. An omitted
`u0` means zero. The saved reference stays fixed across solves; to change it,
copy a new vector into `cache.cacheval.reference`.

`sol.stats.numerical_rank` and `sol.stats.relative_residual` report the final
rank and scaled residual. The same fields are available in `cache.cacheval`.
Assign `cache.A` (even after mutating it in place) to request refactorization;
assigning only `cache.b` reuses the factors. Workspace is reused for fixed
dimensions. LU refactorization and RHS-only solves can be allocation-free;
QR refactorization uses LAPACK's allocating workspace wrapper.

Supported inputs are strided matrices and vectors with matching `Float32`,
`Float64`, `ComplexF32`, or `ComplexF64` element types.
"""
struct CertifiedLUQR{P, R} <: SciMLLinearSolveAlgorithm
    pivot_tol::P
    rank_tol::R
    reference::Symbol
    function CertifiedLUQR(; pivot_tol = nothing, rank_tol = nothing, reference = :zero)
        reference in (:zero, :u0) || throw(ArgumentError("reference must be :zero or :u0"))
        for tol in (pivot_tol, rank_tol)
            isnothing(tol) || (isfinite(tol) && 0 <= tol < 1) ||
                throw(ArgumentError("relative thresholds must be finite and in [0, 1)"))
        end
        return new{typeof(pivot_tol), typeof(rank_tol)}(pivot_tol, rank_tol, reference)
    end
end

needs_concrete_A(::CertifiedLUQR) = true
update_tolerances_internal!(cache, ::CertifiedLUQR, abstol, reltol) = nothing

mutable struct CertifiedLUQRCache{L, Q, C, V, T}
    lu::L
    qr::Q
    cod::C
    reference::V
    rhs::V
    correction::V
    residual::V
    pivot_tol::T
    rank_tol::T
    anorm::T
    numerical_rank::Int
    relative_residual::T
    use_qr::Bool
end

function _certified_lu_algorithm(A)
    n = size(A, 1)
    n <= 10 && return GenericLUFactorization()
    if eltype(A) <: Union{Float32, Float64}
        if (n <= 100 || (isopenblas() && n <= 500) || (usemkl && n <= 200)) &&
                userecursivefactorization(A)
            return RFLUFactorization(throwerror = false)
        elseif n <= 32 || (isopenblas() && n <= 256)
            return GenericLUFactorization()
        end
    end
    return LUFactorization()
end

function init_cacheval(
        alg::CertifiedLUQR, A::StridedMatrix{T}, b::StridedVector{T}, u::StridedVector{T},
        Pl, Pr, maxiters::Int, abstol, reltol, verbose, assumptions
    ) where {T <: BLASELTYPES}
    n = size(A, 1)
    size(A, 2) == length(b) == length(u) == n || throw(DimensionMismatch("expected square A and matching vectors"))
    RT = real(T)
    pivot_tol = isnothing(alg.pivot_tol) ? sqrt(eps(RT)) : RT(alg.pivot_tol)
    rank_tol = isnothing(alg.rank_tol) ? n * eps(RT) : RT(alg.rank_tol)
    lu_cache = init(
        LinearProblem(Matrix(A), Vector(b)), _certified_lu_algorithm(A);
        alias = LinearAliasSpecifier(alias_A = true, alias_b = true), verbose = false
    )
    qr_cache = LinearAlgebra.QRPivoted(zeros(T, n, n), zeros(T, n), zeros(BlasInt, n))
    cod = LinearAlgebra.QR(zeros(T, n, n), zeros(T, n))
    reference = alg.reference === :u0 ? Vector(u) : zeros(T, n)
    return CertifiedLUQRCache(
        lu_cache, qr_cache, cod, reference, zeros(T, n), zeros(T, n), zeros(T, n),
        pivot_tol, rank_tol, zero(RT), 0, RT(Inf), false
    )
end

function init_cacheval(alg::CertifiedLUQR, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol, verbose, assumptions)
    throw(ArgumentError("CertifiedLUQR requires strided CPU floating-point matrices and vectors with matching element types"))
end

_certified_lu_factor(cache, ::GenericLUFactorization) = cache.cacheval.fact
_certified_lu_factor(cache, ::RFLUFactorization) = cache.cacheval[1]
_certified_lu_factor(cache, ::LUFactorization) = cache.cacheval

function _certified_qr!(cv, A)
    F = cv.qr
    copyto!(F.factors, A)
    fill!(F.jpvt, 0)
    LAPACK.geqp3!(F.factors, F.jpvt, F.τ)
    n = size(A, 1)
    r = 0
    threshold = n == 0 ? zero(cv.rank_tol) : cv.rank_tol * abs(F.factors[1, 1])
    for i in 1:n
        abs(F.factors[i, i]) > threshold || break
        r = i
    end
    cv.numerical_rank = r

    # A*P = Q*R. Factor R[1:r,:]' = Z*T, so the minimum-norm correction
    # is P*Z*[T' \ (Q'*rhs)[1:r]; 0]. Avoid forming normal equations.
    C = cv.cod
    fill!(C.factors, 0)
    fill!(C.τ, 0)
    for j in 1:r, i in j:n
        C.factors[i, j] = conj(F.factors[j, i])
    end
    # The stdlib unblocked QR kernel, with cache-owned reflector coefficients.
    for k in 1:min(r, n - 1 + !(eltype(A) <: Real))
        v = view(C.factors, k:n, k)
        C.τ[k] = LinearAlgebra.reflector!(v)
        LinearAlgebra.reflectorApply!(v, C.τ[k], view(C.factors, k:n, (k + 1):r))
    end
    cv.use_qr = true
    return nothing
end

# Select stdlib's generic reflector application to avoid the temporary LAPACK
# work arrays allocated by its strided-vector specialization.
const _CertifiedAdjointPackedQ = LinearAlgebra.AdjointQ{<:Any, <:LinearAlgebra.QRPackedQ}

function _certified_qmul!(F, x, ::Val{false})
    return invoke(LinearAlgebra.lmul!, Tuple{LinearAlgebra.QRPackedQ, AbstractVecOrMat}, F.Q, x)
end
function _certified_qmul!(F, x, ::Val{true})
    return invoke(
        LinearAlgebra.lmul!, Tuple{_CertifiedAdjointPackedQ, AbstractVecOrMat},
        adjoint(F.Q), x
    )
end

function _certified_qr_solve!(cv, A, b, u)
    mul!(cv.rhs, A, cv.reference)
    @. cv.rhs = b - cv.rhs
    _certified_qmul!(cv.qr, cv.rhs, Val(true))
    r = cv.numerical_rank
    fill!(cv.correction, 0)
    copyto!(cv.correction, 1, cv.rhs, 1, r)
    ldiv!(
        adjoint(UpperTriangular(view(cv.cod.factors, 1:r, 1:r))),
        view(cv.correction, 1:r)
    )
    _certified_qmul!(cv.cod, cv.correction, Val(false))
    for i in eachindex(u)
        u[cv.qr.jpvt[i]] = cv.reference[cv.qr.jpvt[i]] + cv.correction[i]
    end
    return nothing
end

function _certified_residual!(cv, A, b, u)
    mul!(cv.residual, A, u)
    cv.residual .-= b
    residual = norm(cv.residual)
    scale = cv.anorm * norm(u) + norm(b)
    cv.relative_residual = if !isfinite(residual) || !isfinite(scale)
        oftype(cv.relative_residual, Inf)
    elseif iszero(scale)
        iszero(residual) ? zero(scale) : oftype(scale, Inf)
    else
        residual / scale
    end
    return cv.relative_residual
end

function SciMLBase.solve!(cache::LinearCache, alg::CertifiedLUQR; kwargs...)
    cv = cache.cacheval
    A, b, u = cache.A, cache.b, cache.u
    size(A) == size(cv.qr.factors) && length(b) == length(u) == length(cv.rhs) ||
        throw(DimensionMismatch("initialize a new CertifiedLUQR cache when dimensions change"))
    retcode = ReturnCode.Failure
    cv.relative_residual = oftype(cv.relative_residual, Inf)
    if cache.isfresh
        cv.numerical_rank = 0
        cv.use_qr = false
        cv.anorm = norm(A)
    end
    if isfinite(cv.anorm) && all(isfinite, A) && all(isfinite, b) && all(isfinite, cv.reference)
        if cache.isfresh
            copyto!(cv.lu.A, A)
            cv.lu.A = cv.lu.A
        end
        if !cv.use_qr
            cv.lu.b = b
            sol = solve!(cv.lu)
            F = _certified_lu_factor(cv.lu, cv.lu.alg)
            good_pivots = sol.retcode == ReturnCode.Success && all(
                i -> abs(F.factors[i, i]) > cv.pivot_tol * cv.anorm, axes(A, 1)
            )
            if good_pivots
                copyto!(u, sol.u)
                cv.numerical_rank = size(A, 1)
                if _certified_residual!(cv, A, b, u) <= cache.reltol
                    retcode = ReturnCode.Success
                end
            end
            if retcode != ReturnCode.Success
                _certified_qr!(cv, A)
            end
        end
        if cv.use_qr
            _certified_qr_solve!(cv, A, b, u)
            relative_residual = _certified_residual!(cv, A, b, u)
            retcode = !isfinite(relative_residual) ? ReturnCode.Failure :
                relative_residual <= cache.reltol ? ReturnCode.Success : ReturnCode.Infeasible
        end
        cache.isfresh = false
    end
    stats = (; cv.numerical_rank, cv.relative_residual)
    return SciMLBase.build_linear_solution(alg, u, nothing, nothing; retcode, stats)
end
