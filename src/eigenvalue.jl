# `EigenvalueProblem`, `EigenvalueSolution`, `EigenvalueTarget`, and
# `build_eigenvalue_solution` are defined natively in SciMLBase (analogous to
# `LinearProblem`/`LinearSolution`) once an upstream release adds them. Older
# SciMLBase versions lack them, so fall back to local definitions here,
# preserving the same public interface either way. This mirrors the existing
# `@static if isdefined(SciMLBase, :DiffEqArrayOperator)` gate further down in
# this module for the same reason: keep a wide SciMLBase compat range instead
# of forcing every LinearSolve user onto an upstream version they may not need.
@static if isdefined(SciMLBase, :EigenvalueProblem)
    using SciMLBase: EigenvalueProblem, EigenvalueSolution, EigenvalueTarget,
        build_eigenvalue_solution
else
    """
        EigenvalueTarget

    Enum selecting which part of the spectrum is returned when only a subset of the
    eigenpairs is requested (via `nev`) in an [`EigenvalueProblem`](@ref). The `which`
    keyword accepts either an `EigenvalueTarget` value or, for convenience, the
    corresponding ARPACK-style `Symbol` noted for each variant below.
    """
    EnumX.@enumx EigenvalueTarget begin
        "Eigenvalues of largest magnitude, `abs(λ)` largest (symbol `:LM`)."
        LargestMagnitude
        "Eigenvalues of smallest magnitude, `abs(λ)` smallest (symbol `:SM`)."
        SmallestMagnitude
        "Eigenvalues with the largest (most positive) real part (symbol `:LR`)."
        LargestRealPart
        "Eigenvalues with the smallest (most negative) real part (symbol `:SR`)."
        SmallestRealPart
        "Eigenvalues with the largest (most positive) imaginary part (symbol `:LI`)."
        LargestImaginaryPart
        "Eigenvalues with the smallest (most negative) imaginary part (symbol `:SI`)."
        SmallestImaginaryPart
    end

    # Normalize a user-supplied `which` (an `EigenvalueTarget` or an ARPACK-style
    # `Symbol`) to an `EigenvalueTarget`, throwing on anything else.
    _eigenvalue_target(w::EigenvalueTarget.T) = w
    function _eigenvalue_target(w::Symbol)
        return w === :LM ? EigenvalueTarget.LargestMagnitude :
            w === :SM ? EigenvalueTarget.SmallestMagnitude :
            w === :LR ? EigenvalueTarget.LargestRealPart :
            w === :SR ? EigenvalueTarget.SmallestRealPart :
            w === :LI ? EigenvalueTarget.LargestImaginaryPart :
            w === :SI ? EigenvalueTarget.SmallestImaginaryPart :
            throw(ArgumentError("unsupported eigenvalue selector `which = $(repr(w))`; expected an `EigenvalueTarget` or one of :LM, :SM, :LR, :SR, :LI, :SI"))
    end
    _eigenvalue_target(w) = throw(ArgumentError("`which` must be an `EigenvalueTarget` or a Symbol (:LM, :SM, :LR, :SR, :LI, :SI), got $(repr(w))"))

    """
        EigenvalueProblem(A[, B], p = SciMLBase.NullParameters();
            nev = nothing, which = EigenvalueTarget.LargestMagnitude,
            sigma = nothing, u0 = nothing)

    Define a standard or generalized eigenvalue problem.

    The standard problem is ``A v = λ v``. If `B` is supplied, the generalized problem
    is ``A v = λ B v``.

    ## Keyword arguments

      - `nev`: the number of eigenpairs (eigenvalues together with their eigenvectors) to
        compute. `nothing` (the default) requests every eigenpair for the dense solver, or
        a solver-chosen default for the iterative backends.
      - `which`: which part of the spectrum to return, as an [`EigenvalueTarget`](@ref).
        An ARPACK-style `Symbol` (`:LM`, `:SM`, `:LR`, `:SR`, `:LI`, `:SI`) is also accepted
        and converted to the corresponding `EigenvalueTarget`. Defaults to the eigenvalues of
        largest magnitude.
      - `sigma`: if supplied, return the eigenvalues nearest this shift (shift-and-invert).
      - `u0`: optional initial guess for the iterative backends.
    """
    struct EigenvalueProblem{AType, BType, NevType, WhichType, SigmaType, U0Type, PType, KType}
        A::AType
        B::BType
        nev::NevType
        which::WhichType
        sigma::SigmaType
        u0::U0Type
        p::PType
        kwargs::KType
    end

    function EigenvalueProblem(
            A, B = nothing, p = SciMLBase.NullParameters();
            nev = nothing, which = EigenvalueTarget.LargestMagnitude,
            sigma = nothing, u0 = nothing, kwargs...
        )
        target = _eigenvalue_target(which)
        return EigenvalueProblem{
            typeof(A), typeof(B), typeof(nev), typeof(target), typeof(sigma),
            typeof(u0), typeof(p), typeof(kwargs),
        }(A, B, nev, target, sigma, u0, p, kwargs)
    end

    struct EigenvalueSolution{T, N, U, V, P, A, R, S} <: SciMLBase.AbstractNoTimeSolution{T, N}
        u::U
        vectors::V
        prob::P
        alg::A
        retcode::ReturnCode.T
        resid::R
        stats::S
    end

    function build_eigenvalue_solution(
            prob, alg, values, vectors;
            retcode = ReturnCode.Success, resid = nothing, stats = nothing
        )
        T = eltype(eltype(values))
        N = length((size(values)...,))
        return EigenvalueSolution{
            T, N, typeof(values), typeof(vectors), typeof(prob), typeof(alg),
            typeof(resid), typeof(stats),
        }(values, vectors, prob, alg, retcode, resid, stats)
    end

    export EigenvalueProblem, EigenvalueSolution, EigenvalueTarget
end

abstract type AbstractEigenvalueAlgorithm <: SciMLBase.AbstractLinearAlgorithm end

struct DenseEigen <: AbstractEigenvalueAlgorithm end

# The iterative backends forward any extra keyword arguments to the underlying
# solver (`Arpack.eigs`, `ArnoldiMethod.partialschur`, `KrylovKit.eigsolve`,
# `JacobiDavidson.jdqr`). They are keyword-only: passing positional arguments is
# an error, and unrecognized keywords are rejected by the underlying solver.
struct ArpackJL{K <: NamedTuple} <: AbstractEigenvalueAlgorithm
    kwargs::K
end
ArpackJL(; kwargs...) = ArpackJL((; kwargs...))

struct ArnoldiMethodJL{K <: NamedTuple} <: AbstractEigenvalueAlgorithm
    kwargs::K
end
ArnoldiMethod(; kwargs...) = ArnoldiMethodJL((; kwargs...))

struct KrylovKitEigen{K <: NamedTuple} <: AbstractEigenvalueAlgorithm
    kwargs::K
end
KrylovKitEigen(; kwargs...) = KrylovKitEigen((; kwargs...))

struct JacobiDavidsonJL{K <: NamedTuple} <: AbstractEigenvalueAlgorithm
    kwargs::K
end
JacobiDavidsonJL(; kwargs...) = JacobiDavidsonJL((; kwargs...))

SciMLBase.solve(prob::EigenvalueProblem, args...; kwargs...) =
    solve(prob, nothing, args...; kwargs...)

function SciMLBase.solve(prob::EigenvalueProblem, ::Nothing, args...; kwargs...)
    return solve(prob, DenseEigen(), args...; kwargs...)
end

function SciMLBase.solve(prob::EigenvalueProblem, alg::DenseEigen, args...; kwargs...)
    kw = (; prob.kwargs..., kwargs...)
    F = if isnothing(prob.B)
        LinearAlgebra.eigen(prob.A; kw...)
    elseif prob.B isa UniformScaling
        LinearAlgebra.eigen(prob.A / prob.B.λ; kw...)
    else
        LinearAlgebra.eigen(prob.A, prob.B; kw...)
    end
    values, vectors = _select_eigenpairs(F.values, F.vectors, prob.nev, prob.which, prob.sigma)
    return build_eigenvalue_solution(prob, alg, values, vectors)
end

function SciMLBase.solve(
        prob::EigenvalueProblem, alg::AbstractEigenvalueAlgorithm, args...; kwargs...
    )
    error("The eigenvalue backend $(typeof(alg)) is not available. Load its package before solving with this algorithm.")
end

function default_nev(prob::EigenvalueProblem)
    n = size(prob.A, 2)
    # Only the iterative backends call this; requesting the full dimension `n`
    # is invalid/degenerate for them, so default to a small subset.
    return prob.nev === nothing ? min(n, 6) : prob.nev
end

function _select_eigenpairs(values, vectors, nev, which, sigma)
    nvals = length(values)
    howmany = nev === nothing ? nvals : min(nev, nvals)
    ord = _eigenvalue_order(values, which, sigma)
    idxs = ord[1:howmany]
    return values[idxs], vectors[:, idxs]
end

# ARPACK-style symbol for the backends that take `which` as a `Symbol`.
function _target_symbol(w::EigenvalueTarget.T)
    return w == EigenvalueTarget.LargestMagnitude ? :LM :
        w == EigenvalueTarget.SmallestMagnitude ? :SM :
        w == EigenvalueTarget.LargestRealPart ? :LR :
        w == EigenvalueTarget.SmallestRealPart ? :SR :
        w == EigenvalueTarget.LargestImaginaryPart ? :LI :
        :SI
end

function _eigenvalue_order(values, which::EigenvalueTarget.T, sigma)
    if sigma !== nothing
        return sortperm(abs.(values .- sigma))
    elseif which == EigenvalueTarget.LargestMagnitude
        return sortperm(abs.(values); rev = true)
    elseif which == EigenvalueTarget.SmallestMagnitude
        return sortperm(abs.(values))
    elseif which == EigenvalueTarget.LargestRealPart
        return sortperm(real.(values); rev = true)
    elseif which == EigenvalueTarget.SmallestRealPart
        return sortperm(real.(values))
    elseif which == EigenvalueTarget.LargestImaginaryPart
        return sortperm(imag.(values); rev = true)
    else # EigenvalueTarget.SmallestImaginaryPart
        return sortperm(imag.(values))
    end
end
