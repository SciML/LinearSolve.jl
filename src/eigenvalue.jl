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
    eigenpairs is requested (via `num_eigenpairs`) in an [`EigenvalueProblem`](@ref).
    """
    EnumX.@enumx EigenvalueTarget begin
        "Eigenvalues of largest magnitude, `abs(λ)` largest."
        LargestMagnitude
        "Eigenvalues of smallest magnitude, `abs(λ)` smallest."
        SmallestMagnitude
        "Eigenvalues with the largest (most positive) real part."
        LargestRealPart
        "Eigenvalues with the smallest (most negative) real part."
        SmallestRealPart
        "Eigenvalues with the largest (most positive) imaginary part."
        LargestImaginaryPart
        "Eigenvalues with the smallest (most negative) imaginary part."
        SmallestImaginaryPart
    end

    """
        EigenvalueProblem(A[, B], p = SciMLBase.NullParameters();
            num_eigenpairs = nothing, eigentarget = EigenvalueTarget.LargestMagnitude,
            shift = nothing, u0 = nothing)

    Define a standard or generalized eigenvalue problem.

    The standard problem is ``A v = λ v``. If `B` is supplied, the generalized problem
    is ``A v = λ B v``.

    ## Keyword arguments

      - `num_eigenpairs`: the number of eigenpairs (eigenvalues together with their
        eigenvectors) to compute. `nothing` (the default) requests every eigenpair for the
        dense solver, or a solver-chosen default for the iterative backends.
      - `eigentarget`: which part of the spectrum to return, as an
        [`EigenvalueTarget`](@ref). Defaults to the eigenvalues of largest magnitude.
      - `shift`: if supplied, return the eigenvalues nearest this shift (shift-and-invert).
      - `u0`: optional initial guess for the iterative backends.
    """
    struct EigenvalueProblem{
            AType, BType, NevType, TargetType, ShiftType, U0Type, PType, KType,
        }
        A::AType
        B::BType
        num_eigenpairs::NevType
        eigentarget::TargetType
        shift::ShiftType
        u0::U0Type
        p::PType
        kwargs::KType
    end

    function EigenvalueProblem(
            A, B = nothing, p = SciMLBase.NullParameters();
            num_eigenpairs = nothing,
            eigentarget::EigenvalueTarget.T = EigenvalueTarget.LargestMagnitude,
            shift = nothing, u0 = nothing, kwargs...
        )
        return EigenvalueProblem{
            typeof(A), typeof(B), typeof(num_eigenpairs), typeof(eigentarget),
            typeof(shift), typeof(u0), typeof(p), typeof(kwargs),
        }(A, B, num_eigenpairs, eigentarget, shift, u0, p, kwargs)
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
    values, vectors = _select_eigenpairs(
        F.values, F.vectors, prob.num_eigenpairs, prob.eigentarget, prob.shift
    )
    return build_eigenvalue_solution(prob, alg, values, vectors)
end

function SciMLBase.solve(
        prob::EigenvalueProblem, alg::AbstractEigenvalueAlgorithm, args...; kwargs...
    )
    error("The eigenvalue backend $(typeof(alg)) is not available. Load its package before solving with this algorithm.")
end

function default_num_eigenpairs(prob::EigenvalueProblem)
    n = size(prob.A, 2)
    # Only the iterative backends call this; requesting the full dimension `n`
    # is invalid/degenerate for them, so default to a small subset.
    return prob.num_eigenpairs === nothing ? min(n, 6) : prob.num_eigenpairs
end

function _select_eigenpairs(values, vectors, num_eigenpairs, eigentarget, shift)
    nvals = length(values)
    howmany = num_eigenpairs === nothing ? nvals : min(num_eigenpairs, nvals)
    ord = _eigenvalue_order(values, eigentarget, shift)
    idxs = ord[1:howmany]
    return values[idxs], vectors[:, idxs]
end

function _eigenvalue_order(values, eigentarget::EigenvalueTarget.T, shift)
    if shift !== nothing
        return sortperm(abs.(values .- shift))
    elseif eigentarget == EigenvalueTarget.LargestMagnitude
        return sortperm(abs.(values); rev = true)
    elseif eigentarget == EigenvalueTarget.SmallestMagnitude
        return sortperm(abs.(values))
    elseif eigentarget == EigenvalueTarget.LargestRealPart
        return sortperm(real.(values); rev = true)
    elseif eigentarget == EigenvalueTarget.SmallestRealPart
        return sortperm(real.(values))
    elseif eigentarget == EigenvalueTarget.LargestImaginaryPart
        return sortperm(imag.(values); rev = true)
    else # EigenvalueTarget.SmallestImaginaryPart
        return sortperm(imag.(values))
    end
end
