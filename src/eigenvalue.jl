using SciMLBase: EigenvalueProblem, EigenvalueSolution, EigenvalueTarget,
    build_eigenvalue_solution

"""
    AbstractEigenvalueAlgorithm

Base type for algorithms that solve an [`EigenvalueProblem`](@ref).
"""
abstract type AbstractEigenvalueAlgorithm <: SciMLBase.AbstractLinearAlgorithm end

"""
    DenseEigen()

Solve the `EigenvalueProblem` with `LinearAlgebra.eigen`. This is the default
algorithm: it computes the full dense eigendecomposition and then selects the
requested eigenpairs (via `num_eigenpairs`, `eigentarget`, or `shift`) from it.
Best for small to moderately sized dense matrices where every eigenpair (or a
sizable fraction of them) is needed.
"""
struct DenseEigen <: AbstractEigenvalueAlgorithm end

# The iterative backends forward any extra keyword arguments to the underlying
# solver (`Arpack.eigs`, `ArnoldiMethod.partialschur`, `KrylovKit.eigsolve`,
# `JacobiDavidson.jdqr`). They are keyword-only: passing positional arguments is
# an error, and unrecognized keywords are rejected by the underlying solver.
"""
    ArpackJL(; kwargs...)

Solve the `EigenvalueProblem` with [Arpack.jl](https://github.com/JuliaLinearAlgebra/Arpack.jl)'s
`eigs`, an iterative Krylov (implicitly restarted Arnoldi/Lanczos) solver well suited to
computing a handful of extremal eigenpairs of a large sparse or structured matrix.
Extra `kwargs` are forwarded to `Arpack.eigs`.

!!! note

    Using this solver requires loading Arpack.jl, i.e. `using Arpack`.
"""
struct ArpackJL{K <: NamedTuple} <: AbstractEigenvalueAlgorithm
    kwargs::K
end
ArpackJL(; kwargs...) = ArpackJL((; kwargs...))

"""
    ArnoldiMethodJL

Algorithm type constructed by [`ArnoldiMethod`](@ref); see its docstring for details.
"""
struct ArnoldiMethodJL{K <: NamedTuple} <: AbstractEigenvalueAlgorithm
    kwargs::K
end

"""
    ArnoldiMethod(; kwargs...)

Solve the `EigenvalueProblem` with
[ArnoldiMethod.jl](https://github.com/JuliaLinearAlgebra/ArnoldiMethod.jl)'s
`partialschur`, a pure-Julia implicitly restarted Arnoldi method for large sparse or
structured matrices. Does not support `eigentarget = EigenvalueTarget.SmallestMagnitude`
directly; use `shift` for shift-and-invert instead, or another backend. Extra `kwargs`
are forwarded to `ArnoldiMethod.partialschur`.

!!! note

    Using this solver requires loading ArnoldiMethod.jl, i.e. `using ArnoldiMethod`.
"""
ArnoldiMethod(; kwargs...) = ArnoldiMethodJL((; kwargs...))

"""
    KrylovKitEigen(; kwargs...)

Solve the `EigenvalueProblem` with
[KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl)'s `eigsolve`, a Krylov solver
supporting both standard and generalized eigenvalue problems, extremal and interior
(shifted) targets. Extra `kwargs` are forwarded to `KrylovKit.eigsolve`.

!!! note

    Using this solver requires loading KrylovKit.jl, i.e. `using KrylovKit`.
"""
struct KrylovKitEigen{K <: NamedTuple} <: AbstractEigenvalueAlgorithm
    kwargs::K
end
KrylovKitEigen(; kwargs...) = KrylovKitEigen((; kwargs...))

"""
    JacobiDavidsonJL(; kwargs...)

Solve the `EigenvalueProblem` with
[JacobiDavidson.jl](https://github.com/haampie/JacobiDavidson.jl)'s `jdqr`, a
target/interior method that finds the eigenvalues nearest a given `shift`. Does not
support generalized eigenvalue problems (upstream `jdqz` is broken). Extra `kwargs` are
forwarded to `JacobiDavidson.jdqr`.

!!! note

    Using this solver requires loading JacobiDavidson.jl, i.e. `using JacobiDavidson`.
"""
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
