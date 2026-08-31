using SciMLBase: EigenvalueProblem, EigenvalueSolution, EigenvalueTarget,
    build_eigenvalue_solution

"""
    AbstractEigenvalueAlgorithm

Base type for algorithms that solve an `EigenvalueProblem`. An instance is passed as
the second argument of `solve(prob::EigenvalueProblem, alg)`; when no algorithm is
given, `DenseEigen()` is used.

The concrete algorithms are:

  - `DenseEigen`: dense `LinearAlgebra.eigen`, the default; always available.
  - `ArpackJL`: `Arpack.eigs` (requires `using Arpack`).
  - `ArnoldiMethodJL` / `ArnoldiMethod`: `ArnoldiMethod.partialschur` (requires
    `using ArnoldiMethod`).
  - `KrylovKitEigen`: `KrylovKit.eigsolve` (requires `using KrylovKit`).
  - `JacobiDavidsonJL`: `JacobiDavidson.jdqr` (requires `using JacobiDavidson`).

A new backend subtypes this type and defines
`SciMLBase.solve(prob::EigenvalueProblem, alg::MyAlg, args...; kwargs...)`, returning
an `EigenvalueSolution` (see `SciMLBase.build_eigenvalue_solution`). The fallback
method for the abstract type throws "The eigenvalue backend ... is not available",
which is what a user sees when an extension-backed algorithm is used before its
package has been loaded.
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
    ArnoldiMethodJL(; kwargs...)

Solve the `EigenvalueProblem` with
[ArnoldiMethod.jl](https://github.com/JuliaLinearAlgebra/ArnoldiMethod.jl)'s
`partialschur`, a pure-Julia implicitly restarted Arnoldi method for computing a few
eigenpairs of a large sparse or structured matrix. Extra `kwargs` are forwarded to
`ArnoldiMethod.partialschur` (for example `tol`, `mindim`, `maxdim`, `restarts`).

Restrictions of the backend:

  - Standard problems only: a generalized problem (`B !== nothing`) raises an error.
  - `eigentarget = EigenvalueTarget.SmallestMagnitude` is not supported directly
    (ArnoldiMethod has no such target); supply `shift` for shift-and-invert, which
    factorizes `A - shift*I` and finds the eigenvalues nearest `shift`, or use another
    backend such as `KrylovKitEigen()` or `ArpackJL()`.

`ArnoldiMethodJL(; kwargs...)` and `ArnoldiMethod(; kwargs...)` build the same
algorithm object; this is the spelling to prefer once the ArnoldiMethod.jl package is
loaded. Loading that package is what makes the solver available, and it binds the name
`ArnoldiMethod` to the module, so after `using LinearSolve, ArnoldiMethod` a bare
`ArnoldiMethod(; kwargs...)` reaches the module rather than the constructor and fails
with "objects of type Module are not callable" (and `?ArnoldiMethod` shows the module's
help). `ArnoldiMethodJL` does not collide, and matches how the other backends are named.

!!! note

    Using this solver requires loading ArnoldiMethod.jl, i.e. `using ArnoldiMethod`.
"""
struct ArnoldiMethodJL{K <: NamedTuple} <: AbstractEigenvalueAlgorithm
    kwargs::K
end

ArnoldiMethodJL(; kwargs...) = ArnoldiMethodJL((; kwargs...))

"""
    ArnoldiMethod(; kwargs...)

Solve the `EigenvalueProblem` with
[ArnoldiMethod.jl](https://github.com/JuliaLinearAlgebra/ArnoldiMethod.jl)'s
`partialschur`, a pure-Julia implicitly restarted Arnoldi method for large sparse or
structured matrices. Extra `kwargs` are forwarded to `ArnoldiMethod.partialschur`.

Restrictions of the backend:

  - Standard problems only: a generalized problem (`B !== nothing`) raises an error.
  - `eigentarget = EigenvalueTarget.SmallestMagnitude` is not supported directly;
    supply `shift` for shift-and-invert instead, or use another backend such as
    `KrylovKitEigen()` or `ArpackJL()`.

This constructor returns an `ArnoldiMethodJL`; the two names build the same algorithm.
Once the ArnoldiMethod.jl package is loaded the bare name `ArnoldiMethod` refers to
that module, so use `ArnoldiMethodJL(; kwargs...)` (or `LinearSolve.ArnoldiMethod`)
in that case.

!!! note

    Using this solver requires loading ArnoldiMethod.jl, i.e. `using ArnoldiMethod`.
"""
ArnoldiMethod(; kwargs...) = ArnoldiMethodJL((; kwargs...))

"""
    KrylovKitEigen(; kwargs...)

Solve the `EigenvalueProblem` with
[KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl)'s `eigsolve`, a Krylov solver
for a few extremal or interior (shifted) eigenpairs of a large sparse or structured
matrix. Extra `kwargs` are forwarded to the KrylovKit call (for example `tol`,
`krylovdim`, `maxiter`, `issymmetric`, `ishermitian`, `isposdef`).

Which KrylovKit routine runs depends on the problem:

  - Standard problem, no `shift`: `KrylovKit.eigsolve(A, nev, which)` with `which`
    derived from `eigentarget`; no symmetry or definiteness is required of `A`.
  - Any problem with a `shift`: shift-and-invert. `A - shift*I` (or `A - shift*B` for a
    generalized problem) is factorized and `KrylovKit.eigsolve` is run on the inverse
    operator, so general (non-Hermitian) pencils are supported on this path.
  - Generalized problem without a `shift`: `KrylovKit.geneigsolve((A, B), nev, which)`.
    KrylovKit only implements the symmetric/Hermitian case with positive definite `B`
    and throws an `ArgumentError` otherwise (it also rejects the imaginary-part
    targets). For matrices these properties are detected automatically; for other
    operator types pass `issymmetric = true`/`ishermitian = true` and `isposdef = true`
    through `kwargs`. For a non-Hermitian pencil supply a `shift` instead.

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
