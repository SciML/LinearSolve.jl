# The `SciMLLinearSolveAlgorithm` interface: the fallbacks that give the
# abstract type its contract, and the introspection used to check that every
# algorithm honours it.

const ALGORITHM_INTERFACE_DOCS = """
Required:
  * `SciMLBase.solve!(cache::LinearCache, alg::MyAlg; kwargs...)`
  * `LinearSolve.needs_concrete_A(alg::MyAlg)::Bool`
Optional (documented defaults apply when not defined):
  * `LinearSolve.init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assumptions)`
  * `LinearSolve.default_alias_A(alg, A, b)::Bool`
  * `LinearSolve.default_alias_b(alg, A, b)::Bool`
  * `LinearSolve.needs_square_A(alg)::Bool`
  * `LinearSolve.update_tolerances_internal!(cache, alg, abstol, reltol)`
See the "Linear Solver Algorithm Interface" page of the LinearSolve.jl documentation."""

function _interface_error(@nospecialize(alg), what::AbstractString, howto::AbstractString)
    T = alg isa Type ? alg : typeof(alg)
    return ArgumentError(
        "`$(nameof(T))` does not implement `$what`, which the " *
            "`SciMLLinearSolveAlgorithm` interface requires. $howto\n\n" *
            ALGORITHM_INTERFACE_DOCS
    )
end

# Without these fallbacks a type that subtypes `SciMLLinearSolveAlgorithm`
# directly -- rather than one of the categorized abstract types, which carry
# their own trait definitions -- fails with a `MethodError` far from the missing
# definition, or, for `solve!`, recurses without bound through the
# `solve!(cache::LinearCache, args...)` forwarding method (#277).

function SciMLBase.solve!(
        cache::LinearCache, alg::SciMLLinearSolveAlgorithm, args...; kwargs...
    )
    throw(
        _interface_error(
            alg, "SciMLBase.solve!(::LinearCache, ::$(nameof(typeof(alg))))",
            "If this algorithm is provided by a package extension, load its backend " *
                "package (for example `using HYPRE` for `HYPREAlgorithm`) before solving."
        )
    )
end

function needs_concrete_A(alg::SciMLLinearSolveAlgorithm)
    throw(
        _interface_error(
            alg, "LinearSolve.needs_concrete_A",
            "Define `LinearSolve.needs_concrete_A(::$(nameof(typeof(alg)))) = true` if the " *
                "algorithm needs the entries of `A` (for instance to factorize it), or " *
                "`= false` if matrix-vector products suffice. Define it alongside the " *
                "algorithm struct rather than in a package extension: downstream solvers " *
                "query this trait before the backend is necessarily loaded."
        )
    )
end

"""
    LinearSolve.update_tolerances_internal!(cache, alg, abstol, reltol)

Algorithm hook of [`update_tolerances!`](@ref), called after `cache.abstol` and
`cache.reltol` have been set to the new values. `abstol`/`reltol` are the
requested values, either of which may be `nothing` when only the other was
given.

Define it as `nothing` for an algorithm that reads `cache.abstol`/`cache.reltol`
at solve time, and define it to write into `cache.cacheval` for an algorithm
that snapshots the tolerances into its own solver object. The default throws:
an algorithm that never defines it is taken to have no tolerances to update.
"""
function update_tolerances_internal!(
        cache, alg::SciMLLinearSolveAlgorithm, abstol, reltol
    )
    throw(
        ArgumentError(
            "`$(nameof(typeof(alg)))` does not support updating tolerances after `init`. " *
                "Algorithms that read `cache.abstol`/`cache.reltol` at solve time should " *
                "define `LinearSolve.update_tolerances_internal!(cache, " *
                "::$(nameof(typeof(alg))), abstol, reltol) = nothing`; algorithms that " *
                "snapshot the tolerances into `cache.cacheval` should define it to write " *
                "the new values there."
        )
    )
end

# User-supplied solve functions read `cache.abstol`/`cache.reltol` at solve
# time, so the updated `LinearCache` fields are all they need.
update_tolerances_internal!(cache, ::AbstractSolveFunction, abstol, reltol) = nothing

"""
    LinearSolve.concrete_algorithm_types(T = SciMLLinearSolveAlgorithm)

Every loaded concrete subtype of `T`, including those reachable only through
intermediate abstract types. Used to check the algorithm interface across all
algorithms a session knows about, so the result depends on which package
extensions are loaded.
"""
function concrete_algorithm_types(::Type{T} = SciMLLinearSolveAlgorithm) where {T}
    return _collect_concrete_subtypes!(Any[], T)
end

function _collect_concrete_subtypes!(types, @nospecialize(T))
    for S in InteractiveUtils.subtypes(T)
        isabstracttype(S) ? _collect_concrete_subtypes!(types, S) : push!(types, S)
    end
    return types
end

# A method counts as implemented for `sig` when some method other than the
# interface fallback applies to it. `methods` is used rather than `which` so
# that an algorithm dispatching on a type parameter (`DirectLdiv!{true}`) or on
# the cache's matrix type still counts.
function _implements(@nospecialize(f), @nospecialize(sig), @nospecialize(fallback_sig))
    fallback = which(f, fallback_sig)
    return any(m -> m !== fallback, methods(f, sig))
end

# Traits are queried during `init`, so a non-`Bool` return type leaks a dynamic
# dispatch into the rest of the cache construction.
function _returns_bool(@nospecialize(f), @nospecialize(sig))
    rts = Base.return_types(f, sig)
    return !isempty(rts) && all(R -> R === Bool, rts)
end

# Extension modules are top level, so `parentmodule` cannot identify them;
# ask the candidate parent packages instead.
_is_extension_of(m::Module, pkg::Module) = Base.get_extension(pkg, nameof(m)) === m

# A trait defined in a package extension is invisible until the backend loads,
# so callers see the inherited default instead. Only the module that owns the
# algorithm type may define its traits.
function _extension_defining_trait(@nospecialize(f), @nospecialize(sig), ::Type{T}) where {T}
    m = which(f, sig).module
    owner = parentmodule(T)
    m === owner && return nothing
    any(pkg -> _is_extension_of(m, pkg), (owner, LinearSolve)) || return nothing
    return m
end

"""
    LinearSolve.algorithm_interface_issues(alg; check_solve = true) -> Vector{String}

Check `alg` -- an algorithm instance or an algorithm type -- against the
`SciMLLinearSolveAlgorithm` interface and return one message per violation. An
empty result means `alg` is interface compliant.

## Required methods

  - `SciMLBase.solve!(cache::LinearCache, alg::MyAlg; kwargs...)` performs the solve and
    returns `SciMLBase.build_linear_solution(alg, u, resid, cache)`.
  - [`LinearSolve.needs_concrete_A`](@ref)`(alg::MyAlg)::Bool` states whether the algorithm
    needs the entries of `A` or only matrix-vector products. Downstream solvers
    (OrdinaryDiffEq.jl, NonlinearSolve.jl) query it to decide whether to build a concrete
    Jacobian, so it must be defined next to the algorithm struct and never in a package
    extension: it is called before the backend package is necessarily loaded.

Subtyping `AbstractFactorization`, `AbstractSparseFactorization`,
`AbstractKrylovSubspaceMethod` or `AbstractSolveFunction` supplies
`needs_concrete_A`; subtyping `SciMLLinearSolveAlgorithm` directly does not.

## Trait placement

`needs_concrete_A`, `needs_square_A`, `default_alias_A` and `default_alias_b`
must be defined in the module that defines the algorithm type, never in a
package extension. A trait defined in an extension is invisible until the
backend loads, so callers silently get the inherited default instead — that is
reported as a violation too.

## Optional methods

These have defaults, so they are never reported as issues; they are listed here
because they complete the interface:

| Method | Default |
|:------ |:------- |
| `init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assumptions)` | `nothing` |
| `default_alias_A(alg, A, b)` | `false`; `true` for Krylov and sparse factorizations |
| `default_alias_b(alg, A, b)` | `false`; `true` for Krylov and sparse factorizations |
| `needs_square_A(alg)` | `true` |
| `update_tolerances_internal!(cache, alg, abstol, reltol)` | throws: the algorithm has no tolerances to update |

The four `Bool`-valued traits are additionally checked to be inferred as `Bool`,
since `init` calls them while building the cache.

## Keyword arguments

  - `check_solve`: whether to require `SciMLBase.solve!`. Set it to `false` when the
    algorithm's `solve!` lives in a package extension whose backend is not loaded. The
    trait checks still apply in that case, because traits must resolve without the
    backend.

## Example

```julia
struct MyLUFactorization <: LinearSolve.SciMLLinearSolveAlgorithm end
LinearSolve.needs_concrete_A(::MyLUFactorization) = true
function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::MyLUFactorization; kwargs...)
    # ...
end

@test isempty(LinearSolve.algorithm_interface_issues(MyLUFactorization()))
```
"""
function algorithm_interface_issues(@nospecialize(alg); check_solve::Bool = true)
    T = alg isa Type ? alg : typeof(alg)
    T <: SciMLLinearSolveAlgorithm || throw(
        ArgumentError(
            "`algorithm_interface_issues` expects a `SciMLLinearSolveAlgorithm` or one " *
                "of its subtypes, got `$T`."
        )
    )

    issues = String[]

    if check_solve && !_implements(
            SciMLBase.solve!, Tuple{LinearCache, T},
            Tuple{LinearCache, SciMLLinearSolveAlgorithm}
        )
        push!(issues, "$T does not implement `SciMLBase.solve!(::LinearCache, ::$T)`.")
    end

    has_needs_concrete_A = _implements(
        needs_concrete_A, Tuple{T}, Tuple{SciMLLinearSolveAlgorithm}
    )
    has_needs_concrete_A || push!(
        issues,
        "$T does not implement `LinearSolve.needs_concrete_A(::$T)`. Define it next to " *
            "the algorithm struct, not in a package extension."
    )

    traits = has_needs_concrete_A ?
        ((needs_concrete_A, Tuple{T}), (needs_square_A, Tuple{T})) :
        ((needs_square_A, Tuple{T}),)
    for (trait, sig) in (
            traits..., (default_alias_A, Tuple{T, Any, Any}),
            (default_alias_b, Tuple{T, Any, Any}),
        )
        _returns_bool(trait, sig) || push!(
            issues, "`LinearSolve.$(nameof(trait))` is not inferred as `Bool` for $T."
        )
        ext = _extension_defining_trait(trait, sig, T)
        ext === nothing || push!(
            issues,
            "`LinearSolve.$(nameof(trait))` for $T is defined in the package extension " *
                "$ext. Traits are queried before the backend is loaded, so they must be " *
                "defined in $(parentmodule(T)), where the algorithm type lives."
        )
    end

    return issues
end
