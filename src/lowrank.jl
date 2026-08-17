# Not an `AbstractDenseFactorization`: this wraps another algorithm and carries the
# update, so it has no zero-argument form, while the factorization subtypes are swept
# and constructed generically (`test/Core/resolve.jl`).
"""
    LowRankUpdatedFactorization(alg = LUFactorization(); U, V, C = I)

Solve a system whose matrix is a low-rank update of one that is cheap to keep
factorized:

```math
(A + U C V^{*}) x = b
```

`A` is factorized once by `alg` and the update is applied through the Woodbury
identity

```math
(A + U C V^{*})^{-1} = A^{-1} - A^{-1} U (C^{-1} + V^{*} A^{-1} U)^{-1} V^{*} A^{-1}
```

so changing `U`, `V` or `C` costs a `k * k` factorization rather than a fresh
factorization of the full matrix, where `k` is the rank of the update. Pass the
base matrix `A` to the `LinearProblem`; the update is carried by the algorithm.

`U` is `n * k`, `V` is `n * k`, and `C` is `k * k` (the default `I` gives the
Sherman-Morrison form `A + U V^{*}`). A vector `U` or `V` is treated as a rank-1
update.

!!! note

    The identity requires that `A` and the capacitance matrix
    `C^{-1} + V^{*} A^{-1} U` are both nonsingular. A rank-1 update that makes the
    full matrix singular shows up as a singular capacitance matrix rather than as
    a failure of the outer solve.
"""
struct LowRankUpdatedFactorization{Alg, TU, TV, TC} <: SciMLLinearSolveAlgorithm
    alg::Alg
    U::TU
    V::TV
    C::TC
end

function LowRankUpdatedFactorization(
        alg::AbstractFactorization = LUFactorization(); U, V, C = I
    )
    return LowRankUpdatedFactorization(alg, U, V, C)
end

needs_concrete_A(::LowRankUpdatedFactorization) = true

# `U` and `V` are stored as matrices so the capacitance system is always `k * k`.
_lowrank_factor(M::AbstractMatrix) = M
_lowrank_factor(v::AbstractVector) = reshape(v, length(v), 1)

"""
    LowRankUpdatedCache

Holds the factorization of the base matrix, `A \\ U`, and the factorized
capacitance matrix `C^{-1} + V^{*} A^{-1} U`.
"""
mutable struct LowRankUpdatedCache{F, TAiU, TCap, TV}
    fact::F
    AiU::TAiU
    capfact::TCap
    V::TV
end

function init_cacheval(
        alg::LowRankUpdatedFactorization, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    # Instances rather than real work, so `cacheval` is typed as `solve!` will store it.
    U = _lowrank_factor(alg.U)
    V = _lowrank_factor(alg.V)
    fact = init_cacheval(
        alg.alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assumptions
    )
    T = eltype(u)
    k = size(U, 2)
    return LowRankUpdatedCache(
        fact, similar(U, T, size(U, 1), k), lu(Matrix{T}(I, k, k); check = false), V
    )
end

function _capacitance(alg::LowRankUpdatedFactorization, AiU, V)
    VAiU = V' * AiU
    cap = alg.C === I ? VAiU + I : inv(alg.C) + VAiU
    return lu(cap; check = false)
end

function SciMLBase.solve!(
        cache::LinearCache, alg::LowRankUpdatedFactorization; kwargs...
    )
    U = _lowrank_factor(alg.U)
    V = _lowrank_factor(alg.V)

    if cache.isfresh
        A = convert(AbstractMatrix, cache.A)
        # `do_factorization` may overwrite its argument, and the Woodbury correction
        # needs `A` only through the factorization, so hand it a copy unless the
        # caller has allowed `A` to be consumed.
        fact = do_factorization(alg.alg, cache.alias_A ? A : copy(A), cache.b, cache.u)
        if _notsuccessful(fact)
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, cache; retcode = ReturnCode.Failure
            )
        end
        AiU = fact \ U
        cache.cacheval = LowRankUpdatedCache(fact, AiU, _capacitance(alg, AiU, V), V)
        cache.isfresh = false
    end

    cacheval = cache.cacheval
    if !issuccess(cacheval.capfact)
        return SciMLBase.build_linear_solution(
            alg, cache.u, nothing, cache; retcode = ReturnCode.Failure
        )
    end

    y = cacheval.fact \ cache.b
    # x = y - AiU * (capacitance \ (V' * y))
    cache.u .= y .- cacheval.AiU * (cacheval.capfact \ (V' * y))
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache; retcode = ReturnCode.Success
    )
end
