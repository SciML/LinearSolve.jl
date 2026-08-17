"""
    LowRankUpdatedMatrix(A, U, V; C = I)

The matrix ``A + U C V^{*}``, held in that form rather than assembled.

This is a problem-side type: pass it as the matrix of a `LinearProblem` and solve with
whichever factorization suits `A`. The solve factorizes `A` once and applies the Woodbury
identity

```math
(A + U C V^{*})^{-1} = A^{-1} - A^{-1} U (C^{-1} + V^{*} A^{-1} U)^{-1} V^{*} A^{-1}
```

so a low-rank change costs a `k * k` factorization rather than a fresh one of the whole
matrix, where `k` is the rank of the update. It matters most when `A` factorizes cheaply
and the update does not preserve that structure: a dense rank-1 update to a sparse `A`
would otherwise assemble to a dense matrix.

`U` is `n * k` and `V` is `n * k`, with `C` a `k * k` middle factor defaulting to `I`,
which gives the Sherman-Morrison form ``A + U V^{*}``. A vector `U` or `V` is treated as a
rank-1 update.

```julia
A = spdiagm(-1 => -ones(n - 1), 0 => 4ones(n), 1 => -ones(n - 1))
u = rand(n)
v = rand(n)
sol = solve(LinearProblem(LowRankUpdatedMatrix(A, u, v), b))
```

!!! note

    The identity needs `A` and the capacitance matrix ``C^{-1} + V^{*} A^{-1} U`` to be
    nonsingular. An update that makes the whole matrix singular surfaces as a singular
    capacitance matrix, and the solve reports failure rather than returning a wrong answer.
"""
struct LowRankUpdatedMatrix{T, TA, TU, TV, TC} <: AbstractMatrix{T}
    A::TA
    U::TU
    V::TV
    C::TC
end

# `U` and `V` are kept as matrices so the capacitance system is always `k * k`.
_lowrank_factor(M::AbstractMatrix) = M
_lowrank_factor(v::AbstractVector) = reshape(v, length(v), 1)

function LowRankUpdatedMatrix(A, U, V; C = I)
    Um = _lowrank_factor(U)
    Vm = _lowrank_factor(V)
    size(Um, 1) == size(A, 1) ||
        throw(DimensionMismatch("`U` has $(size(Um, 1)) rows, `A` has $(size(A, 1))"))
    size(Vm, 1) == size(A, 2) ||
        throw(DimensionMismatch("`V` has $(size(Vm, 1)) rows, `A` has $(size(A, 2)) columns"))
    size(Um, 2) == size(Vm, 2) ||
        throw(DimensionMismatch("`U` and `V` disagree on the rank of the update"))
    T = promote_type(eltype(A), eltype(Um), eltype(Vm), C === I ? Bool : eltype(C))
    return LowRankUpdatedMatrix{T, typeof(A), typeof(Um), typeof(Vm), typeof(C)}(
        A, Um, Vm, C
    )
end

Base.size(M::LowRankUpdatedMatrix) = size(M.A)
Base.size(M::LowRankUpdatedMatrix, i::Integer) = size(M.A, i)

function Base.getindex(M::LowRankUpdatedMatrix, i::Integer, j::Integer)
    upd = M.C === I ? dot(view(M.U, i, :), view(M.V, j, :)) :
        dot(view(M.U, i, :), M.C * view(M.V, j, :))
    return M.A[i, j] + upd
end

function LinearAlgebra.mul!(y::AbstractVector, M::LowRankUpdatedMatrix, x::AbstractVector)
    mul!(y, M.A, x)
    Vx = M.V' * x
    return mul!(y, M.U, M.C === I ? Vx : M.C * Vx, true, true)
end

Base.:*(M::LowRankUpdatedMatrix, x::AbstractVector) =
    mul!(similar(x, promote_type(eltype(M), eltype(x)), size(M, 1)), M, x)

"""
    LowRankUpdatedCache

The factorization of the base matrix, `A \\ U`, and the factorized capacitance matrix
`C^{-1} + V^{*} A^{-1} U`.
"""
mutable struct LowRankUpdatedCache{F, TAiU, TCap}
    fact::F
    AiU::TAiU
    capfact::TCap
end

_lowrank_capacitance(C, VAiU) = lu((C === I ? VAiU + I : inv(C) + VAiU); check = false)

function init_cacheval(
        alg::AbstractFactorization, M::LowRankUpdatedMatrix, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    # Instances rather than real work, so `cacheval` is typed as `solve!` will store it.
    inner = init_cacheval(
        alg, M.A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assumptions
    )
    T = eltype(u)
    k = size(M.U, 2)
    return LowRankUpdatedCache(
        inner, similar(M.U, T, size(M.U, 1), k), lu(Matrix{T}(I, k, k); check = false)
    )
end

function SciMLBase.solve!(
        cache::LinearCache{<:LowRankUpdatedMatrix}, alg::AbstractFactorization; kwargs...
    )
    M = cache.A
    if cache.isfresh
        # `do_factorization` may consume its argument, and the correction needs `A` only
        # through the factorization, so hand it a copy unless the caller allows otherwise.
        A = cache.alias_A ? M.A : copy(M.A)
        fact = do_factorization(alg, A, cache.b, cache.u)
        if _notsuccessful(fact)
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, cache; retcode = ReturnCode.Failure
            )
        end
        AiU = fact \ Matrix(M.U)
        cache.cacheval = LowRankUpdatedCache(
            fact, AiU, _lowrank_capacitance(M.C, M.V' * AiU)
        )
        cache.isfresh = false
    end

    cacheval = cache.cacheval
    if !issuccess(cacheval.capfact)
        return SciMLBase.build_linear_solution(
            alg, cache.u, nothing, cache; retcode = ReturnCode.Failure
        )
    end

    y = cacheval.fact \ cache.b
    cache.u .= y .- cacheval.AiU * (cacheval.capfact \ (M.V' * y))
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache; retcode = ReturnCode.Success
    )
end

# The update is dense even when `A` is not, so a factorization of `A` plus the identity
# beats anything that would assemble the sum.
defaultalg(::LowRankUpdatedMatrix, b, ::OperatorAssumptions{Bool}) = LUFactorization()
