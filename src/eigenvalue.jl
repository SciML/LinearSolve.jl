"""
    EigenvalueProblem(A[, B]; nev = nothing, which = :LM, sigma = nothing, u0 = nothing, p = SciMLBase.NullParameters())

Define a standard or generalized eigenvalue problem.

The standard problem is ``A v = lambda v``. If `B` is supplied, the generalized
problem is ``A v = lambda B v``. `nev` requests a subset of eigenpairs, `which`
selects the part of the spectrum (`:LM`, `:SM`, `:LR`, `:SR`, `:LI`, `:SI`),
and `sigma` requests eigenvalues nearest to the shift.
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
        nev = nothing, which = :LM, sigma = nothing, u0 = nothing, kwargs...
    )
    return EigenvalueProblem{
        typeof(A), typeof(B), typeof(nev), typeof(which), typeof(sigma),
        typeof(u0), typeof(p), typeof(kwargs),
    }(A, B, nev, which, sigma, u0, p, kwargs)
end

abstract type AbstractEigenvalueAlgorithm <: SciMLBase.AbstractLinearAlgorithm end

struct DenseEigen <: AbstractEigenvalueAlgorithm end

struct ArpackJL{A, K} <: AbstractEigenvalueAlgorithm
    args::A
    kwargs::K
end
ArpackJL(args...; kwargs...) = ArpackJL(args, kwargs)

struct ArnoldiMethodJL{A, K} <: AbstractEigenvalueAlgorithm
    args::A
    kwargs::K
end
ArnoldiMethod(args...; kwargs...) = ArnoldiMethodJL(args, kwargs)

struct KrylovKitEigen{A, K} <: AbstractEigenvalueAlgorithm
    args::A
    kwargs::K
end
KrylovKitEigen(args...; kwargs...) = KrylovKitEigen(args, kwargs)

struct JacobiDavidsonJL{A, K} <: AbstractEigenvalueAlgorithm
    args::A
    kwargs::K
end
JacobiDavidsonJL(args...; kwargs...) = JacobiDavidsonJL(args, kwargs)

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

function _eigenvalue_order(values, which, sigma)
    if sigma !== nothing
        return sortperm(abs.(values .- sigma))
    elseif which === :LM
        return sortperm(abs.(values); rev = true)
    elseif which === :SM
        return sortperm(abs.(values))
    elseif which === :LR
        return sortperm(real.(values); rev = true)
    elseif which === :SR
        return sortperm(real.(values))
    elseif which === :LI
        return sortperm(imag.(values); rev = true)
    elseif which === :SI
        return sortperm(imag.(values))
    else
        throw(ArgumentError("unsupported eigenvalue selector `which = $which`; expected one of :LM, :SM, :LR, :SR, :LI, or :SI"))
    end
end
