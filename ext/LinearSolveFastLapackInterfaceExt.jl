module LinearSolveFastLapackInterfaceExt

using LinearSolve, LinearAlgebra
using FastLapackInterface

struct WorkspaceAndFactors{W, F}
    workspace::W
    factors::F
end

function LinearSolve.init_cacheval(::FastLUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    ws = LUWs(A)
    return WorkspaceAndFactors(
        ws, LinearSolve.ArrayInterface.lu_instance(convert(AbstractMatrix, A)))
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::FastLUFactorization; kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    ws_and_fact = LinearSolve.@get_cacheval(cache, :FastLUFactorization)
    if cache.isfresh
        # we will fail here if A is a different *size* than in a previous version of the same cache.
        # it may instead be desirable to resize the workspace.
        LinearSolve.@set! ws_and_fact.factors = LinearAlgebra.LU(LAPACK.getrf!(
            ws_and_fact.workspace,
            A)...)
        cache.cacheval = ws_and_fact
        cache.isfresh = false
    end
    y = ldiv!(cache.u, cache.cacheval.factors, cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

function LinearSolve.init_cacheval(
        alg::FastQRFactorization{NoPivot}, A::AbstractMatrix, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    ws = QRWYWs(A; blocksize = alg.blocksize)
    return WorkspaceAndFactors(ws,
        LinearSolve.ArrayInterface.qr_instance(convert(AbstractMatrix, A)))
end
function LinearSolve.init_cacheval(
        ::FastQRFactorization{ColumnNorm}, A::AbstractMatrix, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    ws = QRpWs(A)
    return WorkspaceAndFactors(ws,
        LinearSolve.ArrayInterface.qr_instance(convert(AbstractMatrix, A)))
end

function LinearSolve.init_cacheval(alg::FastQRFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    return init_cacheval(alg, convert(AbstractMatrix, A), b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::FastQRFactorization{P};
        kwargs...) where {P}
    A = cache.A
    A = convert(AbstractMatrix, A)
    ws_and_fact = LinearSolve.@get_cacheval(cache, :FastQRFactorization)
    if cache.isfresh
        # we will fail here if A is a different *size* than in a previous version of the same cache.
        # it may instead be desirable to resize the workspace.
        if P === NoPivot
            LinearSolve.@set! ws_and_fact.factors = LinearAlgebra.QRCompactWY(LAPACK.geqrt!(
                ws_and_fact.workspace,
                A)...)
        else
            LinearSolve.@set! ws_and_fact.factors = LinearAlgebra.QRPivoted(LAPACK.geqp3!(
                ws_and_fact.workspace,
                A)...)
        end
        cache.cacheval = ws_and_fact
        cache.isfresh = false
    end
    y = ldiv!(cache.u, cache.cacheval.factors, cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

end
