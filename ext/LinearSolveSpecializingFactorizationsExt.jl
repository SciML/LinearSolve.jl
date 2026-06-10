module LinearSolveSpecializingFactorizationsExt

using LinearSolve, LinearAlgebra
using LinearSolve: LinearCache, OperatorAssumptions, LinearVerbosity, @get_cacheval
using SciMLBase: SciMLBase, ReturnCode
using SpecializingFactorizations: specializinglu, specializinglu!,
    specializingqr, specializingqr!

function LinearSolve.init_cacheval(
        ::SpecializedLUFactorization, A, b, u, Pl, Pr, maxiters::Int, abstol,
        reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    A isa AbstractMatrix || return nothing
    return specializinglu(convert(AbstractMatrix, A))
end

function SciMLBase.solve!(
        cache::LinearCache, alg::SpecializedLUFactorization; kwargs...
    )
    A = convert(AbstractMatrix, cache.A)
    F = @get_cacheval(cache, :SpecializedLUFactorization)
    if cache.isfresh
        if F === nothing
            F = specializinglu(A)
        else
            specializinglu!(F, A)
        end
        cache.cacheval = F
        cache.isfresh = false
    end
    F = @get_cacheval(cache, :SpecializedLUFactorization)
    if !LinearAlgebra.issuccess(F)
        return SciMLBase.build_linear_solution(
            alg, cache.u, nothing, cache; retcode = ReturnCode.Failure
        )
    end
    ldiv!(cache.u, F, cache.b)
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache; retcode = ReturnCode.Success
    )
end

function LinearSolve.init_cacheval(
        ::SpecializedQRFactorization, A, b, u, Pl, Pr, maxiters::Int, abstol,
        reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    A isa AbstractMatrix || return nothing
    return specializingqr(convert(AbstractMatrix, A))
end

function SciMLBase.solve!(
        cache::LinearCache, alg::SpecializedQRFactorization; kwargs...
    )
    A = convert(AbstractMatrix, cache.A)
    F = @get_cacheval(cache, :SpecializedQRFactorization)
    if cache.isfresh
        if F === nothing
            F = specializingqr(A)
        else
            specializingqr!(F, A)
        end
        cache.cacheval = F
        cache.isfresh = false
    end
    F = @get_cacheval(cache, :SpecializedQRFactorization)
    # SpecializedQR is rank-safe and never reports failure for rank deficiency;
    # issuccess only guards genuine numerical breakdown.
    if !LinearAlgebra.issuccess(F)
        return SciMLBase.build_linear_solution(
            alg, cache.u, nothing, cache; retcode = ReturnCode.Failure
        )
    end
    ldiv!(cache.u, F, cache.b)
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache; retcode = ReturnCode.Success
    )
end

end
