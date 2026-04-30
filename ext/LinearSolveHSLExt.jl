module LinearSolveHSLExt

using HSL
using LinearSolve
using LinearSolve: LinearVerbosity, OperatorAssumptions
using SciMLBase: SciMLBase, ReturnCode
using SciMLLogging: @SciMLMessage
using SparseArrays: AbstractSparseMatrixCSC, SparseMatrixCSC
using LinearAlgebra: ishermitian, issymmetric

mutable struct HSLMA57Cache{M}
    ma57::M
end

mutable struct HSLMA97Cache{M}
    ma97::M
end

function _is_symmetric_like(A)
    T = eltype(A)
    if T <: Real
        return issymmetric(A)
    else
        return ishermitian(A)
    end
end

function _as_int_csc(A::SparseMatrixCSC{T, Int}) where {T}
    return A
end

function _as_int_csc(A::AbstractSparseMatrixCSC{T}) where {T}
    return SparseMatrixCSC{T, Int}(A)
end

function LinearSolve.init_cacheval(
        alg::LinearSolve.HSLMA57Factorization,
        A::AbstractSparseMatrixCSC{<:Union{Float32, Float64}}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return HSLMA57Cache(HSL.Ma57(A; alg.kwargs...))
end

function LinearSolve.init_cacheval(
        alg::LinearSolve.HSLMA57Factorization,
        A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

function LinearSolve.init_cacheval(
        alg::LinearSolve.HSLMA97Factorization,
        A::AbstractSparseMatrixCSC{<:Union{Float32, Float64, ComplexF32, ComplexF64}}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    A_int = _as_int_csc(A)
    return HSLMA97Cache(HSL.Ma97(A_int; alg.kwargs...))
end

function LinearSolve.init_cacheval(
        alg::LinearSolve.HSLMA97Factorization,
        A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache,
        alg::LinearSolve.HSLMA57Factorization;
        kwargs...
    )
    A = convert(AbstractMatrix, cache.A)
    A isa AbstractSparseMatrixCSC ||
        error("HSLMA57Factorization currently supports only sparse CSC matrices")
    size(A, 1) == size(A, 2) ||
        error("HSLMA57Factorization requires a square matrix")
    _is_symmetric_like(A) ||
        error("HSLMA57Factorization requires a symmetric/Hermitian matrix")

    hcache = LinearSolve.@get_cacheval(cache, :HSLMA57Factorization)
    hcache === nothing && error(
        "HSLMA57Factorization supports `AbstractSparseMatrixCSC{<:Union{Float32, Float64}}`"
    )

    try
        if cache.isfresh
            hcache.ma57 = HSL.Ma57(A; alg.kwargs...)
            HSL.ma57_factorize!(hcache.ma57)
            cache.isfresh = false
        end

        cache.u .= HSL.ma57_solve(hcache.ma57, cache.b)

        return SciMLBase.build_linear_solution(
            alg, cache.u, nothing, cache; retcode = ReturnCode.Success
        )
    catch err
        if err isa HSL.Ma57Exception
            @SciMLMessage(
                "MA57 failed: $(err.msg) (flag $(err.flag))",
                cache.verbose,
                :solver_failure
            )
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, cache; retcode = ReturnCode.Failure
            )
        end
        rethrow(err)
    end
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache,
        alg::LinearSolve.HSLMA97Factorization;
        kwargs...
    )
    A = convert(AbstractMatrix, cache.A)
    A isa AbstractSparseMatrixCSC ||
        error("HSLMA97Factorization currently supports only sparse CSC matrices")
    size(A, 1) == size(A, 2) ||
        error("HSLMA97Factorization requires a square matrix")
    _is_symmetric_like(A) ||
        error("HSLMA97Factorization requires a symmetric/Hermitian matrix")

    hcache = LinearSolve.@get_cacheval(cache, :HSLMA97Factorization)
    hcache === nothing && error(
        "HSLMA97Factorization supports `AbstractSparseMatrixCSC{<:Union{Float32, Float64, ComplexF32, ComplexF64}}`"
    )

    try
        if cache.isfresh
            A_int = _as_int_csc(A)
            hcache.ma97 = HSL.Ma97(A_int; alg.kwargs...)
            HSL.ma97_factorize!(hcache.ma97, matrix_type = alg.matrix_type)
            cache.isfresh = false
        end

        cache.u .= HSL.ma97_solve(hcache.ma97, cache.b)

        return SciMLBase.build_linear_solution(
            alg, cache.u, nothing, cache; retcode = ReturnCode.Success
        )
    catch err
        if err isa HSL.Ma97Exception
            @SciMLMessage(
                "MA97 failed: $(err.msg) (flag $(err.flag))",
                cache.verbose,
                :solver_failure
            )
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, cache; retcode = ReturnCode.Failure
            )
        end
        rethrow(err)
    end
end

end
