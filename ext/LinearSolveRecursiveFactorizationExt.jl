module LinearSolveRecursiveFactorizationExt

using LinearSolve: LinearSolve, userecursivefactorization, LinearCache, @get_cacheval,
                   RFLUFactorization, ButterflyFactorization, RF32MixedLUFactorization, default_alias_A,
                   default_alias_b
using LinearSolve.LinearAlgebra, LinearSolve.ArrayInterface, RecursiveFactorization
using SciMLBase: SciMLBase, ReturnCode

LinearSolve.userecursivefactorization(A::Union{Nothing, AbstractMatrix}) = true

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::RFLUFactorization{P, T};
        kwargs...) where {P, T}
    A = cache.A
    A = convert(AbstractMatrix, A)
    fact, ipiv = LinearSolve.@get_cacheval(cache, :RFLUFactorization)
    if cache.isfresh
        if length(ipiv) != min(size(A)...)
            ipiv = Vector{LinearAlgebra.BlasInt}(undef, min(size(A)...))
        end
        fact = RecursiveFactorization.lu!(A, ipiv, Val(P), Val(T), check = false)
        cache.cacheval = (fact, ipiv)
        if !LinearAlgebra.issuccess(fact)
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, cache; retcode = ReturnCode.Failure)
        end

        cache.isfresh = false
    end
    y = ldiv!(cache.u, LinearSolve.@get_cacheval(cache, :RFLUFactorization)[1], cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache; retcode = ReturnCode.Success)
end

# Mixed precision RecursiveFactorization implementation
LinearSolve.default_alias_A(::RF32MixedLUFactorization, ::Any, ::Any) = false
LinearSolve.default_alias_b(::RF32MixedLUFactorization, ::Any, ::Any) = false

const PREALLOCATED_RF32_LU = begin
    A = rand(Float32, 0, 0)
    luinst = ArrayInterface.lu_instance(A)
    (luinst, Vector{LinearAlgebra.BlasInt}(undef, 0))
end

function LinearSolve.init_cacheval(alg::RF32MixedLUFactorization{P, T}, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::LinearSolve.OperatorAssumptions) where {P, T}
    # Pre-allocate appropriate 32-bit arrays based on input type
    m, n = size(A)
    T32 = eltype(A) <: Complex ? ComplexF32 : Float32
    A_32 = similar(A, T32)
    b_32 = similar(b, T32)
    u_32 = similar(u, T32)
    luinst = ArrayInterface.lu_instance(rand(T32, 0, 0))
    ipiv = Vector{LinearAlgebra.BlasInt}(undef, min(m, n))
    # Return tuple with pre-allocated arrays
    (luinst, ipiv, A_32, b_32, u_32)
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::RF32MixedLUFactorization{P, T};
        kwargs...) where {P, T}
    A = cache.A
    A = convert(AbstractMatrix, A)

    if cache.isfresh
        # Get pre-allocated arrays from cacheval
        luinst, ipiv, A_32, b_32, u_32 = LinearSolve.@get_cacheval(cache, :RF32MixedLUFactorization)
        # Compute 32-bit type on demand and copy A
        T32 = eltype(A) <: Complex ? ComplexF32 : Float32
        A_32 .= T32.(A)

        # Ensure ipiv is the right size
        if length(ipiv) != min(size(A_32)...)
            resize!(ipiv, min(size(A_32)...))
        end

        fact = RecursiveFactorization.lu!(A_32, ipiv, Val(P), Val(T), check = false)
        cache.cacheval = (fact, ipiv, A_32, b_32, u_32)

        if !LinearAlgebra.issuccess(fact)
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, cache; retcode = ReturnCode.Failure)
        end

        cache.isfresh = false
    end

    # Get the factorization and pre-allocated arrays from the cache
    fact_cached, ipiv, A_32, b_32, u_32 = LinearSolve.@get_cacheval(cache, :RF32MixedLUFactorization)
    
    # Compute types on demand for conversions
    T32 = eltype(cache.A) <: Complex ? ComplexF32 : Float32
    Torig = eltype(cache.u)
    
    # Copy b to pre-allocated 32-bit array
    b_32 .= T32.(cache.b)

    # Solve in 32-bit precision
    ldiv!(u_32, fact_cached, b_32)

    # Convert back to original precision
    cache.u .= Torig.(u_32)

    SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache; retcode = ReturnCode.Success)
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::ButterflyFactorization;
        kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    b = cache.b
    M, N = size(A)
    if cache.isfresh
        @assert M==N "A must be square"
        ws = RecursiveFactorization.🦋workspace(A, b)    
        cache.cacheval = (ws)
        cache.isfresh = false
    end
    out = RecursiveFactorization.🦋lu!(ws, M, alg.thread)
    SciMLBase.build_linear_solution(alg, out, nothing, cache)
end

function LinearSolve.init_cacheval(alg::ButterflyFactorization, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Bool, assumptions::LinearSolve.OperatorAssumptions)
    ws = RecursiveFactorization.🦋workspace(A, b)    
end

end

