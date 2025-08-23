module LinearSolveRecursiveFactorizationExt

using LinearSolve: LinearSolve, userecursivefactorization, LinearCache, @get_cacheval,
                   RFLUFactorization, RF32MixedLUFactorization, default_alias_A,
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
    if eltype(A) <: Complex
        A_32 = similar(A, ComplexF32)
        b_32 = similar(b, ComplexF32)
        u_32 = similar(u, ComplexF32)
    else
        A_32 = similar(A, Float32)
        b_32 = similar(b, Float32)
        u_32 = similar(u, Float32)
    end
    luinst = ArrayInterface.lu_instance(rand(eltype(A_32), 0, 0))
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
        # Copy A to pre-allocated 32-bit array
        A_32 .= eltype(A_32).(A)

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
    
    # Copy b to pre-allocated 32-bit array
    b_32 .= eltype(b_32).(cache.b)

    # Solve in 32-bit precision
    ldiv!(u_32, fact_cached, b_32)

    # Convert back to original precision
    T_orig = eltype(cache.u)
    cache.u .= T_orig.(u_32)

    SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache; retcode = ReturnCode.Success)
end

end
