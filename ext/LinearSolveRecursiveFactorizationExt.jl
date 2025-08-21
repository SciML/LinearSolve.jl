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
        maxiters::Int, abstol, reltol, verbose::LinearSolve.LinearVerbosity,
        assumptions::LinearSolve.OperatorAssumptions) where {P, T}
    # Pre-allocate appropriate 32-bit arrays based on input type
    if eltype(A) <: Complex
        A_32 = rand(ComplexF32, 0, 0)
    else
        A_32 = rand(Float32, 0, 0)
    end
    luinst = ArrayInterface.lu_instance(A_32)
    (luinst, Vector{LinearAlgebra.BlasInt}(undef, min(size(A)...)))
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::RF32MixedLUFactorization{P, T};
        kwargs...) where {P, T}
    A = cache.A
    A = convert(AbstractMatrix, A)

    # Check if we have complex numbers
    iscomplex = eltype(A) <: Complex

    if cache.isfresh
        fact, ipiv = LinearSolve.@get_cacheval(cache, :RF32MixedLUFactorization)

        # Convert to appropriate 32-bit type for factorization
        if iscomplex
            A_f32 = ComplexF32.(A)
        else
            A_f32 = Float32.(A)
        end

        # Ensure ipiv is the right size
        if length(ipiv) != min(size(A_f32)...)
            ipiv = Vector{LinearAlgebra.BlasInt}(undef, min(size(A_f32)...))
        end

        fact = RecursiveFactorization.lu!(A_f32, ipiv, Val(P), Val(T), check = false)
        cache.cacheval = (fact, ipiv)

        if !LinearAlgebra.issuccess(fact)
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, cache; retcode = ReturnCode.Failure)
        end

        cache.isfresh = false
    end

    fact, ipiv = LinearSolve.@get_cacheval(cache, :RF32MixedLUFactorization)

    # Convert b to appropriate 32-bit type for solving
    if iscomplex
        b_f32 = ComplexF32.(cache.b)
    else
        b_f32 = Float32.(cache.b)
    end

    # Solve in 32-bit precision
    u_f32 = similar(b_f32)
    ldiv!(u_f32, fact, b_f32)

    # Convert back to original precision
    T_orig = eltype(cache.u)
    cache.u .= T_orig.(u_f32)

    SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache; retcode = ReturnCode.Success)
end

end
