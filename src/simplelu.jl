## From https://github.com/JuliaGNI/SimpleSolvers.jl/blob/master/src/linear/lu_solver.jl

mutable struct LUSolver{T}
    n::Int
    A::Matrix{T}
    b::Vector{T}
    x::Vector{T}
    pivots::Vector{Int}
    perms::Vector{Int}
    info::Int

    function LUSolver{T}(n) where {T}
        new(n, zeros(T, n, n), zeros(T, n), zeros(T, n), zeros(Int, n), zeros(Int, n), 0)
    end
end

function LUSolver(A::Matrix{T}) where {T}
    n = LinearAlgebra.checksquare(A)
    lu = LUSolver{eltype(A)}(n)
    lu.A .= A
    lu
end

function LUSolver(A::Matrix{T}, b::Vector{T}) where {T}
    n = LinearAlgebra.checksquare(A)
    @assert n == length(b)
    lu = LUSolver{eltype(A)}(n)
    lu.A .= A
    lu.b .= b
    lu
end

function simplelu_factorize!(lu::LUSolver{T}, pivot = true) where {T}
    A = lu.A

    begin
        @inbounds for i in eachindex(lu.perms)
            lu.perms[i] = i
        end

        @inbounds for k in 1:(lu.n)
            # find index max
            kp = k
            if pivot
                amax = real(zero(T))
                for i in k:(lu.n)
                    absi = abs(A[i, k])
                    if absi > amax
                        kp = i
                        amax = absi
                    end
                end
            end
            lu.pivots[k] = kp
            lu.perms[k], lu.perms[kp] = lu.perms[kp], lu.perms[k]

            if A[kp, k] != 0
                if k != kp
                    # Interchange
                    for i in 1:(lu.n)
                        tmp = A[k, i]
                        A[k, i] = A[kp, i]
                        A[kp, i] = tmp
                    end
                end
                # Scale first column
                Akkinv = inv(A[k, k])
                for i in (k + 1):(lu.n)
                    A[i, k] *= Akkinv
                end
            elseif lu.info == 0
                lu.info = k
            end
            # Update the rest
            for j in (k + 1):(lu.n)
                for i in (k + 1):(lu.n)
                    A[i, j] -= A[i, k] * A[k, j]
                end
            end
        end

        lu.info
    end
end

function simplelu_solve!(lu::LUSolver{T}) where {T}
    @inbounds for i in 1:(lu.n)
        lu.x[i] = lu.b[lu.perms[i]]
    end

    @inbounds for i in 2:(lu.n)
        s = zero(T)
        for j in 1:(i - 1)
            s += lu.A[i, j] * lu.x[j]
        end
        lu.x[i] -= s
    end

    lu.x[lu.n] /= lu.A[lu.n, lu.n]
    @inbounds for i in (lu.n - 1):-1:1
        s = zero(T)
        for j in (i + 1):(lu.n)
            s += lu.A[i, j] * lu.x[j]
        end
        lu.x[i] -= s
        lu.x[i] /= lu.A[i, i]
    end

    copyto!(lu.b, lu.x)

    lu.x
end

### Wrapper

"""
`SimpleLUFactorization(pivot::Bool = true)`

A simple LU-factorization implementation without BLAS. Fast for small matrices.

## Positional Arguments

  - pivot::Bool: whether to perform pivoting. Defaults to `true`
"""
struct SimpleLUFactorization <: AbstractFactorization
    pivot::Bool
    SimpleLUFactorization(pivot = true) = new(pivot)
end

default_alias_A(::SimpleLUFactorization, ::Any, ::Any) = true
default_alias_b(::SimpleLUFactorization, ::Any, ::Any) = true

function SciMLBase.solve!(cache::LinearCache, alg::SimpleLUFactorization; kwargs...)
    if cache.isfresh
        cache.cacheval.A .= cache.A
        simplelu_factorize!(cache.cacheval, alg.pivot)
        cache.isfresh = false
    end
    cache.cacheval.b .= cache.b
    cache.cacheval.x .= cache.u
    y = simplelu_solve!(cache.cacheval)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

function init_cacheval(alg::SimpleLUFactorization, A, b, u, Pl, Pr, maxiters::Int, abstol,
        reltol, verbose::Bool, assumptions::OperatorAssumptions)
    LUSolver(convert(AbstractMatrix, A))
end
