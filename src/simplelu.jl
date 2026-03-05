## From https://github.com/JuliaGNI/SimpleSolvers.jl/blob/master/src/linear/lu_solver.jl

"""
    LUSolver{T}

A mutable workspace for performing LU factorization and solving linear systems.
This struct maintains all necessary arrays and state information for the 
factorization and solve phases, allowing for efficient reuse when solving
multiple systems with the same matrix structure.

## Fields
- `n::Int`: Dimension of the square matrix
- `A::Matrix{T}`: Working copy of the matrix to be factorized (modified in-place)
- `b::Vector{T}`: Right-hand side vector storage
- `x::Vector{T}`: Solution vector storage  
- `pivots::Vector{Int}`: Pivot indices from the factorization
- `perms::Vector{Int}`: Permutation vector tracking row exchanges
- `info::Int`: Status information (0 = success, >0 indicates singularity)

## Constructor
```julia
LUSolver{T}(n)  # Create solver for n×n matrix with element type T
```

## Usage
The solver is typically created from a matrix using the convenience constructors:
```julia
solver = LUSolver(A)        # From matrix A
solver = LUSolver(A, b)     # From matrix A and RHS b
```

Then factorized and solved:
```julia
simplelu_factorize!(solver)    # Perform LU factorization
simplelu_solve!(solver)        # Solve for the stored RHS
```

## Notes
This is a pure Julia implementation primarily for educational purposes and
small matrices. For production use, prefer optimized LAPACK-based factorizations.
"""
mutable struct LUSolver{T}
    n::Int
    A::Matrix{T}
    b::Vector{T}
    x::Vector{T}
    pivots::Vector{Int}
    perms::Vector{Int}
    info::Int

    function LUSolver{T}(n) where {T}
        return new(n, zeros(T, n, n), zeros(T, n), zeros(T, n), zeros(Int, n), zeros(Int, n), 0)
    end
end

function LUSolver(A::Matrix{T}) where {T}
    n = LinearAlgebra.checksquare(A)
    lu = LUSolver{eltype(A)}(n)
    lu.A .= A
    return lu
end

function LUSolver(A::Matrix{T}, b::Vector{T}) where {T}
    n = LinearAlgebra.checksquare(A)
    @assert n == length(b)
    lu = LUSolver{eltype(A)}(n)
    lu.A .= A
    lu.b .= b
    return lu
end

function simplelu_factorize!(lu::LUSolver{T}, pivot = true) where {T}
    A = lu.A

    return begin
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

    return lu.x
end

### Wrapper

"""
    SimpleLUFactorization(pivot::Bool = true)

A pure Julia LU factorization implementation without BLAS dependencies.
This solver is optimized for small matrices and situations where BLAS 
is not available or desirable.

## Constructor Arguments
- `pivot::Bool = true`: Whether to perform partial pivoting for numerical stability.
  Set to `false` for slightly better performance at the cost of stability.

## Features
- Pure Julia implementation (no BLAS dependencies)
- Partial pivoting support for numerical stability
- In-place matrix modification for memory efficiency  
- Fast for small matrices (typically < 100×100)
- Educational value for understanding LU factorization

## Performance Characteristics
- Optimal for small dense matrices
- No overhead from BLAS calls
- Linear scaling with problem size (O(n³) operations)
- Memory efficient due to in-place operations

## Use Cases
- Small matrices where BLAS overhead is significant
- Systems without optimized BLAS libraries
- Educational and prototyping purposes
- Embedded systems with memory constraints

## Example
```julia
# Stable version with pivoting (default)
alg1 = SimpleLUFactorization()
# Faster version without pivoting
alg2 = SimpleLUFactorization(false)

prob = LinearProblem(A, b)
sol = solve(prob, alg1)
```

## Notes
For larger matrices (> 100×100), consider using BLAS-based factorizations 
like `LUFactorization()` for better performance.
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
    return SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

function init_cacheval(
        alg::SimpleLUFactorization, A, b, u, Pl, Pr, maxiters::Int, abstol,
        reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return LUSolver(convert(AbstractMatrix, A))
end

function resize_cacheval!(cache, cacheval::LUSolver{T}, i) where {T}
    return setfield!(cache, :cacheval, LUSolver{T}(i))
end
