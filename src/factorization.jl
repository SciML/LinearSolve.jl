default_factorize(A) = lu(A)


mutable struct LinSolveFactorize{F}
    factorization::F
    A::Any
end

LinSolveFactorize(factorization) = LinSolveFactorize(factorization, nothing)

function (p::LinSolveFactorize)(x, A, b, update_matrix = false; kwargs...)
    if update_matrix
        p.A = p.factorization(A)
    end
    if typeof(p.A) <: SuiteSparse.UMFPACK.UmfpackLU ||
       typeof(p.factorization) <: typeof(lu)
        ldiv!(x, p.A, b) # No 2-arg form for SparseArrays!
    else
        x .= b
        ldiv!(p.A, x)
    end
end

function (p::LinSolveFactorize)(::Type{Val{:init}}, f, u0_prototype)
    LinSolveFactorize(p.factorization, nothing)
end

Base.resize!(p::LinSolveFactorize, i) = p.A = nothing


mutable struct LinSolveGPUFactorize{F,T}
    factorization::F
    A::Any
    x_cache::T
end

LinSolveGPUFactorize(factorization = qr) =
    LinSolveGPUFactorize(factorization, nothing, nothing)

function (p::LinSolveGPUFactorize)(x, A, b, update_matrix = false; kwargs...)
    if update_matrix
        p.A = p.factorization(cuify(A))
    end
    ldiv!(p.x_cache, p.A, cuify(b))
    x .= Array(p.x_cache)
end

function (p::LinSolveGPUFactorize)(::Type{Val{:init}}, f, u0_prototype)
    LinSolveGPUFactorize(p.factorization, nothing, cuify(u0_prototype))
end

Base.resize!(p::LinSolveGPUFactorize, i) = p.A = nothing


## A much simpler LU for when we know it's Array
mutable struct LinSolveLUFactorize
    A::LU{Float64,Matrix{Float64}}
    openblas::Bool
end

LinSolveLUFactorize() = LinSolveLUFactorize(lu(rand(1, 1)), isopenblas())

function (p::LinSolveLUFactorize)(
    x::Vector{Float64},
    A::Matrix{Float64},
    b::Vector{Float64},
    update_matrix::Bool = false;
    kwargs...,
)
    if update_matrix
        if ArrayInterface.can_setindex(x) &&
           (size(A, 1) <= 100 || (p.openblas && size(A, 1) <= 500))
            p.A = RecursiveFactorization.lu!(A)
        else
            p.A = lu!(A)
        end
    end
    ldiv!(x, p.A, b)
end

function (p::LinSolveLUFactorize)(::Type{Val{:init}}, f, u0_prototype)
    LinSolveLUFactorize(lu(rand(eltype(u0_prototype), 1, 1)), p.openblas)
end

Base.resize!(p::LinSolveLUFactorize, i) = p.A = nothing
