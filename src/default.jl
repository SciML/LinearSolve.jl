## Default algorithm

# Allows A === nothing as a stand-in for dense matrix
function defaultalg(A, b)
    if A isa DiffEqArrayOperator
        A = A.A
    end

    # Special case on Arrays: avoid BLAS for RecursiveFactorization.jl when
    # it makes sense according to the benchmarks, which is dependent on
    # whether MKL or OpenBLAS is being used
    if (A === nothing && !(b isa GPUArraysCore.AbstractGPUArray)) || A isa Matrix
        if (A === nothing || eltype(A) <: Union{Float32, Float64, ComplexF32, ComplexF64}) &&
           ArrayInterfaceCore.can_setindex(b)
            if length(b) <= 10
                alg = GenericLUFactorization()
            elseif (length(b) <= 100 || (isopenblas() && length(b) <= 500)) &&
                   eltype(A) <: Union{Float32, Float64}
                alg = RFLUFactorization()
                #elseif A === nothing || A isa Matrix
                #    alg = FastLUFactorization()
            else
                alg = LUFactorization()
            end
        else
            alg = LUFactorization()
        end

        # These few cases ensure the choice is optimal without the
        # dynamic dispatching of factorize
    elseif A isa Tridiagonal
        alg = GenericFactorization(; fact_alg = lu!)
    elseif A isa SymTridiagonal
        alg = GenericFactorization(; fact_alg = ldlt!)
    elseif A isa SparseMatrixCSC
        alg = KLUFactorization()

        # This catches the cases where a factorization overload could exist
        # For example, BlockBandedMatrix
    elseif A !== nothing && ArrayInterfaceCore.isstructured(A)
        alg = GenericFactorization()

        # This catches the case where A is a CuMatrix
        # Which does not have LU fully defined
    elseif A isa GPUArraysCore.AbstractGPUArray || b isa GPUArraysCore.AbstractGPUArray
        alg = LUFactorization()

        # Not factorizable operator, default to only using A*x
    else
        alg = KrylovJL_GMRES()
    end
    alg
end

## Other dispatches are to decrease the dispatch cost

function SciMLBase.solve(cache::LinearCache, alg::Nothing,
                         args...; kwargs...)
    @unpack A = cache
    if A isa DiffEqArrayOperator
        A = A.A
    end

    # Special case on Arrays: avoid BLAS for RecursiveFactorization.jl when
    # it makes sense according to the benchmarks, which is dependent on
    # whether MKL or OpenBLAS is being used
    if A isa Matrix
        b = cache.b
        if (A === nothing || eltype(A) <: Union{Float32, Float64, ComplexF32, ComplexF64}) &&
           ArrayInterfaceCore.can_setindex(b)
            if length(b) <= 10
                alg = GenericLUFactorization()
                SciMLBase.solve(cache, alg, args...; kwargs...)
            elseif (length(b) <= 100 || (isopenblas() && length(b) <= 500)) &&
                   eltype(A) <: Union{Float32, Float64}
                alg = RFLUFactorization()
                SciMLBase.solve(cache, alg, args...; kwargs...)
                #elseif A isa Matrix
                #    alg = FastLUFactorization()
                #    SciMLBase.solve(cache, alg, args...; kwargs...)
            else
                alg = LUFactorization()
                SciMLBase.solve(cache, alg, args...; kwargs...)
            end
        else
            alg = LUFactorization()
            SciMLBase.solve(cache, alg, args...; kwargs...)
        end

        # These few cases ensure the choice is optimal without the
        # dynamic dispatching of factorize
    elseif A isa Tridiagonal
        alg = GenericFactorization(; fact_alg = lu!)
        SciMLBase.solve(cache, alg, args...; kwargs...)
    elseif A isa SymTridiagonal
        alg = GenericFactorization(; fact_alg = ldlt!)
        SciMLBase.solve(cache, alg, args...; kwargs...)
    elseif A isa SparseMatrixCSC
        alg = KLUFactorization()
        SciMLBase.solve(cache, alg, args...; kwargs...)

        # This catches the cases where a factorization overload could exist
        # For example, BlockBandedMatrix
    elseif ArrayInterfaceCore.isstructured(A)
        alg = GenericFactorization()
        SciMLBase.solve(cache, alg, args...; kwargs...)

        # This catches the case where A is a CuMatrix
        # Which does not have LU fully defined
    elseif A isa GPUArraysCore.AbstractGPUArray
        alg = LUFactorization()
        SciMLBase.solve(cache, alg, args...; kwargs...)

        # Not factorizable operator, default to only using A*x
        # IterativeSolvers is faster on CPU but not GPU-compatible
    else
        alg = KrylovJL_GMRES()
        SciMLBase.solve(cache, alg, args...; kwargs...)
    end
end

function init_cacheval(alg::Nothing, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
    if A isa DiffEqArrayOperator
        A = A.A
    end

    # Special case on Arrays: avoid BLAS for RecursiveFactorization.jl when
    # it makes sense according to the benchmarks, which is dependent on
    # whether MKL or OpenBLAS is being used
    if A isa Matrix
        if (A === nothing || eltype(A) <: Union{Float32, Float64, ComplexF32, ComplexF64}) &&
           ArrayInterfaceCore.can_setindex(b)
            if length(b) <= 10
                alg = GenericLUFactorization()
                init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
            elseif (length(b) <= 100 || (isopenblas() && length(b) <= 500)) &&
                   eltype(A) <: Union{Float32, Float64}
                alg = RFLUFactorization()
                init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
                #elseif A isa Matrix
                #    alg = FastLUFactorization()
                #    init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
            else
                alg = LUFactorization()
                init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
            end
        else
            alg = LUFactorization()
            init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
        end

        # These few cases ensure the choice is optimal without the
        # dynamic dispatching of factorize
    elseif A isa Tridiagonal
        alg = GenericFactorization(; fact_alg = lu!)
        init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
    elseif A isa SymTridiagonal
        alg = GenericFactorization(; fact_alg = ldlt!)
        init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
    elseif A isa SparseMatrixCSC
        alg = KLUFactorization()
        init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)

        # This catches the cases where a factorization overload could exist
        # For example, BlockBandedMatrix
    elseif ArrayInterfaceCore.isstructured(A)
        alg = GenericFactorization()
        init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)

        # This catches the case where A is a CuMatrix
        # Which does not have LU fully defined
    elseif A isa GPUArraysCore.AbstractGPUArray
        alg = LUFactorization()
        init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)

        # Not factorizable operator, default to only using A*x
        # IterativeSolvers is faster on CPU but not GPU-compatible
    else
        alg = KrylovJL_GMRES()
        init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
    end
end
