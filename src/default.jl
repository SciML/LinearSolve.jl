## Default algorithm

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
        if eltype(A) <: Union{Float32,Float64,ComplexF32,ComplexF64} &&
                    ArrayInterface.can_setindex(cache.b) && (size(A,1) <= 100 ||
                                              (isopenblas() && size(A,1) <= 500)
                                             )
            alg = RFLUFactorization()
            SciMLBase.solve(cache, alg, args...; kwargs...)
        else
            alg = LUFactorization()
            SciMLBase.solve(cache, alg, args...; kwargs...)
        end

    # These few cases ensure the choice is optimal without the
    # dynamic dispatching of factorize
    elseif A isa Tridiagonal
        alg = GenericFactorization(;fact_alg=lu!)
        SciMLBase.solve(cache, alg, args...; kwargs...)
    elseif A isa SymTridiagonal
        alg = GenericFactorization(;fact_alg=ldlt!)
        SciMLBase.solve(cache, alg, args...; kwargs...)
    elseif A isa SparseMatrixCSC
        alg = UMFPACKFactorization()
        SciMLBase.solve(cache, alg, args...; kwargs...)

    # This catches the cases where a factorization overload could exist
    # For example, BlockBandedMatrix
    elseif ArrayInterface.isstructured(A)
        alg = GenericFactorization()
        SciMLBase.solve(cache, alg, args...; kwargs...)

    # This catches the case where A is a CuMatrix
    # Which does not have LU fully defined
    elseif !(A isa AbstractDiffEqOperator)
        alg = QRFactorization()
        SciMLBase.solve(cache, alg, args...; kwargs...)

    # Not factorizable operator, default to only using A*x
    # IterativeSolvers is faster on CPU but not GPU-compatible
    elseif cache.u isa Array
        alg = IterativeSolversJL_GMRES()
        SciMLBase.solve(cache, alg, args...; kwargs...)
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
        if eltype(A) <: Union{Float32,Float64,ComplexF32,ComplexF64} &&
                    ArrayInterface.can_setindex(b) && (size(A,1) <= 100 ||
                                              (isopenblas() && size(A,1) <= 500)
                                             )
            alg = RFLUFactorization()
            init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
        else
            alg = LUFactorization()
            init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
        end

    # These few cases ensure the choice is optimal without the
    # dynamic dispatching of factorize
    elseif A isa Tridiagonal
        alg = GenericFactorization(;fact_alg=lu!)
        init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
    elseif A isa SymTridiagonal
        alg = GenericFactorization(;fact_alg=ldlt!)
        init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
    elseif A isa SparseMatrixCSC
        alg = UMFPACKFactorization()
        init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)

    # This catches the cases where a factorization overload could exist
    # For example, BlockBandedMatrix
    elseif ArrayInterface.isstructured(A)
        alg = GenericFactorization()
        init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)

    # This catches the case where A is a CuMatrix
    # Which does not have LU fully defined
    elseif !(A isa AbstractDiffEqOperator)
        alg = QRFactorization()
        init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)

    # Not factorizable operator, default to only using A*x
    # IterativeSolvers is faster on CPU but not GPU-compatible
    elseif u isa Array
        alg = IterativeSolversJL_GMRES()
        init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
    else
        alg = KrylovJL_GMRES()
        init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
    end
end
