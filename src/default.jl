# Legacy fallback
# For SciML algorithms already using `defaultalg`, all assume square matrix.
defaultalg(A, b) = defaultalg(A, b, OperatorAssumptions(Val(true)))

function defaultalg(A::Union{DiffEqArrayOperator,MatrixOperator}, b, assumptions::OperatorAssumptions)
    defaultalg(A.A, b, assumptions)
end

# Ambiguity handling
function defaultalg(A::Union{DiffEqArrayOperator,MatrixOperator}, b, assumptions::OperatorAssumptions{nothing})
    defaultalg(A.A, b, assumptions)
end

function defaultalg(A::Union{DiffEqArrayOperator,MatrixOperator}, b, assumptions::OperatorAssumptions{false})
    defaultalg(A.A, b, assumptions)
end

function defaultalg(A::Union{DiffEqArrayOperator,MatrixOperator}, b, assumptions::OperatorAssumptions{true})
    defaultalg(A.A, b, assumptions)
end

function defaultalg(A, b, ::OperatorAssumptions{Nothing})
    issq = issquare(A)
    defaultalg(A, b, OperatorAssumptions(Val(issq)))
end

function defaultalg(A::Tridiagonal, b, ::OperatorAssumptions{true})
    GenericFactorization(; fact_alg = lu!)
end
function defaultalg(A::Tridiagonal, b, ::OperatorAssumptions{false})
    GenericFactorization(; fact_alg = qr!)
end
function defaultalg(A::SymTridiagonal, b, ::OperatorAssumptions{true})
    GenericFactorization(; fact_alg = ldlt!)
end
function defaultalg(A::Bidiagonal, b, ::OperatorAssumptions{true})
    DirectLdiv!()
end
function defaultalg(A::Factorization, b, ::OperatorAssumptions{true})
    DirectLdiv!()
end
function defaultalg(A::Diagonal, b, ::OperatorAssumptions{true})
    DiagonalFactorization()
end
function defaultalg(A::Diagonal, b, ::OperatorAssumptions{Nothing})
    DiagonalFactorization()
end

function defaultalg(A::SparseMatrixCSC{Tv, Ti}, b,
                    ::OperatorAssumptions{true}) where {Tv, Ti}
    SparspakFactorization()
end

@static if INCLUDE_SPARSE
    function defaultalg(A::SparseMatrixCSC{<:Union{Float64, ComplexF64}, Ti}, b,
                        ::OperatorAssumptions{true}) where {Ti}
        if length(b) <= 10_000
            KLUFactorization()
        else
            UMFPACKFactorization()
        end
    end
end

function defaultalg(A::GPUArraysCore.AbstractGPUArray, b, ::OperatorAssumptions{true})
    if VERSION >= v"1.8-"
        LUFactorization()
    else
        QRFactorization()
    end
end

function defaultalg(A, b::GPUArraysCore.AbstractGPUArray, ::OperatorAssumptions{true})
    if VERSION >= v"1.8-"
        LUFactorization()
    else
        QRFactorization()
    end
end

function defaultalg(A::SciMLBase.AbstractSciMLOperator, b,
                    assumptions::OperatorAssumptions{true})
    if has_ldiv!(A)
        return DirectLdiv!()
    end

    KrylovJL_GMRES()
end

# Ambiguity handling
function defaultalg(A::SciMLBase.AbstractSciMLOperator, b,
                    assumptions::OperatorAssumptions{Nothing})
    if has_ldiv!(A)
        return DirectLdiv!()
    end

    KrylovJL_GMRES()
end

function defaultalg(A::SciMLBase.AbstractSciMLOperator, b,
                    assumptions::OperatorAssumptions{false})
    m, n = size(A)
    if m < n
        KrylovJL_CRAIGMR()
    else
        KrylovJL_LSMR()
    end
end

# Handle ambiguity
function defaultalg(A::GPUArraysCore.AbstractGPUArray, b::GPUArraysCore.AbstractGPUArray,
                    ::OperatorAssumptions{true})
    if VERSION >= v"1.8-"
        LUFactorization()
    else
        QRFactorization()
    end
end

function defaultalg(A::GPUArraysCore.AbstractGPUArray, b, ::OperatorAssumptions{false})
    QRFactorization()
end

function defaultalg(A, b::GPUArraysCore.AbstractGPUArray, ::OperatorAssumptions{false})
    QRFactorization()
end

# Handle ambiguity
function defaultalg(A::GPUArraysCore.AbstractGPUArray, b::GPUArraysCore.AbstractGPUArray,
                    ::OperatorAssumptions{false})
    QRFactorization()
end

# Allows A === nothing as a stand-in for dense matrix
function defaultalg(A, b, ::OperatorAssumptions{true})
    # Special case on Arrays: avoid BLAS for RecursiveFactorization.jl when
    # it makes sense according to the benchmarks, which is dependent on
    # whether MKL or OpenBLAS is being used
    if (A === nothing && !(b isa GPUArraysCore.AbstractGPUArray)) || A isa Matrix
        if (A === nothing || eltype(A) <: Union{Float32, Float64, ComplexF32, ComplexF64}) &&
           ArrayInterfaceCore.can_setindex(b)
            if length(b) <= 10
                alg = GenericLUFactorization()
            elseif (length(b) <= 100 || (isopenblas() && length(b) <= 500)) &&
                   (A === nothing ? eltype(b) <: Union{Float32, Float64} :
                    eltype(A) <: Union{Float32, Float64})
                alg = RFLUFactorization()
                #elseif A === nothing || A isa Matrix
                #    alg = FastLUFactorization()
            else
                alg = LUFactorization()
            end
        else
            alg = LUFactorization()
        end

        # This catches the cases where a factorization overload could exist
        # For example, BlockBandedMatrix
    elseif A !== nothing && ArrayInterfaceCore.isstructured(A)
        alg = GenericFactorization()

        # Not factorizable operator, default to only using A*x
    else
        alg = KrylovJL_GMRES()
    end
    alg
end

function defaultalg(A, b, ::OperatorAssumptions{false})
    QRFactorization()
end

## Catch high level interface

function SciMLBase.solve(cache::LinearCache, alg::Nothing,
                         args...; assumptions::OperatorAssumptions = OperatorAssumptions(),
                         kwargs...)
    @unpack A, b = cache
    SciMLBase.solve(cache, defaultalg(A, b, assumptions), args...; kwargs...)
end

function init_cacheval(alg::Nothing, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
                       verbose::Bool, assumptions::OperatorAssumptions)
    init_cacheval(defaultalg(A, b, assumptions), A, b, u, Pl, Pr, maxiters, abstol, reltol,
                  verbose,
                  assumptions)
end
