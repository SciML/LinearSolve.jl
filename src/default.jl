struct DefaultLinearSolver <: SciMLLinearSolveAlgorithm
    alg::DefaultAlgorithmChoice
end

EnumX.@enumx DefaultAlgorithmChoice begin
    LUFactorization
    QRFactorization
    DiagonalFactorization
    DirectLdiv!
    SparspakFactorization
    KLUFactorization
    UMFPACKFactorization
    KrylovJL_GMRES
    GenericLUFactorization
    RowMaximumGenericLUFactorization
    RFLUFactorization
    PivotedRFLUFactorization
    LDLtFactorization
    SVDFactorization
end

# Legacy fallback
# For SciML algorithms already using `defaultalg`, all assume square matrix.
defaultalg(A, b) = defaultalg(A, b, OperatorAssumptions(Val(true)))

function defaultalg(A::Union{DiffEqArrayOperator, MatrixOperator}, b,
                    assumptions::OperatorAssumptions)
    DefaultLinearSolver(defaultalg(A.A, b, assumptions))
end

# Ambiguity handling
function defaultalg(A::Union{DiffEqArrayOperator, MatrixOperator}, b,
                    assumptions::OperatorAssumptions{nothing})
    DefaultLinearSolver(defaultalg(A.A, b, assumptions))
end

function defaultalg(A::Union{DiffEqArrayOperator, MatrixOperator}, b,
                    assumptions::OperatorAssumptions{false})
                    DefaultLinearSolver(defaultalg(A.A, b, assumptions))
end

function defaultalg(A::Union{DiffEqArrayOperator, MatrixOperator}, b,
                    assumptions::OperatorAssumptions{true})
                    DefaultLinearSolver(defaultalg(A.A, b, assumptions))
end

function defaultalg(A, b, ::OperatorAssumptions{Nothing})
    issq = issquare(A)
    DefaultLinearSolver(defaultalg(A, b, OperatorAssumptions(Val(issq))))
end

function defaultalg(A::Tridiagonal, b, ::OperatorAssumptions{true})
    DefaultAlgorithmChoice.LUFactorization
end
function defaultalg(A::Tridiagonal, b, ::OperatorAssumptions{false})
    DefaultAlgorithmChoice.QRFactorization
end
function defaultalg(A::SymTridiagonal, b, ::OperatorAssumptions{true})
    DefaultAlgorithmChoice.LDLtFactorization
end
function defaultalg(A::Bidiagonal, b, ::OperatorAssumptions{true})
    DefaultAlgorithmChoice.DirectLdiv!
end
function defaultalg(A::Factorization, b, ::OperatorAssumptions{true})
    DefaultAlgorithmChoice.DirectLdiv!
end
function defaultalg(A::Diagonal, b, ::OperatorAssumptions{true})
    DefaultAlgorithmChoice.DiagonalFactorization
end
function defaultalg(A::Diagonal, b, ::OperatorAssumptions{Nothing})
    DefaultAlgorithmChoice.DiagonalFactorization
end

function defaultalg(A::AbstractSparseMatrixCSC{Tv, Ti}, b,
                    ::OperatorAssumptions{true}) where {Tv, Ti}
    DefaultAlgorithmChoice.SparspakFactorization
end

@static if INCLUDE_SPARSE
    function defaultalg(A::AbstractSparseMatrixCSC{<:Union{Float64, ComplexF64}, Ti}, b,
                        ::OperatorAssumptions{true}) where {Ti}
        if length(b) <= 10_000
            DefaultAlgorithmChoice.KLUFactorization
        else
            DefaultAlgorithmChoice.UMFPACKFactorization
        end
    end
end

function defaultalg(A::GPUArraysCore.AbstractGPUArray, b, assump::OperatorAssumptions{true})
    if VERSION >= v"1.8-"
        DefaultAlgorithmChoice.LUFactorization
    else
        DefaultAlgorithmChoice.QRFactorization
    end
end

function defaultalg(A::GPUArraysCore.AbstractGPUArray, b,
                    assump::OperatorAssumptions{true, OperatorCondition.IllConditioned})
    DefaultAlgorithmChoice.QRFactorization
end

function defaultalg(A, b::GPUArraysCore.AbstractGPUArray, assump::OperatorAssumptions{true})
    if VERSION >= v"1.8-"
        DefaultAlgorithmChoice.LUFactorization
    else
        DefaultAlgorithmChoice.QRFactorization
    end
end

function defaultalg(A, b::GPUArraysCore.AbstractGPUArray,
                    assump::OperatorAssumptions{true, OperatorCondition.IllConditioned})
    DefaultAlgorithmChoice.QRFactorization
end

function defaultalg(A::SciMLBase.AbstractSciMLOperator, b,
                    assumptions::OperatorAssumptions{true})
    if has_ldiv!(A)
        return DefaultAlgorithmChoice.DirectLdiv!
    end

    DefaultAlgorithmChoice.KrylovJL_GMRES
end

# Ambiguity handling
function defaultalg(A::SciMLBase.AbstractSciMLOperator, b,
                    assumptions::OperatorAssumptions{Nothing})
    if has_ldiv!(A)
        return DefaultAlgorithmChoice.DirectLdiv!
    end

    DefaultAlgorithmChoice.KrylovJL_GMRES
end

function defaultalg(A::SciMLBase.AbstractSciMLOperator, b,
                    assumptions::OperatorAssumptions{false})
    m, n = size(A)
    if m < n
        DefaultAlgorithmChoice.KrylovJL_CRAIGMR
    else
        DefaultAlgorithmChoice.KrylovJL_LSMR
    end
end

# Handle ambiguity
function defaultalg(A::GPUArraysCore.AbstractGPUArray, b::GPUArraysCore.AbstractGPUArray,
                    ::OperatorAssumptions{true})
    if VERSION >= v"1.8-"
        DefaultAlgorithmChoice.LUFactorization
    else
        DefaultAlgorithmChoice.QRFactorization
    end
end

function defaultalg(A::GPUArraysCore.AbstractGPUArray, b::GPUArraysCore.AbstractGPUArray,
                    ::OperatorAssumptions{true, OperatorCondition.IllConditioned})
    DefaultAlgorithmChoice.QRFactorization
end

function defaultalg(A::GPUArraysCore.AbstractGPUArray, b, ::OperatorAssumptions{false})
    DefaultAlgorithmChoice.QRFactorization
end

function defaultalg(A, b::GPUArraysCore.AbstractGPUArray, ::OperatorAssumptions{false})
    DefaultAlgorithmChoice.QRFactorization
end

# Handle ambiguity
function defaultalg(A::GPUArraysCore.AbstractGPUArray, b::GPUArraysCore.AbstractGPUArray,
                    ::OperatorAssumptions{false})
    DefaultAlgorithmChoice.QRFactorization
end

# Allows A === nothing as a stand-in for dense matrix
function defaultalg(A, b, assump::OperatorAssumptions{true})
    # Special case on Arrays: avoid BLAS for RecursiveFactorization.jl when
    # it makes sense according to the benchmarks, which is dependent on
    # whether MKL or OpenBLAS is being used
    if (A === nothing && !(b isa GPUArraysCore.AbstractGPUArray)) || A isa Matrix
        if (A === nothing || eltype(A) <: Union{Float32, Float64, ComplexF32, ComplexF64}) &&
           ArrayInterface.can_setindex(b) &&
           (__conditioning(assump) === OperatorCondition.IllConditioned ||
            __conditioning(assump) === OperatorCondition.WellConditioned)
            if length(b) <= 10
                if __conditioning(assump) === OperatorCondition.IllConditioned
                    alg = DefaultAlgorithmChoice.RowMaximumGenericLUFactorization
                else
                    alg = DefaultAlgorithmChoice.GenericLUFactorization
                end          
            elseif (length(b) <= 100 || (isopenblas() && length(b) <= 500)) &&
                   (A === nothing ? eltype(b) <: Union{Float32, Float64} :
                    eltype(A) <: Union{Float32, Float64})
                DefaultAlgorithmChoice.RFLUFactorization
                #elseif A === nothing || A isa Matrix
                #    alg = FastLUFactorization()
            else
                if __conditioning(assump) === OperatorCondition.IllConditioned
                    alg = DefaultAlgorithmChoice.RowMaximumGenericLUFactorization
                else
                    alg = DefaultAlgorithmChoice.GenericLUFactorization
                end
            end
        elseif __conditioning(assump) === OperatorCondition.VeryIllConditioned
            alg = DefaultAlgorithmChoice.QRFactorization
        elseif __conditioning(assump) === OperatorCondition.SuperIllConditioned
            alg = DefaultAlgorithmChoice.SVDFactorization
        else
            alg = DefaultAlgorithmChoice.LUFactorization
        end

        # This catches the cases where a factorization overload could exist
        # For example, BlockBandedMatrix
    elseif A !== nothing && ArrayInterface.isstructured(A)
        alg = DefaultAlgorithmChoice.GenericFactorization

        # Not factorizable operator, default to only using A*x
    else
        alg = DefaultAlgorithmChoice.KrylovJL_GMRES
    end
    alg
end

function defaultalg(A, b, ::OperatorAssumptions{false, OperatorCondition.WellConditioned})
    DefaultAlgorithmChoice.NormalCholeskyFactorization
end

function defaultalg(A, b, ::OperatorAssumptions{false, OperatorCondition.IllConditioned})
    DefaultAlgorithmChoice.QRFactorization
end

function defaultalg(A, b,
                    ::OperatorAssumptions{false, OperatorCondition.VeryIllConditioned})
    DefaultAlgorithmChoice.QRFactorization
end

function defaultalg(A, b,
                    ::OperatorAssumptions{false, OperatorCondition.SuperIllConditioned})
    DefaultAlgorithmChoice.SVDFactorization
end

## Catch high level interface

function SciMLBase.init(prob::LinearProblem, alg::Nothing,
                        args...;
                        assumptions = OperatorAssumptions(Val(issquare(prob.A))),
                        kwargs...)
    alg = defaultalg(prob.A, prob.b, assumptions)
    SciMLBase.init(prob, alg, args...; assumptions, kwargs...)
end

function SciMLBase.solve!(cache::LinearCache, alg::Nothing,
                          args...; assumptions::OperatorAssumptions = OperatorAssumptions(),
                          kwargs...)
    @unpack A, b = cache
    SciMLBase.solve!(cache, defaultalg(A, b, assumptions), args...; kwargs...)
end

function init_cacheval(alg::Nothing, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
                       verbose::Bool, assumptions::OperatorAssumptions)
    init_cacheval(defaultalg(A, b, assumptions), A, b, u, Pl, Pr, maxiters, abstol, reltol,
                  verbose,
                  assumptions)
end

"""
cache.cacheval = NamedTuple(LUFactorization = cache of LUFactorization, ...)
"""
@generated function init_cacheval(alg::DefaultLinearSolver, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
                       verbose::Bool, assumptions::OperatorAssumptions)
    caches = [$alg = :(init_cacheval($alg(), A, b, u, Pl, Pr, maxiters, abstol, reltol,
                verbose,
                assumptions)) for alg in first.(EnumX.symbol_map(DefaultAlgorithmChoice.T))]
    quote
        return NamedTuple($caches...)
    end
end

"""
if cache.alg.alg === DefaultAlgorithmChoice.LUFactorization
    cache.cacheval[LUFactorization]
else
    ...
end
"""
@generated function get_cacheval(cache::LinearCache)
    ex = :()
    for alg in first.(EnumX.symbol_map(DefaultAlgorithmChoice.T))
        ex = if expr === :()
            Expr(:elseif, :(cache.alg.alg === $alg),:(cache.cacheval[$alg]))
        else
            Expr(:elseif, :(cache.alg.alg === $alg),:(cache.cacheval[$alg]),ex)
        end
    end
    ex = Expr(:if,ex.args...)

    quote
        if cache.alg === DefaultLinearSolver
            $ex
        else
            cache.cacheval
        end
    end
end

"""
if alg.alg === DefaultAlgorithmChoice.LUFactorization
    SciMLBase.solve!(cache, LUFactorization(), args...; kwargs...))
else
    ...
end
"""
function SciMLBase.solve!(cache::LinearCache, alg::DefaultLinearSolver,
                          args...; assumptions::OperatorAssumptions = OperatorAssumptions(),
                          kwargs...)
    ex = :()
    for alg in first.(EnumX.symbol_map(DefaultAlgorithmChoice.T))
        ex = if expr === :()
            Expr(:elseif, :(alg.alg === $alg),:(SciMLBase.solve!(cache, $alg(), args...; kwargs...)))
        else
            Expr(:elseif, :(alg.alg === $alg),:(SciMLBase.solve!(cache, $alg(), args...; kwargs...)),ex)
        end
    end
    ex = Expr(:if,ex.args...)          
    
end