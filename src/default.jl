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
    LDLtFactorization
    BunchKaufmanFactorization
    CHOLMODFactorization
    SVDFactorization
    CholeskyFactorization
    NormalCholeskyFactorization
end

struct DefaultLinearSolver <: SciMLLinearSolveAlgorithm
    alg::DefaultAlgorithmChoice.T
end

mutable struct DefaultLinearSolverInit{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
                                       T13, T14, T15, T16, T17}
    LUFactorization::T1
    QRFactorization::T2
    DiagonalFactorization::T3
    DirectLdiv!::T4
    SparspakFactorization::T5
    KLUFactorization::T6
    UMFPACKFactorization::T7
    KrylovJL_GMRES::T8
    GenericLUFactorization::T9
    RowMaximumGenericLUFactorization::T10
    RFLUFactorization::T11
    LDLtFactorization::T12
    BunchKaufmanFactorization::T13
    CHOLMODFactorization::T14
    SVDFactorization::T15
    CholeskyFactorization::T16
    NormalCholeskyFactorization::T17
end

# Legacy fallback
# For SciML algorithms already using `defaultalg`, all assume square matrix.
defaultalg(A, b) = defaultalg(A, b, OperatorAssumptions(true))

function defaultalg(A::Union{DiffEqArrayOperator, MatrixOperator}, b,
                    assump::OperatorAssumptions)
    DefaultLinearSolver(defaultalg(A.A, b, assump))
end

function defaultalg(A, b, assump::OperatorAssumptions{Nothing})
    issq = issquare(A)
    DefaultLinearSolver(defaultalg(A, b, OperatorAssumptions(issq, assump.condition)))
end

function defaultalg(A::Tridiagonal, b, assump::OperatorAssumptions)
    if assump.issq
        DefaultLinearSolver(DefaultAlgorithmChoice.LUFactorization)
    else
        DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
    end
end

function defaultalg(A::SymTridiagonal, b, ::OperatorAssumptions)
    DefaultLinearSolver(DefaultAlgorithmChoice.LDLtFactorization)
end
function defaultalg(A::Bidiagonal, b, ::OperatorAssumptions)
    DefaultLinearSolver(DefaultAlgorithmChoice.DirectLdiv!)
end
function defaultalg(A::Factorization, b, ::OperatorAssumptions)
    DefaultLinearSolver(DefaultAlgorithmChoice.DirectLdiv!)
end
function defaultalg(A::Diagonal, b, ::OperatorAssumptions)
    DefaultLinearSolver(DefaultAlgorithmChoice.DiagonalFactorization)
end

function defaultalg(A::Hermitian, b, ::OperatorAssumptions)
    DefaultLinearSolver(DefaultAlgorithmChoice.CholeskyFactorization)
end

function defaultalg(A::Symmetric{<:Number, <:Array}, b, ::OperatorAssumptions)
    DefaultLinearSolver(DefaultAlgorithmChoice.BunchKaufmanFactorization)
end

function defaultalg(A::Symmetric{<:Number, <:SparseMatrixCSC}, b, ::OperatorAssumptions)
    DefaultLinearSolver(DefaultAlgorithmChoice.CHOLMODFactorization)
end

function defaultalg(A::AbstractSparseMatrixCSC{Tv, Ti}, b,
                    assump::OperatorAssumptions) where {Tv, Ti}
    if assump.issq
        DefaultLinearSolver(DefaultAlgorithmChoice.SparspakFactorization)
    else
        error("Generic number sparse factorization for non-square is not currently handled")
    end
end

@static if INCLUDE_SPARSE
    function defaultalg(A::AbstractSparseMatrixCSC{<:Union{Float64, ComplexF64}, Ti}, b,
                        assump::OperatorAssumptions) where {Ti}
        if assump.issq
            if length(b) <= 10_000
                DefaultLinearSolver(DefaultAlgorithmChoice.KLUFactorization)
            else
                DefaultLinearSolver(DefaultAlgorithmChoice.UMFPACKFactorization)
            end
        else
            DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
        end
    end
end

function defaultalg(A::GPUArraysCore.AbstractGPUArray, b, assump::OperatorAssumptions)
    if assump.condition === OperatorConodition.IllConditioned || !assump.issq
        DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
    else
        @static if VERSION >= v"1.8-"
            DefaultLinearSolver(DefaultAlgorithmChoice.LUFactorization)
        else
            DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
        end
    end
end

# A === nothing case
function defaultalg(A, b::GPUArraysCore.AbstractGPUArray, assump::OperatorAssumptions)
    if assump.condition === OperatorConodition.IllConditioned || !assump.issq
        DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
    else
        @static if VERSION >= v"1.8-"
            DefaultLinearSolver(DefaultAlgorithmChoice.LUFactorization)
        else
            DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
        end
    end
end

# Ambiguity handling
function defaultalg(A::GPUArraysCore.AbstractGPUArray, b::GPUArraysCore.AbstractGPUArray,
                    assump::OperatorAssumptions)
    if assump.condition === OperatorConodition.IllConditioned || !assump.issq
        DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
    else
        @static if VERSION >= v"1.8-"
            DefaultLinearSolver(DefaultAlgorithmChoice.LUFactorization)
        else
            DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
        end
    end
end

function defaultalg(A::SciMLBase.AbstractSciMLOperator, b,
                    assump::OperatorAssumptions)
    if has_ldiv!(A)
        return DefaultLinearSolver(DefaultAlgorithmChoice.DirectLdiv!)
    elseif !assump.issq
        m, n = size(A)
        if m < n
            DefaultLinearSolver(DefaultAlgorithmChoice.KrylovJL_CRAIGMR)
        else
            DefaultLinearSolver(DefaultAlgorithmChoice.KrylovJL_LSMR)
        end
    else
        DefaultLinearSolver(DefaultAlgorithmChoice.KrylovJL_GMRES)
    end
end

# Allows A === nothing as a stand-in for dense matrix
function defaultalg(A, b, assump::OperatorAssumptions)
    alg = if assump.issq
        # Special case on Arrays: avoid BLAS for RecursiveFactorization.jl when
        # it makes sense according to the benchmarks, which is dependent on
        # whether MKL or OpenBLAS is being used
        if (A === nothing && !(b isa GPUArraysCore.AbstractGPUArray)) || A isa Matrix
            if (A === nothing ||
                eltype(A) <: Union{Float32, Float64, ComplexF32, ComplexF64}) &&
               ArrayInterface.can_setindex(b) &&
               (__conditioning(assump) === OperatorCondition.IllConditioned ||
                __conditioning(assump) === OperatorCondition.WellConditioned)
                if length(b) <= 10
                    if __conditioning(assump) === OperatorCondition.IllConditioned
                        DefaultAlgorithmChoice.RowMaximumGenericLUFactorization
                    else
                        DefaultAlgorithmChoice.GenericLUFactorization
                    end
                elseif (length(b) <= 100 || (isopenblas() && length(b) <= 500)) &&
                       (A === nothing ? eltype(b) <: Union{Float32, Float64} :
                        eltype(A) <: Union{Float32, Float64})
                    DefaultAlgorithmChoice.RFLUFactorization
                    #elseif A === nothing || A isa Matrix
                    #    alg = FastLUFactorization()
                else
                    if __conditioning(assump) === OperatorCondition.IllConditioned
                        DefaultAlgorithmChoice.RowMaximumGenericLUFactorization
                    else
                        DefaultAlgorithmChoice.GenericLUFactorization
                    end
                end
            elseif __conditioning(assump) === OperatorCondition.VeryIllConditioned
                DefaultAlgorithmChoice.QRFactorization
            elseif __conditioning(assump) === OperatorCondition.SuperIllConditioned
                DefaultAlgorithmChoice.SVDFactorization
            else
                DefaultAlgorithmChoice.LUFactorization
            end

            # This catches the cases where a factorization overload could exist
            # For example, BlockBandedMatrix
        elseif A !== nothing && ArrayInterface.isstructured(A)
            error("Special factorization not handled in current default algorithm")

            # Not factorizable operator, default to only using A*x
        else
            DefaultAlgorithmChoice.KrylovJL_GMRES
        end
    elseif assump.condition === OperatorCondition.WellConditioned
        DefaultAlgorithmChoice.NormalCholeskyFactorization
    elseif assump.condition === OperatorCondition.IllConditioned
        DefaultAlgorithmChoice.QRFactorization
    elseif assump.condition === OperatorCondition.VeryIllConditioned
        DefaultAlgorithmChoice.QRFactorization
    elseif assump.condition === OperatorCondition.SuperIllConditioned
        DefaultAlgorithmChoice.SVDFactorization
    else
        error("Special factorization not handled in current default algorithm")
    end
    DefaultLinearSolver(alg)
end

function algchoice_to_alg(alg::Symbol)
    if alg === :SVDFactorization
        SVDFactorization(false, LinearAlgebra.QRIteration())
    elseif alg === :RowMaximumGenericLUFactorization
        GenericLUFactorization(RowMaximum())
    elseif alg === :LDLtFactorization
        LDLtFactorization()
    elseif alg === :LUFactorization
        LUFactorization()
    elseif alg === :QRFactorization
        QRFactorization()
    elseif alg === :DiagonalFactorization
        DiagonalFactorization()
    elseif alg === :DirectLdiv!
        DirectLdiv!()
    elseif alg === :SparspakFactorization
        SparspakFactorization()
    elseif alg === :KLUFactorization
        KLUFactorization()
    elseif alg === :UMFPACKFactorization
        UMFPACKFactorization()
    elseif alg === :KrylovJL_GMRES
        KrylovJL_GMRES()
    elseif alg === :GenericLUFactorization
        GenericLUFactorization()
    elseif alg === :RFLUFactorization
        RFLUFactorization()
    elseif alg === :BunchKaufmanFactorization
        BunchKaufmanFactorization()
    elseif alg === :CHOLMODFactorization
        CHOLMODFactorization()
    elseif alg === :CholeskyFactorization
        CholeskyFactorization()
    elseif alg === :NormalCholeskyFactorization
        NormalCholeskyFactorization()
    else
        error("Algorithm choice symbol $alg not allowed in the default")
    end
end

## Catch high level interface

function SciMLBase.init(prob::LinearProblem, alg::Nothing,
                        args...;
                        assumptions = OperatorAssumptions(issquare(prob.A)),
                        kwargs...)
    alg = defaultalg(prob.A, prob.b, assumptions)
    SciMLBase.init(prob, alg, args...; assumptions, kwargs...)
end

function SciMLBase.solve!(cache::LinearCache, alg::Nothing,
                          args...; assump::OperatorAssumptions = OperatorAssumptions(),
                          kwargs...)
    @unpack A, b = cache
    SciMLBase.solve!(cache, defaultalg(A, b, assump), args...; kwargs...)
end

function init_cacheval(alg::Nothing, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
                       verbose::Bool, assump::OperatorAssumptions)
    init_cacheval(defaultalg(A, b, assump), A, b, u, Pl, Pr, maxiters, abstol, reltol,
                  verbose,
                  assump)
end

"""
cache.cacheval = NamedTuple(LUFactorization = cache of LUFactorization, ...)
"""
@generated function init_cacheval(alg::DefaultLinearSolver, A, b, u, Pl, Pr, maxiters::Int,
                                  abstol, reltol,
                                  verbose::Bool, assump::OperatorAssumptions)
    caches = map(first.(EnumX.symbol_map(DefaultAlgorithmChoice.T))) do alg
        if alg === :KrylovJL_GMRES
            quote
                if A isa Matrix || A isa SparseMatrixCSC
                    nothing
                else
                    init_cacheval($(algchoice_to_alg(alg)), A, b, u, Pl, Pr, maxiters,
                                  abstol, reltol,
                                  verbose,
                                  assump)
                end
            end
        else
            quote
                init_cacheval($(algchoice_to_alg(alg)), A, b, u, Pl, Pr, maxiters, abstol,
                              reltol,
                              verbose,
                              assump)
            end
        end
    end
    Expr(:call, :DefaultLinearSolverInit, caches...)
end

"""
if algsym === :LUFactorization
    cache.cacheval.LUFactorization = ...
else
    ...
end
"""
@generated function get_cacheval(cache::LinearCache, algsym::Symbol)
    ex = :()
    for alg in first.(EnumX.symbol_map(DefaultAlgorithmChoice.T))
        ex = if ex == :()
            Expr(:elseif, :(algsym === $(Meta.quot(alg))),
                 :(getfield(cache.cacheval, $(Meta.quot(alg)))))
        else
            Expr(:elseif, :(algsym === $(Meta.quot(alg))),
                 :(getfield(cache.cacheval, $(Meta.quot(alg)))), ex)
        end
    end
    ex = Expr(:if, ex.args...)

    quote
        if cache.alg isa DefaultLinearSolver
            $ex
        else
            cache.cacheval
        end
    end
end

function defaultalg_symbol(::Type{T}) where {T}
    Symbol(split(string(SciMLBase.parameterless_type(T)), ".")[end])
end
function defaultalg_symbol(::Type{<:GenericLUFactorization{LinearAlgebra.RowMaximum}})
    :RowMaximumGenericLUFactorization
end
defaultalg_symbol(::Type{<:GenericFactorization{typeof(ldlt!)}}) = :LDLtFactorization

"""
if alg.alg === DefaultAlgorithmChoice.LUFactorization
    SciMLBase.solve!(cache, LUFactorization(), args...; kwargs...))
else
    ...
end
"""
@generated function SciMLBase.solve!(cache::LinearCache, alg::DefaultLinearSolver,
                                     args...;
                                     assump::OperatorAssumptions = OperatorAssumptions(),
                                     kwargs...)
    ex = :()
    for alg in first.(EnumX.symbol_map(DefaultAlgorithmChoice.T))
        newex = quote
            sol = SciMLBase.solve!(cache, $(algchoice_to_alg(alg)), args...; kwargs...)
            SciMLBase.build_linear_solution(alg, sol.u, sol.resid, sol.cache;
                                            retcode = sol.retcode,
                                            iters = sol.iters, stats = sol.stats)
        end
        ex = if ex == :()
            Expr(:elseif, :(Symbol(alg.alg) === $(Meta.quot(alg))), newex,
                 :(error("Algorithm Choice not Allowed")))
        else
            Expr(:elseif, :(Symbol(alg.alg) === $(Meta.quot(alg))), newex, ex)
        end
    end
    ex = Expr(:if, ex.args...)
end
