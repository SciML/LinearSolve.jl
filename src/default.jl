needs_concrete_A(alg::DefaultLinearSolver) = true
mutable struct DefaultLinearSolverInit{T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
    T13, T14, T15, T16, T17, T18, T19, T20, T21}
    LUFactorization::T1
    QRFactorization::T2
    DiagonalFactorization::T3
    DirectLdiv!::T4
    SparspakFactorization::T5
    KLUFactorization::T6
    UMFPACKFactorization::T7
    KrylovJL_GMRES::T8
    GenericLUFactorization::T9
    RFLUFactorization::T10
    LDLtFactorization::T11
    BunchKaufmanFactorization::T12
    CHOLMODFactorization::T13
    SVDFactorization::T14
    CholeskyFactorization::T15
    NormalCholeskyFactorization::T16
    AppleAccelerateLUFactorization::T17
    MKLLUFactorization::T18
    QRFactorizationPivoted::T19
    KrylovJL_CRAIGMR::T20
    KrylovJL_LSMR::T21
end

@generated function __setfield!(cache::DefaultLinearSolverInit, alg::DefaultLinearSolver, v)
    ex = :()
    for alg in first.(EnumX.symbol_map(DefaultAlgorithmChoice.T))
        newex = quote
            setfield!(cache, $(Meta.quot(alg)), v)
        end
        alg_enum = getproperty(LinearSolve.DefaultAlgorithmChoice, alg)
        ex = if ex == :()
            Expr(:elseif, :(alg.alg == $(alg_enum)), newex,
                :(error("Algorithm Choice not Allowed")))
        else
            Expr(:elseif, :(alg.alg == $(alg_enum)), newex, ex)
        end
    end
    ex = Expr(:if, ex.args...)
end

# Legacy fallback
# For SciML algorithms already using `defaultalg`, all assume square matrix.
defaultalg(A, b) = defaultalg(A, b, OperatorAssumptions(true))

function defaultalg(A::Union{DiffEqArrayOperator, MatrixOperator}, b,
        assump::OperatorAssumptions{Bool})
    defaultalg(A.A, b, assump)
end

function defaultalg(A, b, assump::OperatorAssumptions{Nothing})
    issq = issquare(A)
    defaultalg(A, b, OperatorAssumptions(issq, assump.condition))
end

function defaultalg(A::SMatrix{S1, S2}, b, assump::OperatorAssumptions{Bool}) where {S1, S2}
    if S1 == S2
        return LUFactorization()
    else
        return SVDFactorization()  # QR(...) \ b is not defined currently
    end
end

function defaultalg(A::Tridiagonal, b, assump::OperatorAssumptions{Bool})
    if assump.issq
        DefaultLinearSolver(DefaultAlgorithmChoice.LUFactorization)
    else
        DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
    end
end

function defaultalg(A::SymTridiagonal, b, ::OperatorAssumptions{Bool})
    DefaultLinearSolver(DefaultAlgorithmChoice.LDLtFactorization)
end
function defaultalg(A::Bidiagonal, b, ::OperatorAssumptions{Bool})
    DefaultLinearSolver(DefaultAlgorithmChoice.DirectLdiv!)
end
function defaultalg(A::Factorization, b, ::OperatorAssumptions{Bool})
    DefaultLinearSolver(DefaultAlgorithmChoice.DirectLdiv!)
end
function defaultalg(A::Diagonal, b, ::OperatorAssumptions{Bool})
    DefaultLinearSolver(DefaultAlgorithmChoice.DiagonalFactorization)
end

function defaultalg(A::Hermitian, b, ::OperatorAssumptions{Bool})
    DefaultLinearSolver(DefaultAlgorithmChoice.CholeskyFactorization)
end

function defaultalg(A::Symmetric{<:Number, <:Array}, b, ::OperatorAssumptions{Bool})
    DefaultLinearSolver(DefaultAlgorithmChoice.BunchKaufmanFactorization)
end

function defaultalg(
        A::Symmetric{<:Number, <:SparseMatrixCSC}, b, ::OperatorAssumptions{Bool})
    DefaultLinearSolver(DefaultAlgorithmChoice.CHOLMODFactorization)
end

function defaultalg(A::AbstractSparseMatrixCSC{Tv, Ti}, b,
        assump::OperatorAssumptions{Bool}) where {Tv, Ti}
    if assump.issq
        DefaultLinearSolver(DefaultAlgorithmChoice.SparspakFactorization)
    else
        error("Generic number sparse factorization for non-square is not currently handled")
    end
end

@static if INCLUDE_SPARSE
    function defaultalg(A::AbstractSparseMatrixCSC{<:Union{Float64, ComplexF64}, Ti}, b,
            assump::OperatorAssumptions{Bool}) where {Ti}
        if assump.issq
            if length(b) <= 10_000 && length(nonzeros(A)) / length(A) < 2e-4
                DefaultLinearSolver(DefaultAlgorithmChoice.KLUFactorization)
            else
                DefaultLinearSolver(DefaultAlgorithmChoice.UMFPACKFactorization)
            end
        else
            DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
        end
    end
end

function defaultalg(A::GPUArraysCore.AnyGPUArray, b, assump::OperatorAssumptions{Bool})
    if assump.condition === OperatorCondition.IllConditioned || !assump.issq
        DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
    else
        DefaultLinearSolver(DefaultAlgorithmChoice.LUFactorization)
    end
end

# A === nothing case
function defaultalg(
        A::Nothing, b::GPUArraysCore.AnyGPUArray, assump::OperatorAssumptions{Bool})
    if assump.condition === OperatorCondition.IllConditioned || !assump.issq
        DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
    else
        DefaultLinearSolver(DefaultAlgorithmChoice.LUFactorization)
    end
end

# Ambiguity handling
function defaultalg(A::GPUArraysCore.AnyGPUArray, b::GPUArraysCore.AnyGPUArray,
        assump::OperatorAssumptions{Bool})
    if assump.condition === OperatorCondition.IllConditioned || !assump.issq
        DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
    else
        DefaultLinearSolver(DefaultAlgorithmChoice.LUFactorization)
    end
end

function defaultalg(A::SciMLBase.AbstractSciMLOperator, b,
        assump::OperatorAssumptions{Bool})
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
function defaultalg(A, b, assump::OperatorAssumptions{Bool})
    alg = if assump.issq
        # Special case on Arrays: avoid BLAS for RecursiveFactorization.jl when
        # it makes sense according to the benchmarks, which is dependent on
        # whether MKL or OpenBLAS is being used
        if (A === nothing && !(b isa GPUArraysCore.AnyGPUArray)) || A isa Matrix
            if (A === nothing ||
                eltype(A) <: BLASELTYPES) &&
               ArrayInterface.can_setindex(b) &&
               (__conditioning(assump) === OperatorCondition.IllConditioned ||
                __conditioning(assump) === OperatorCondition.WellConditioned)
                if length(b) <= 10
                    DefaultAlgorithmChoice.RFLUFactorization
                elseif appleaccelerate_isavailable()
                    DefaultAlgorithmChoice.AppleAccelerateLUFactorization
                elseif (length(b) <= 100 || (isopenblas() && length(b) <= 500) ||
                        (usemkl && length(b) <= 200)) &&
                       (A === nothing ? eltype(b) <: Union{Float32, Float64} :
                        eltype(A) <: Union{Float32, Float64})
                    DefaultAlgorithmChoice.RFLUFactorization
                    #elseif A === nothing || A isa Matrix
                    #    alg = FastLUFactorization()
                elseif usemkl
                    DefaultAlgorithmChoice.MKLLUFactorization
                else
                    DefaultAlgorithmChoice.LUFactorization
                end
            elseif __conditioning(assump) === OperatorCondition.VeryIllConditioned
                DefaultAlgorithmChoice.QRFactorization
            elseif __conditioning(assump) === OperatorCondition.SuperIllConditioned
                DefaultAlgorithmChoice.SVDFactorization
            elseif usemkl && (A === nothing ? eltype(b) <: BLASELTYPES :
                    eltype(A) <: BLASELTYPES)
                DefaultAlgorithmChoice.MKLLUFactorization
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
        if is_underdetermined(A)
            # Underdetermined
            DefaultAlgorithmChoice.QRFactorizationPivoted
        else
            DefaultAlgorithmChoice.QRFactorization
        end
    elseif assump.condition === OperatorCondition.VeryIllConditioned
        if is_underdetermined(A)
            # Underdetermined
            DefaultAlgorithmChoice.QRFactorizationPivoted
        else
            DefaultAlgorithmChoice.QRFactorization
        end
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
    elseif alg === :LDLtFactorization
        LDLtFactorization()
    elseif alg === :LUFactorization
        LUFactorization()
    elseif alg === :MKLLUFactorization
        MKLLUFactorization()
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
    elseif alg === :AppleAccelerateLUFactorization
        AppleAccelerateLUFactorization()
    elseif alg === :QRFactorizationPivoted
        QRFactorization(ColumnNorm())
    elseif alg === :KrylovJL_CRAIGMR
        KrylovJL_CRAIGMR()
    elseif alg === :KrylovJL_LSMR
        KrylovJL_LSMR()
    else
        error("Algorithm choice symbol $alg not allowed in the default")
    end
end

## Catch high level interface

function SciMLBase.init(prob::LinearProblem, alg::Nothing,
        args...;
        assumptions = OperatorAssumptions(issquare(prob.A)),
        kwargs...)
    SciMLBase.init(
        prob, defaultalg(prob.A, prob.b, assumptions), args...; assumptions, kwargs...)
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
        if alg === :KrylovJL_GMRES || alg === :KrylovJL_CRAIGMR || alg === :KrylovJL_LSMR
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

function defaultalg_symbol(::Type{T}) where {T}
    Base.typename(SciMLBase.parameterless_type(T)).name
end
defaultalg_symbol(::Type{<:GenericFactorization{typeof(ldlt!)}}) = :LDLtFactorization

defaultalg_symbol(::Type{<:QRFactorization{ColumnNorm}}) = :QRFactorizationPivoted

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
        alg_enum = getproperty(LinearSolve.DefaultAlgorithmChoice, alg)
        ex = if ex == :()
            Expr(:elseif, :(alg.alg == $(alg_enum)), newex,
                :(error("Algorithm Choice not Allowed")))
        else
            Expr(:elseif, :(alg.alg == $(alg_enum)), newex, ex)
        end
    end
    ex = Expr(:if, ex.args...)
end

"""
```
elseif DefaultAlgorithmChoice.LUFactorization === cache.alg
    (cache.cacheval.LUFactorization)' \\ dy
else
    ...
end
```
"""
@generated function defaultalg_adjoint_eval(cache::LinearCache, dy)
    ex = :()
    for alg in first.(EnumX.symbol_map(DefaultAlgorithmChoice.T))
        newex = if alg in Symbol.((DefaultAlgorithmChoice.MKLLUFactorization,
            DefaultAlgorithmChoice.AppleAccelerateLUFactorization,
            DefaultAlgorithmChoice.RFLUFactorization))
            quote
                getproperty(cache.cacheval, $(Meta.quot(alg)))[1]' \ dy
            end
        elseif alg in Symbol.((DefaultAlgorithmChoice.LUFactorization,
            DefaultAlgorithmChoice.QRFactorization,
            DefaultAlgorithmChoice.KLUFactorization,
            DefaultAlgorithmChoice.UMFPACKFactorization,
            DefaultAlgorithmChoice.LDLtFactorization,
            DefaultAlgorithmChoice.SparspakFactorization,
            DefaultAlgorithmChoice.BunchKaufmanFactorization,
            DefaultAlgorithmChoice.CHOLMODFactorization,
            DefaultAlgorithmChoice.SVDFactorization,
            DefaultAlgorithmChoice.CholeskyFactorization,
            DefaultAlgorithmChoice.NormalCholeskyFactorization,
            DefaultAlgorithmChoice.QRFactorizationPivoted,
            DefaultAlgorithmChoice.GenericLUFactorization))
            quote
                getproperty(cache.cacheval, $(Meta.quot(alg)))' \ dy
            end
        elseif alg in Symbol.((
            DefaultAlgorithmChoice.KrylovJL_GMRES, DefaultAlgorithmChoice.KrylovJL_LSMR,
            DefaultAlgorithmChoice.KrylovJL_CRAIGMR))
            quote
                invprob = LinearSolve.LinearProblem(transpose(cache.A), dy)
                solve(invprob, cache.alg;
                    abstol = cache.val.abstol,
                    reltol = cache.val.reltol,
                    verbose = cache.val.verbose)
            end
        else
            quote
                error("Default linear solver with algorithm $(alg) is currently not supported by Enzyme rules on LinearSolve.jl. Please open an issue on LinearSolve.jl detailing which algorithm is missing the adjoint handling")
            end
        end

        ex = if ex == :()
            Expr(:elseif,
                :(getproperty(DefaultAlgorithmChoice, $(Meta.quot(alg))) === cache.alg.alg),
                newex,
                :(error("Algorithm Choice not Allowed")))
        else
            Expr(:elseif,
                :(getproperty(DefaultAlgorithmChoice, $(Meta.quot(alg))) === cache.alg.alg),
                newex,
                ex)
        end
    end
    ex = Expr(:if, ex.args...)
end
