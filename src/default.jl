needs_concrete_A(alg::DefaultLinearSolver) = true
mutable struct DefaultLinearSolverInit{
        T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
        T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24,
        TA,
    }
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
    BLISLUFactorization::T22
    CudaOffloadLUFactorization::T23
    MetalLUFactorization::T24
    A_backup::TA  # reference to original prob.A for restoring cache.A after in-place LU
end

@generated function __setfield!(cache::DefaultLinearSolverInit, alg::DefaultLinearSolver, v)
    ex = :()
    for alg in first.(EnumX.symbol_map(DefaultAlgorithmChoice.T))
        newex = quote
            setfield!(cache, $(Meta.quot(alg)), v)
        end
        alg_enum = getproperty(LinearSolve.DefaultAlgorithmChoice, alg)
        ex = if ex == :()
            Expr(
                :elseif, :(alg.alg == $(alg_enum)), newex,
                :(error("Algorithm Choice not Allowed"))
            )
        else
            Expr(:elseif, :(alg.alg == $(alg_enum)), newex, ex)
        end
    end
    return ex = Expr(:if, ex.args...)
end

# Handle special case of Column-pivoted QR fallback for LU
function __setfield!(
        cache::DefaultLinearSolverInit,
        alg::DefaultLinearSolver, v::LinearAlgebra.QRPivoted
    )
    return setfield!(cache, :QRFactorizationPivoted, v)
end

"""
    defaultalg(A, b, assumptions::OperatorAssumptions)

Select the most appropriate linear solver algorithm based on the matrix `A`, 
right-hand side `b`, and operator assumptions. This is the core algorithm 
selection logic used by LinearSolve.jl's automatic algorithm choice.

## Arguments
- `A`: The matrix operator (can be a matrix, factorization, or abstract operator)
- `b`: The right-hand side vector
- `assumptions`: Operator assumptions including square matrix flag and conditioning

## Returns
A `DefaultLinearSolver` instance configured with the most appropriate algorithm choice,
or a specific algorithm instance for certain special cases.

## Algorithm Selection Logic

The function uses a hierarchy of dispatch rules based on:

1. **Matrix Type**: Special handling for structured matrices (Diagonal, Tridiagonal, etc.)
2. **Matrix Properties**: Square vs. rectangular, sparse vs. dense
3. **Hardware**: GPU vs. CPU arrays
4. **Conditioning**: Well-conditioned vs. ill-conditioned systems
5. **Size**: Small vs. large matrices for performance optimization

## Common Algorithm Choices

- **Diagonal matrices**: `DiagonalFactorization` for optimal O(n) performance
- **Tridiagonal/Bidiagonal**: Direct methods or specialized factorizations
- **Dense matrices**: LU, QR, or Cholesky based on structure and conditioning
- **Sparse matrices**: Specialized sparse factorizations (UMFPACK, KLU, etc.)
- **GPU arrays**: QR or LU factorizations optimized for GPU computation
- **Abstract operators**: Krylov methods (GMRES, CRAIGMR, LSMR)
- **Symmetric positive definite**: Cholesky factorization
- **Symmetric indefinite**: Bunch-Kaufman factorization

## Examples

```julia
# Dense square matrix - typically chooses LU
A = rand(100, 100)
b = rand(100)
alg = defaultalg(A, b, OperatorAssumptions(true))

# Overdetermined system - typically chooses QR  
A = rand(100, 50)
b = rand(100)
alg = defaultalg(A, b, OperatorAssumptions(false))

# Diagonal matrix - chooses diagonal factorization
A = Diagonal(rand(100))
alg = defaultalg(A, b, OperatorAssumptions(true))
```

## Notes
This function is primarily used internally by `solve(::LinearProblem)` when no
explicit algorithm is provided. For manual algorithm selection, users can
directly instantiate specific algorithm types.
"""
# Legacy fallback
# For SciML algorithms already using `defaultalg`, all assume square matrix.
defaultalg(A, b) = defaultalg(A, b, OperatorAssumptions(true))

function defaultalg(
        A::MatrixOperator, b,
        assump::OperatorAssumptions{Bool}
    )
    return defaultalg(A.A, b, assump)
end

function defaultalg(A, b, assump::OperatorAssumptions{Nothing})
    issq = issquare(A)
    return defaultalg(A, b, OperatorAssumptions(issq, assump.condition))
end

function defaultalg(A::SMatrix{S1, S2}, b, assump::OperatorAssumptions{Bool}) where {S1, S2}
    if S1 == S2
        return LUFactorization()
    else
        return SVDFactorization()  # QR(...) \ b is not defined currently
    end
end

function defaultalg(A::Tridiagonal, b, assump::OperatorAssumptions{Bool})
    return if assump.issq
        @static if VERSION >= v"1.11"
            DirectLdiv!()
        else
            DefaultLinearSolver(DefaultAlgorithmChoice.LUFactorization)
        end
    else
        DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
    end
end

function defaultalg(A::SymTridiagonal, b, ::OperatorAssumptions{Bool})
    return DefaultLinearSolver(DefaultAlgorithmChoice.LDLtFactorization)
end
function defaultalg(A::Bidiagonal, b, ::OperatorAssumptions{Bool})
    return @static if VERSION >= v"1.11"
        DirectLdiv!()
    else
        DefaultLinearSolver(DefaultAlgorithmChoice.LUFactorization)
    end
end
function defaultalg(A::Factorization, b, ::OperatorAssumptions{Bool})
    return DefaultLinearSolver(DefaultAlgorithmChoice.DirectLdiv!)
end
function defaultalg(A::Diagonal, b, ::OperatorAssumptions{Bool})
    return DefaultLinearSolver(DefaultAlgorithmChoice.DiagonalFactorization)
end

function defaultalg(A::Hermitian, b, ::OperatorAssumptions{Bool})
    return DefaultLinearSolver(DefaultAlgorithmChoice.CholeskyFactorization)
end

function defaultalg(A::Symmetric{<:Number, <:Array}, b, ::OperatorAssumptions{Bool})
    return DefaultLinearSolver(DefaultAlgorithmChoice.BunchKaufmanFactorization)
end

function defaultalg(A::GPUArraysCore.AnyGPUArray, b, assump::OperatorAssumptions{Bool})
    return if assump.condition === OperatorCondition.IllConditioned || !assump.issq
        DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
    else
        DefaultLinearSolver(DefaultAlgorithmChoice.LUFactorization)
    end
end

# A === nothing case
function defaultalg(
        A::Nothing, b::GPUArraysCore.AnyGPUArray, assump::OperatorAssumptions{Bool}
    )
    return if assump.condition === OperatorCondition.IllConditioned || !assump.issq
        DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
    else
        DefaultLinearSolver(DefaultAlgorithmChoice.LUFactorization)
    end
end

# Ambiguity handling
function defaultalg(
        A::GPUArraysCore.AnyGPUArray, b::GPUArraysCore.AnyGPUArray,
        assump::OperatorAssumptions{Bool}
    )
    return if assump.condition === OperatorCondition.IllConditioned || !assump.issq
        DefaultLinearSolver(DefaultAlgorithmChoice.QRFactorization)
    else
        DefaultLinearSolver(DefaultAlgorithmChoice.LUFactorization)
    end
end

function defaultalg(
        A::SciMLOperators.AbstractSciMLOperator, b,
        assump::OperatorAssumptions{Bool}
    )
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

# Fix ambiguity
function defaultalg(
        A::SciMLOperators.AbstractSciMLOperator, b::GPUArraysCore.AnyGPUArray,
        assump::OperatorAssumptions{Bool}
    )
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

userecursivefactorization(A) = false

"""
    get_tuned_algorithm(::Type{eltype_A}, ::Type{eltype_b}, matrix_size) where {eltype_A, eltype_b}

Get the tuned algorithm preference for the given element type and matrix size.
Returns `nothing` if no preference exists. Uses preloaded constants for efficiency.
Fast path when no preferences are set.
"""
@inline function get_tuned_algorithm(
        ::Type{eltype_A}, ::Type{eltype_b}, matrix_size::Integer
    ) where {eltype_A, eltype_b}
    # Determine the element type to use for preference lookup
    target_eltype = eltype_A !== Nothing ? eltype_A : eltype_b

    # Determine size category based on matrix size (matching LinearSolveAutotune categories)
    size_category = if matrix_size <= 20
        :tiny
    elseif matrix_size <= 100
        :small
    elseif matrix_size <= 300
        :medium
    elseif matrix_size <= 1000
        :large
    else
        :big
    end

    # Fast path: if no preferences are set, return nothing immediately
    AUTOTUNE_PREFS_SET || return nothing

    # Look up the tuned algorithm from preloaded constants with type specialization
    return _get_tuned_algorithm_impl(target_eltype, size_category)
end

# Type-specialized implementation with availability checking and fallback logic
@inline function _get_tuned_algorithm_impl(::Type{Float32}, size_category::Symbol)
    prefs = getproperty(AUTOTUNE_PREFS.Float32, size_category)
    return _choose_available_algorithm(prefs)
end

@inline function _get_tuned_algorithm_impl(::Type{Float64}, size_category::Symbol)
    prefs = getproperty(AUTOTUNE_PREFS.Float64, size_category)
    return _choose_available_algorithm(prefs)
end

@inline function _get_tuned_algorithm_impl(::Type{ComplexF32}, size_category::Symbol)
    prefs = getproperty(AUTOTUNE_PREFS.ComplexF32, size_category)
    return _choose_available_algorithm(prefs)
end

@inline function _get_tuned_algorithm_impl(::Type{ComplexF64}, size_category::Symbol)
    prefs = getproperty(AUTOTUNE_PREFS.ComplexF64, size_category)
    return _choose_available_algorithm(prefs)
end

@inline _get_tuned_algorithm_impl(::Type, ::Symbol) = nothing  # Fallback for other types

# Convenience method for when A is nothing - delegate to main implementation
@inline get_tuned_algorithm(
    ::Type{Nothing},
    ::Type{eltype_b},
    matrix_size::Integer
) where {eltype_b} = get_tuned_algorithm(eltype_b, eltype_b, matrix_size)

# Allows A === nothing as a stand-in for dense matrix
function defaultalg(A, b, assump::OperatorAssumptions{Bool})
    alg = if assump.issq
        # Special case on Arrays: avoid BLAS for RecursiveFactorization.jl when
        # it makes sense according to the benchmarks, which is dependent on
        # whether MKL or OpenBLAS is being used
        if (A === nothing && !(b isa GPUArraysCore.AnyGPUArray)) || A isa Matrix
            if (
                    A === nothing ||
                        eltype(A) <: BLASELTYPES
                ) &&
                    ArrayInterface.can_setindex(b) &&
                    (
                    __conditioning(assump) === OperatorCondition.IllConditioned ||
                        __conditioning(assump) === OperatorCondition.WellConditioned
                )

                # Small matrix override - always use GenericLUFactorization for tiny problems
                if length(b) <= 10
                    DefaultAlgorithmChoice.GenericLUFactorization
                else
                    # Check if autotune preferences exist for larger matrices
                    matrix_size = length(b)
                    eltype_A = A === nothing ? Nothing : eltype(A)
                    tuned_alg = get_tuned_algorithm(eltype_A, eltype(b), matrix_size)

                    if tuned_alg !== nothing
                        tuned_alg
                    elseif appleaccelerate_isavailable() && b isa Array &&
                            eltype(b) <: Union{Float32, Float64, ComplexF32, ComplexF64}
                        DefaultAlgorithmChoice.AppleAccelerateLUFactorization
                    elseif (
                            length(b) <= 100 || (isopenblas() && length(b) <= 500) ||
                                (usemkl && length(b) <= 200)
                        ) &&
                            (
                            A === nothing ? eltype(b) <: Union{Float32, Float64} :
                                eltype(A) <: Union{Float32, Float64}
                        ) &&
                            userecursivefactorization(A)
                        DefaultAlgorithmChoice.RFLUFactorization
                        #elseif A === nothing || A isa Matrix
                        #    alg = FastLUFactorization()
                    elseif usemkl && b isa Array &&
                            eltype(b) <: Union{Float32, Float64, ComplexF32, ComplexF64}
                        DefaultAlgorithmChoice.MKLLUFactorization
                    else
                        DefaultAlgorithmChoice.LUFactorization
                    end
                end
            elseif __conditioning(assump) === OperatorCondition.VeryIllConditioned
                DefaultAlgorithmChoice.QRFactorization
            elseif __conditioning(assump) === OperatorCondition.SuperIllConditioned
                DefaultAlgorithmChoice.SVDFactorization
            elseif usemkl && (
                    A === nothing ? eltype(b) <: BLASELTYPES :
                        eltype(A) <: BLASELTYPES
                )
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
    return DefaultLinearSolver(alg)
end

function algchoice_to_alg(alg::Symbol)
    return if alg === :SVDFactorization
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
        SparspakFactorization(throwerror = false)
    elseif alg === :KLUFactorization
        KLUFactorization()
    elseif alg === :UMFPACKFactorization
        UMFPACKFactorization()
    elseif alg === :KrylovJL_GMRES
        KrylovJL_GMRES()
    elseif alg === :GenericLUFactorization
        GenericLUFactorization()
    elseif alg === :RFLUFactorization
        RFLUFactorization(throwerror = false)
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
    elseif alg === :BLISLUFactorization
        BLISLUFactorization(throwerror = false)
    elseif alg === :CudaOffloadLUFactorization
        CudaOffloadLUFactorization(throwerror = false)
    elseif alg === :MetalLUFactorization
        MetalLUFactorization(throwerror = false)
    else
        error("Algorithm choice symbol $alg not allowed in the default")
    end
end

## Catch high level interface

function SciMLBase.init(
        prob::LinearProblem, alg::Nothing,
        args...;
        assumptions = OperatorAssumptions(issquare(prob.A)),
        kwargs...
    )
    return SciMLBase.init(
        prob, defaultalg(prob.A, prob.b, assumptions), args...; assumptions, kwargs...
    )
end

function SciMLBase.solve!(
        cache::LinearCache, alg::Nothing,
        args...; assump::OperatorAssumptions = OperatorAssumptions(),
        kwargs...
    )
    (; A, b) = cache
    return SciMLBase.solve!(cache, defaultalg(A, b, assump), args...; kwargs...)
end

function init_cacheval(
        alg::Nothing, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assump::OperatorAssumptions
    )
    return init_cacheval(
        defaultalg(A, b, assump), A, b, u, Pl, Pr, maxiters, abstol, reltol,
        verbose,
        assump
    )
end

"""
cache.cacheval = NamedTuple(LUFactorization = cache of LUFactorization, ...)
"""
function init_cacheval(
        alg::DefaultLinearSolver, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assump::OperatorAssumptions
    )
    return _init_default_cacheval(
        alg, A, b, u, Pl, Pr, maxiters, abstol, reltol,
        verbose, assump, A
    )
end

function init_cacheval(
        alg::DefaultLinearSolver, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assump::OperatorAssumptions,
        A_original
    )
    return _init_default_cacheval(
        alg, A, b, u, Pl, Pr, maxiters, abstol, reltol,
        verbose, assump, A_original
    )
end

@generated function _init_default_cacheval(
        alg::DefaultLinearSolver, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assump::OperatorAssumptions,
        A_original
    )
    caches = map(first.(EnumX.symbol_map(DefaultAlgorithmChoice.T))) do alg
        if alg === :KrylovJL_GMRES || alg === :KrylovJL_CRAIGMR || alg === :KrylovJL_LSMR
            quote
                if A isa Matrix || issparsematrixcsc(A)
                    nothing
                else
                    init_cacheval(
                        $(algchoice_to_alg(alg)), A, b, u, Pl, Pr, maxiters,
                        abstol, reltol,
                        verbose,
                        assump
                    )
                end
            end
        else
            quote
                init_cacheval(
                    $(algchoice_to_alg(alg)), A, b, u, Pl, Pr, maxiters, abstol,
                    reltol,
                    verbose,
                    assump
                )
            end
        end
    end
    return Expr(:call, :DefaultLinearSolverInit, caches..., :A_original)
end

function defaultalg_symbol(::Type{T}) where {T}
    return Base.typename(SciMLBase.parameterless_type(T)).name
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
@generated function SciMLBase.solve!(
        cache::LinearCache, alg::DefaultLinearSolver,
        args...;
        assump::OperatorAssumptions = OperatorAssumptions(),
        kwargs...
    )
    ex = :()
    for alg in first.(EnumX.symbol_map(DefaultAlgorithmChoice.T))
        if alg in Symbol.(
                (
                    DefaultAlgorithmChoice.LUFactorization,
                    DefaultAlgorithmChoice.MKLLUFactorization,
                    DefaultAlgorithmChoice.AppleAccelerateLUFactorization,
                    DefaultAlgorithmChoice.GenericLUFactorization,
                )
            )
            newex = quote
                sol = SciMLBase.solve!(cache, $(algchoice_to_alg(alg)), args...; kwargs...)
                if sol.retcode === ReturnCode.Failure && alg.safetyfallback
                    if cache.A === cache.cacheval.A_backup
                        @SciMLMessage(
                            "LU factorization failed but cannot safely fall back to QR: `alias_A` is set so the original matrix `A` is not available as a backup to restore after in-place LU modification. Set `alias_A=false` (the default) to enable safe fallbacks.",
                            cache.verbose, :default_lu_fallback
                        )
                        SciMLBase.build_linear_solution(
                            alg, sol.u, sol.resid, sol.cache;
                            retcode = sol.retcode,
                            iters = sol.iters, stats = sol.stats
                        )
                    else
                        @SciMLMessage(
                            "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.",
                            cache.verbose, :default_lu_fallback
                        )
                        copyto!(cache.A, cache.cacheval.A_backup)
                        cache.isfresh = true
                        sol = SciMLBase.solve!(
                            cache, QRFactorization(ColumnNorm()), args...; kwargs...
                        )
                        SciMLBase.build_linear_solution(
                            alg, sol.u, sol.resid, sol.cache;
                            retcode = sol.retcode,
                            iters = sol.iters, stats = sol.stats
                        )
                    end
                else
                    SciMLBase.build_linear_solution(
                        alg, sol.u, sol.resid, sol.cache;
                        retcode = sol.retcode,
                        iters = sol.iters, stats = sol.stats
                    )
                end
            end
        elseif alg == Symbol(DefaultAlgorithmChoice.RFLUFactorization)
            newex = quote
                if !userecursivefactorization(nothing)
                    error("Default algorithm calling solve on RecursiveFactorization without the package being loaded. This shouldn't happen.")
                end

                sol = SciMLBase.solve!(cache, $(algchoice_to_alg(alg)), args...; kwargs...)
                if sol.retcode === ReturnCode.Failure && alg.safetyfallback
                    if cache.A === cache.cacheval.A_backup
                        @SciMLMessage(
                            "LU factorization failed but cannot safely fall back to QR: `alias_A` is set so the original matrix `A` is not available as a backup to restore after in-place LU modification. Set `alias_A=false` (the default) to enable safe fallbacks.",
                            cache.verbose, :default_lu_fallback
                        )
                        SciMLBase.build_linear_solution(
                            alg, sol.u, sol.resid, sol.cache;
                            retcode = sol.retcode,
                            iters = sol.iters, stats = sol.stats
                        )
                    else
                        @SciMLMessage(
                            "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.",
                            cache.verbose, :default_lu_fallback
                        )
                        copyto!(cache.A, cache.cacheval.A_backup)
                        cache.isfresh = true
                        sol = SciMLBase.solve!(
                            cache, QRFactorization(ColumnNorm()), args...; kwargs...
                        )
                        SciMLBase.build_linear_solution(
                            alg, sol.u, sol.resid, sol.cache;
                            retcode = sol.retcode,
                            iters = sol.iters, stats = sol.stats
                        )
                    end
                else
                    SciMLBase.build_linear_solution(
                        alg, sol.u, sol.resid, sol.cache;
                        retcode = sol.retcode,
                        iters = sol.iters, stats = sol.stats
                    )
                end
            end
        elseif alg == Symbol(DefaultAlgorithmChoice.BLISLUFactorization)
            newex = quote
                if !useblis(nothing)
                    error("Default algorithm calling solve on BLISLUFactorization without the extension being loaded. This shouldn't happen.")
                end

                sol = SciMLBase.solve!(cache, $(algchoice_to_alg(alg)), args...; kwargs...)
                if sol.retcode === ReturnCode.Failure && alg.safetyfallback
                    if cache.A === cache.cacheval.A_backup
                        @SciMLMessage(
                            "LU factorization failed but cannot safely fall back to QR: `alias_A` is set so the original matrix `A` is not available as a backup to restore after in-place LU modification. Set `alias_A=false` (the default) to enable safe fallbacks.",
                            cache.verbose, :default_lu_fallback
                        )
                        SciMLBase.build_linear_solution(
                            alg, sol.u, sol.resid, sol.cache;
                            retcode = sol.retcode,
                            iters = sol.iters, stats = sol.stats
                        )
                    else
                        @SciMLMessage(
                            "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.",
                            cache.verbose, :default_lu_fallback
                        )
                        copyto!(cache.A, cache.cacheval.A_backup)
                        cache.isfresh = true
                        sol = SciMLBase.solve!(
                            cache, QRFactorization(ColumnNorm()), args...; kwargs...
                        )
                        SciMLBase.build_linear_solution(
                            alg, sol.u, sol.resid, sol.cache;
                            retcode = sol.retcode,
                            iters = sol.iters, stats = sol.stats
                        )
                    end
                else
                    SciMLBase.build_linear_solution(
                        alg, sol.u, sol.resid, sol.cache;
                        retcode = sol.retcode,
                        iters = sol.iters, stats = sol.stats
                    )
                end
            end
        elseif alg == Symbol(DefaultAlgorithmChoice.CudaOffloadLUFactorization)
            newex = quote
                if !usecuda(nothing)
                    error("Default algorithm calling solve on CudaOffloadLUFactorization without CUDA.jl being loaded. This shouldn't happen.")
                end

                sol = SciMLBase.solve!(cache, $(algchoice_to_alg(alg)), args...; kwargs...)
                if sol.retcode === ReturnCode.Failure && alg.safetyfallback
                    if cache.A === cache.cacheval.A_backup
                        @SciMLMessage(
                            "LU factorization failed but cannot safely fall back to QR: `alias_A` is set so the original matrix `A` is not available as a backup to restore after in-place LU modification. Set `alias_A=false` (the default) to enable safe fallbacks.",
                            cache.verbose, :default_lu_fallback
                        )
                        SciMLBase.build_linear_solution(
                            alg, sol.u, sol.resid, sol.cache;
                            retcode = sol.retcode,
                            iters = sol.iters, stats = sol.stats
                        )
                    else
                        @SciMLMessage(
                            "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.",
                            cache.verbose, :default_lu_fallback
                        )
                        copyto!(cache.A, cache.cacheval.A_backup)
                        cache.isfresh = true
                        sol = SciMLBase.solve!(
                            cache, QRFactorization(ColumnNorm()), args...; kwargs...
                        )
                        SciMLBase.build_linear_solution(
                            alg, sol.u, sol.resid, sol.cache;
                            retcode = sol.retcode,
                            iters = sol.iters, stats = sol.stats
                        )
                    end
                else
                    SciMLBase.build_linear_solution(
                        alg, sol.u, sol.resid, sol.cache;
                        retcode = sol.retcode,
                        iters = sol.iters, stats = sol.stats
                    )
                end
            end
        elseif alg == Symbol(DefaultAlgorithmChoice.MetalLUFactorization)
            newex = quote
                if !usemetal(nothing)
                    error("Default algorithm calling solve on MetalLUFactorization without Metal.jl being loaded. This shouldn't happen.")
                end

                sol = SciMLBase.solve!(cache, $(algchoice_to_alg(alg)), args...; kwargs...)
                if sol.retcode === ReturnCode.Failure && alg.safetyfallback
                    if cache.A === cache.cacheval.A_backup
                        @SciMLMessage(
                            "LU factorization failed but cannot safely fall back to QR: `alias_A` is set so the original matrix `A` is not available as a backup to restore after in-place LU modification. Set `alias_A=false` (the default) to enable safe fallbacks.",
                            cache.verbose, :default_lu_fallback
                        )
                        SciMLBase.build_linear_solution(
                            alg, sol.u, sol.resid, sol.cache;
                            retcode = sol.retcode,
                            iters = sol.iters, stats = sol.stats
                        )
                    else
                        @SciMLMessage(
                            "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.",
                            cache.verbose, :default_lu_fallback
                        )
                        copyto!(cache.A, cache.cacheval.A_backup)
                        cache.isfresh = true
                        sol = SciMLBase.solve!(
                            cache, QRFactorization(ColumnNorm()), args...; kwargs...
                        )
                        SciMLBase.build_linear_solution(
                            alg, sol.u, sol.resid, sol.cache;
                            retcode = sol.retcode,
                            iters = sol.iters, stats = sol.stats
                        )
                    end
                else
                    SciMLBase.build_linear_solution(
                        alg, sol.u, sol.resid, sol.cache;
                        retcode = sol.retcode,
                        iters = sol.iters, stats = sol.stats
                    )
                end
            end
        else
            newex = quote
                sol = SciMLBase.solve!(cache, $(algchoice_to_alg(alg)), args...; kwargs...)
                SciMLBase.build_linear_solution(
                    alg, sol.u, sol.resid, sol.cache;
                    retcode = sol.retcode,
                    iters = sol.iters, stats = sol.stats
                )
            end
        end
        alg_enum = getproperty(LinearSolve.DefaultAlgorithmChoice, alg)
        ex = if ex == :()
            Expr(
                :elseif, :(alg.alg == $(alg_enum)), newex,
                :(error("Algorithm Choice not Allowed"))
            )
        else
            Expr(:elseif, :(alg.alg == $(alg_enum)), newex, ex)
        end
    end
    return ex = Expr(:if, ex.args...)
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
        newex = if alg in Symbol.(
                (
                    DefaultAlgorithmChoice.RFLUFactorization,
                    DefaultAlgorithmChoice.GenericLUFactorization,
                )
            )
            quote
                getproperty(cache.cacheval, $(Meta.quot(alg)))[1]' \ dy
            end
        elseif alg == Symbol(DefaultAlgorithmChoice.MKLLUFactorization)
            quote
                A = getproperty(cache.cacheval, $(Meta.quot(alg)))[1]
                getrs!('T', A.factors, A.ipiv, dy)
            end
        elseif alg == Symbol(DefaultAlgorithmChoice.AppleAccelerateLUFactorization)
            quote
                A = getproperty(cache.cacheval, $(Meta.quot(alg)))[1]
                aa_getrs!('T', A.factors, A.ipiv, dy)
            end
        elseif alg in Symbol.(
                (
                    DefaultAlgorithmChoice.LUFactorization,
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
                )
            )
            quote
                getproperty(cache.cacheval, $(Meta.quot(alg)))' \ dy
            end
        elseif alg in Symbol.(
                (
                    DefaultAlgorithmChoice.KrylovJL_GMRES, DefaultAlgorithmChoice.KrylovJL_LSMR,
                    DefaultAlgorithmChoice.KrylovJL_CRAIGMR,
                )
            )
            quote
                invprob = LinearSolve.LinearProblem(transpose(cache.A), dy)
                solve(
                    invprob, cache.alg;
                    abstol = cache.val.abstol,
                    reltol = cache.val.reltol,
                    verbose = cache.val.verbose
                )
            end
        else
            quote
                error("Default linear solver with algorithm $(alg) is currently not supported by Enzyme rules on LinearSolve.jl. Please open an issue on LinearSolve.jl detailing which algorithm is missing the adjoint handling")
            end
        end

        ex = if ex == :()
            Expr(
                :elseif,
                :(getproperty(DefaultAlgorithmChoice, $(Meta.quot(alg))) === cache.alg.alg),
                newex,
                :(error("Algorithm Choice not Allowed"))
            )
        else
            Expr(
                :elseif,
                :(getproperty(DefaultAlgorithmChoice, $(Meta.quot(alg))) === cache.alg.alg),
                newex,
                ex
            )
        end
    end
    return ex = Expr(:if, ex.args...)
end
