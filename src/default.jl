needs_concrete_A(alg::DefaultLinearSolver) = true

# Every algorithm the default can dispatch to either ignores the tolerances
# (the factorizations) or reads `cache.abstol`/`cache.reltol` at solve time (the
# Krylov slots), so updating the `LinearCache` fields is enough.
update_tolerances_internal!(cache, ::DefaultLinearSolver, abstol, reltol) = nothing

mutable struct DefaultLinearSolverInit{
        T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
        T13, T14, T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26,
        TA, Tb, TR,
    }
    LUFactorization::T1
    QRFactorization::T2
    DiagonalFactorization::T3
    DirectLdiv!::T4
    SparspakFactorization::T5
    KLUFactorization::T6
    SupernodalLUFactorization::T7
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
    SparseColumnPivotedQRFactorization::T25
    LHLFactorization::T26
    A_backup::TA  # backup of cache.A for restoring after in-place LU and QR fallback
    residual_buf::Tb  # pre-allocated buffer for post-solve residual check (same size/type as b)
    a_backup_synced::Bool  # true if A_backup content matches cache.A (before LU modifies it)
    a_backup_allocated::Bool  # true once A_backup has been replaced with a private buffer
    fell_back_to_qr::Bool  # true after QR fallback; reuse QR until matrix is refreshed
    # Persistent-nonstructural-zero reduction state, shared across the sparse
    # sub-algorithm slots (the two sparse LU slots + the QR fallback all factor
    # matrix). `nothing` when inactive / non-sparse. See `init_sparse_reduction`.
    sparse_reduction::TR
end

function resize_cacheval!(cache, cacheval::DefaultLinearSolverInit, i)
    resize_cacheval!(cache, cacheval.GenericLUFactorization, i)
    A_backup = cacheval.A_backup
    return if A_backup isa AbstractMatrix
        setfield!(cacheval, :A_backup, similar(A_backup, i, i))
        cacheval.a_backup_allocated = true
        cacheval.a_backup_synced = false
    end
end

function update_cacheval!(cache, cacheval::DefaultLinearSolverInit, name::Symbol, A)
    name === :A && update_cacheval!(cache, cacheval.GenericLUFactorization, name, A)
    return cacheval
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

# Legacy fallback
# For SciML algorithms already using `defaultalg`, all assume square matrix.
defaultalg(A, b) = defaultalg(A, b, OperatorAssumptions(true))

"""
    defaultalg(A, b, assumptions::OperatorAssumptions)

Select a default linear solver algorithm for the operator `A`, right-hand side
`b`, and operator assumptions. This is the dispatch point used by
`solve(::LinearProblem)` when no algorithm is supplied explicitly.

# Arguments

- `A`: Matrix, factorization, or abstract operator to solve with.
- `b`: Right-hand side vector or matrix.
- `assumptions`: [`OperatorAssumptions`](@ref) describing whether `A` is square,
  its conditioning, and its structural-zero behavior.

# Returns

A concrete [`SciMLLinearSolveAlgorithm`](@ref), usually wrapped in a
[`DefaultLinearSolver`](@ref), selected for the input representation and
available extensions.

# Examples

```julia
A = rand(100, 100)
b = rand(100)
alg = defaultalg(A, b, OperatorAssumptions(true))
solve(LinearProblem(A, b), alg)
```

For an abstract matrix-free operator, the default is a Krylov algorithm when
the operator does not provide a direct solve method:

```julia
A = SciMLOperators.MatrixOperator(rand(10, 10))
b = rand(10)
alg = defaultalg(A, b, OperatorAssumptions(true))
```

# Notes

The two-argument form assumes a square system. Use an explicit algorithm when
the application requires a particular factorization or iterative method.
"""
function defaultalg(
        A::MatrixOperator, b,
        assump::OperatorAssumptions{Bool}
    )
    return defaultalg(A.A, b, assump)
end

# Fix ambiguity with the `AbstractSciMLOperator`/`AnyGPUArray` method below: a
# `MatrixOperator` with a GPU `b` matches both, and neither is more specific.
# Unwrapping to `A.A` is what the `MatrixOperator` method above does, and it is the
# better answer here too -- a concretized operator does not need the operator-only
# Krylov fallback.
function defaultalg(
        A::MatrixOperator, b::GPUArraysCore.AnyGPUArray,
        assump::OperatorAssumptions{Bool}
    )
    return defaultalg(A.A, b, assump)
end

function defaultalg(A, b, assump::OperatorAssumptions{Nothing})
    issq = issquare(A)
    return defaultalg(
        A, b,
        OperatorAssumptions(
            issq; condition = assump.condition,
            nonstructural_zeros = assump.nonstructural_zeros
        )
    )
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

"""
    LinearSolve.LHL_DEFAULT_MIN_SIZE

Smallest `n` at which `defaultalg` prefers `LHLFactorization` for a split `WOperator`.

Measured, not derived. Costing one γ-cycle — a re-shift plus the ~14 solves it serves, with
the reduction amortized over the ~55 γ a Jacobian serves, both ratios taken from
instrumented BDF runs — against a fresh LU plus its solves, the default `refine = 1`
crosses at `n ≈ 16` and reaches a 1.4× margin by `n = 32`; `refine = 0` wins from `n = 4`.

That model is per-operation. End to end it is optimistic near the cutoff, because a real
problem need not hit those two ratios: measured on stiff ODE runs, `refine = 1` is roughly
break-even between `n = 32` and `n = 40` (0.98×–1.47×) and only pulls clear above
`n ≈ 100`. The cutoff is therefore where the algorithm stops *losing*, not where it starts
winning; the win grows with `n`, reaching 2–5× by `n = 800`.
"""
const LHL_DEFAULT_MIN_SIZE = 32

# A `WOperator` holds `J` and `gamma` apart, which is the statement that the shift will
# move while `J` stays put. When its Jacobian is a dense matrix, the algorithm that
# exploits that split is the right default; a matrix-free `J` still wants Krylov.
# `defaultalg` and the cacheval initializer must agree exactly: if the slot is selected
# but left uninitialized the solve fails, and if it is initialized but never selected the
# buffers are wasted. Both ask here.
function _lhl_defaultable(A::WOperator, assump::OperatorAssumptions)
    # Only a plain dense `J`, or a plain sparse `J` the sparse block-triangular solver is
    # expected to win on (a reducible pattern — see `lhl_prefers_sparse`). An operator `J`
    # — a `MatrixOperator` in particular — is updated in place by `update_coefficients!`,
    # which moves the numbers while leaving both the object identity and `jac_stale`
    # untouched, so the reduction cannot tell it went stale and would silently answer with
    # the previous Jacobian, so those are excluded.
    (assump.issq && size(A, 1) >= LHL_DEFAULT_MIN_SIZE && _lhl_scalar_massmatrix(A.mass_matrix)) ||
        return false
    A.J isa DenseMatrix && return true
    return issparsematrixcsc(A.J) && lhl_prefers_sparse(A.J)
end
_lhl_defaultable(A, assump) = false

function defaultalg(A::WOperator, b, assump::OperatorAssumptions{Bool})
    if _lhl_defaultable(A, assump)
        # Refinement is the right default for a bare linear solve; a caller running an
        # outer correction loop should ask for `LHLFactorization(refine = 0)`.
        return DefaultLinearSolver(DefaultAlgorithmChoice.LHLFactorization)
    end
    # Everything else — a matrix-free Jacobian, a general mass matrix, or a problem too
    # small to pay for the reduction — keeps whatever the operator path already chose.
    return @invoke defaultalg(A::SciMLOperators.AbstractSciMLOperator, b, assump)
end

# Routed to the operator default rather than to the `WOperator` method above, because
# that one can answer `LHLFactorization`. The reduction behind it lives in
# LHLFactorization.jl and runs on the host, while a GPU array still satisfies
# `DenseMatrix` (`CuArray <: DenseArray`), so `_lhl_defaultable` would accept a device
# Jacobian and the solve would then die on scalar indexing.
function defaultalg(
        A::WOperator, b::GPUArraysCore.AnyGPUArray, assump::OperatorAssumptions{Bool}
    )
    return @invoke defaultalg(
        A::SciMLOperators.AbstractSciMLOperator, b::GPUArraysCore.AnyGPUArray,
        assump::OperatorAssumptions{Bool}
    )
end

_lhl_scalar_massmatrix(::UniformScaling) = true
_lhl_scalar_massmatrix(::Number) = true
_lhl_scalar_massmatrix(::Any) = false

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
        if (A === nothing && !(b isa GPUArraysCore.AnyGPUArray)) || A isa DenseMatrix
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
                # `size(b, 1)` (not `length(b)`) so batched (matrix) right-hand
                # sides don't inflate the apparent problem size.
                if size(b, 1) <= 10
                    DefaultAlgorithmChoice.GenericLUFactorization
                else
                    # Check if autotune preferences exist for larger matrices
                    matrix_size = size(b, 1)
                    eltype_A = A === nothing ? Nothing : eltype(A)
                    tuned_alg = get_tuned_algorithm(eltype_A, eltype(b), matrix_size)

                    if tuned_alg !== nothing
                        tuned_alg
                    elseif appleaccelerate_isavailable() &&
                            b isa DenseArray && !(b isa GPUArraysCore.AnyGPUArray) &&
                            eltype(b) <: Union{Float32, Float64, ComplexF32, ComplexF64}
                        DefaultAlgorithmChoice.AppleAccelerateLUFactorization
                    elseif (
                            size(b, 1) <= 100 || (isopenblas() && size(b, 1) <= 500) ||
                                (usemkl && size(b, 1) <= 200)
                        ) &&
                            (
                            A === nothing ? eltype(b) <: Union{Float32, Float64} :
                                eltype(A) <: Union{Float32, Float64}
                        ) &&
                            userecursivefactorization(A)
                        DefaultAlgorithmChoice.RFLUFactorization
                        #elseif A === nothing || A isa Matrix
                        #    alg = FastLUFactorization()
                        # Blocked generic_lufact! beats vendor getrf ≥ 2x through N = 32
                        # everywhere, and through 256 vs OpenBLAS (badly tuned small-N
                        # threading — same fact the RFLU 500 band above encodes).
                    elseif (
                            matrix_size <= 32 ||
                                (isopenblas() && matrix_size <= 256)
                        ) &&
                            (
                            A === nothing ? eltype(b) <: Union{Float32, Float64} :
                                eltype(A) <: Union{Float32, Float64}
                        )
                        DefaultAlgorithmChoice.GenericLUFactorization
                    elseif usemkl &&
                            b isa DenseArray && !(b isa GPUArraysCore.AnyGPUArray) &&
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
        # The default polyalgorithm's "KLU" slot resolves to the pure-Julia
        # PureKLU. The SuiteSparse `KLUFactorization` is unchanged and remains
        # available when requested explicitly.
        PureKLUFactorization()
    elseif alg === :SupernodalLUFactorization
        SupernodalLUFactorization()
    elseif alg === :LHLFactorization
        LHLFactorization()
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
    elseif alg === :SparseColumnPivotedQRFactorization
        SparseColumnPivotedQRFactorization()
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
    # Promote integer-eltype problems before choosing the algorithm, so the choice
    # and the cache agree on the types; see `__promote_int_problem` in common.jl.
    prob = __promote_int_problem(prob, nothing)
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
        if alg === :LHLFactorization
            # Three n×n buffers, wanted only by the split form. Every other `A` — which is
            # nearly every `A` — would pay for them and never use the slot.
            quote
                if _lhl_defaultable(A, assump)
                    init_cacheval(
                        $(algchoice_to_alg(alg)), A, b, u, Pl, Pr, maxiters, abstol,
                        reltol, verbose, assump
                    )
                else
                    nothing
                end
            end
        elseif alg === :KrylovJL_GMRES || alg === :KrylovJL_CRAIGMR || alg === :KrylovJL_LSMR
            quote
                if A isa DenseMatrix || issparsematrixcsc(A)
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
    return Expr(
        :call, :DefaultLinearSolverInit, caches...,
        :A_original, :(similar(b)), true, false, false,
        :(init_sparse_reduction(A, assump))
    )
end

function defaultalg_symbol(::Type{T}) where {T}
    return Base.typename(SciMLBase.parameterless_type(T)).name
end
defaultalg_symbol(::Type{<:GenericFactorization{typeof(ldlt!)}}) = :LDLtFactorization

defaultalg_symbol(::Type{<:QRFactorization{ColumnNorm}}) = :QRFactorizationPivoted

const _SPARSE_ONLY_ALGORITHMS = Symbol.(
    (
        DefaultAlgorithmChoice.KLUFactorization,
        DefaultAlgorithmChoice.SupernodalLUFactorization,
        DefaultAlgorithmChoice.SparspakFactorization,
        DefaultAlgorithmChoice.CHOLMODFactorization,
        DefaultAlgorithmChoice.SparseColumnPivotedQRFactorization,
    )
)

# Sparse LU algorithms (i.e., not CHOLMOD) that should fall back to SPQR
# (column-pivoted sparse QR) when factorization fails or produces non-finite
# values. Mirrors the dense LU → QR fallback handled by
# `_default_lu_solve_with_fallback`.
const _SPARSE_LU_FALLBACK_ALGORITHMS = Symbol.(
    (
        DefaultAlgorithmChoice.KLUFactorization,
        DefaultAlgorithmChoice.SupernodalLUFactorization,
        DefaultAlgorithmChoice.SparspakFactorization,
    )
)

_qr_fallback_pivot(A::GPUArraysCore.AnyGPUArray) = NoPivot()
function _qr_fallback_pivot(A)
    if _is_gpu_sparse(A)
        return NoPivot()
    else
        return ColumnNorm()
    end
end

function _is_gpu_sparse(A)
    hasfield(typeof(A), :nzVal) && return A.nzVal isa GPUArraysCore.AnyGPUArray
    hasfield(typeof(A), :rowVal) && return A.rowVal isa GPUArraysCore.AnyGPUArray
    return false
end

"""
    _do_qr_fallback(cache::LinearCache, alg, sol, reason::Symbol)

Perform QR fallback after LU failure or residual check failure. Restores `cache.A`
from `A_backup` (since LU may have modified it in-place) and solves with column-pivoted QR
(or NoPivot for GPU arrays which don't support scalar indexing).
`reason` is `:lu_failure` or `:residual_check` for appropriate log messages.
"""
function _do_qr_fallback(cache::LinearCache, alg, sol, reason::Symbol)
    # Always extract solution data from `cache` rather than `sol`. The QR
    # fallback path calls `solve!(cache, QRFactorization(...))` recursively;
    # during precompile inference, that inner call's return type gets capped
    # to a non-concrete UnionAll (Julia's inference complexity limit). Reading
    # `cache.u` (statically typed) and using `cache` for the solution cache
    # field keeps the return type of this helper concrete, which propagates
    # up through `_default_lu_solve_with_fallback` and the @generated
    # `solve!(cache, ::DefaultLinearSolver)` body.
    rc = sol.retcode
    iters = sol.iters
    if is_cusparse(cache.A)
        @SciMLMessage(
            "LU factorization failed for GPU sparse matrix but QR fallback is not supported for CuSparse. Returning LU failure.",
            cache.verbose, :default_lu_fallback
        )
        return SciMLBase.build_linear_solution(
            alg, cache.u, nothing, nothing; retcode = rc, iters = iters, stats = nothing
        )
    end
    if cache.A === cache.cacheval.A_backup
        @SciMLMessage(
            "LU factorization failed but cannot safely fall back to QR: `alias_A` is set so the original matrix `A` is not available as a backup to restore after in-place LU modification. Set `alias_A=false` (the default) to enable safe fallbacks.",
            cache.verbose, :default_lu_fallback
        )
        return SciMLBase.build_linear_solution(
            alg, cache.u, nothing, nothing; retcode = rc, iters = iters, stats = nothing
        )
    end
    if reason === :residual_check
        @SciMLMessage(
            "LU solve residual check failed, falling back to QR factorization. `A` is potentially ill-conditioned.",
            cache.verbose, :default_lu_fallback
        )
    elseif reason === :qr_rank_deficient
        @SciMLMessage(
            "Unpivoted QR detected a rank-deficient `A`, falling back to column-pivoted QR so the least-squares solution matches `A \\ b`.",
            cache.verbose, :default_lu_fallback
        )
    else
        @SciMLMessage(
            "LU factorization failed, falling back to QR factorization. `A` is potentially rank-deficient.",
            cache.verbose, :default_lu_fallback
        )
    end
    copyto!(cache.A, cache.cacheval.A_backup)
    cache.isfresh = true
    pivot = _qr_fallback_pivot(cache.A)
    qr_sol = SciMLBase.solve!(cache, QRFactorization(pivot))
    cache.cacheval.fell_back_to_qr = true
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, nothing;
        retcode = qr_sol.retcode, iters = qr_sol.iters, stats = nothing
    )
end

"""
    _reuse_qr_fallback(cache::LinearCache, alg)

Reuse the cached QR factorization from a previous QR fallback. Called when
`fell_back_to_qr` is `true` and `isfresh` is `false`, meaning the matrix hasn't
changed since the QR fallback and we should keep using QR instead of the
(potentially corrupted) LU factorization.
"""
function _reuse_qr_fallback(cache::LinearCache, alg)
    pivot = _qr_fallback_pivot(cache.A)
    qr_sol = SciMLBase.solve!(cache, QRFactorization(pivot))
    # Use cache directly for type-stable inference (see _do_qr_fallback).
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, nothing;
        retcode = qr_sol.retcode, iters = qr_sol.iters, stats = nothing
    )
end

"""
    _do_sparse_qr_fallback(cache::LinearCache, alg, sol, reason::Symbol)

Perform column-pivoted sparse QR (`SparseColumnPivotedQRFactorization`) fallback
after a sparse LU (`KLUFactorization`, `SupernodalLUFactorization`,
`SparspakFactorization`) solve failed or produced non-finite output.

Sparse LU does not modify `cache.A` in place — UMFPACK and KLU wrap a
`SparseMatrixCSC` over the existing `colptr`/`rowval`/`nzval` arrays and store
the factorization on the factorization object — so there is no `A_backup`
restoration step like in the dense path's `_do_qr_fallback`.

Unlike the dense path, we do not recurse through `solve!(cache, QRFactorization(...))`
to compute the QR. We compute the rank-revealing column-pivoted sparse QR
directly via `sparse_colpivqr_factorize(cache.A)` (implemented in
`src/sparsearrays.jl` over SparseColumnPivotedQR.jl) and stash it in the dedicated
`:SparseColumnPivotedQRFactorization` slot ourselves with `setfield!`. That slot
is pre-initialized to a `SparseColumnPivotedQRFactorization` of the matching element type for the
`SparseMatrixCSC{<:Union{Float64, ComplexF64}, <:Integer}` cases that the sparse
LU defaults select, so the assignment is type-stable.
"""
function _do_sparse_qr_fallback(cache::LinearCache, alg, sol, reason::Symbol)
    if reason === :residual_check
        @SciMLMessage(
            "Sparse LU solve residual check failed, falling back to column-pivoted sparse QR. `A` is potentially ill-conditioned.",
            cache.verbose, :default_lu_fallback
        )
    else
        @SciMLMessage(
            "Sparse LU factorization failed, falling back to column-pivoted sparse QR. `A` is potentially rank-deficient or numerically singular.",
            cache.verbose, :default_lu_fallback
        )
    end
    # Mirror the KLU/UMFPACK structure split on the QR fallback: less-structured
    # (KLU-style) matrices fall back to the pure-Julia column-pivoted sparse QR;
    # more-structured matrices (which would have used UMFPACK) fall back to
    # SuiteSparse SPQR. Without GPL libraries SPQR is unavailable, so always use
    # the pure-Julia solver there.
    if Base.USE_GPL_LIBS && !use_klulike_sparse_structure(cache.A, cache.b)
        qr_fact = qr(convert(AbstractMatrix, cache.A))
        y = _ldiv!(cache.u, qr_fact, cache.b)
        setfield!(cache.cacheval, :QRFactorizationPivoted, qr_fact)
    else
        qr_fact = sparse_colpivqr_factorize(cache.A)
        y = _ldiv!(cache.u, qr_fact, cache.b)
        setfield!(cache.cacheval, :SparseColumnPivotedQRFactorization, qr_fact)
    end
    cache.cacheval.fell_back_to_qr = true
    cache.isfresh = false
    return SciMLBase.build_linear_solution(
        alg, y, nothing, nothing; retcode = ReturnCode.Success, iters = 0, stats = nothing
    )
end

"""
    _reuse_sparse_qr_fallback(cache::LinearCache, alg)

Reuse the cached sparse QR factorization stored by `_do_sparse_qr_fallback`.
Called when `fell_back_to_qr` is `true` and the matrix has not changed since the
fallback — we reuse the QR instead of retrying the (failed) LU. The structure
heuristic is re-derived (unchanged `A`) to read the slot the fallback wrote:
`:SparseColumnPivotedQRFactorization` (pure-Julia) or, for more-structured GPL
cases, `:QRFactorizationPivoted` (SPQR).
"""
function _reuse_sparse_qr_fallback(cache::LinearCache, alg)
    qr_fact = if Base.USE_GPL_LIBS && !use_klulike_sparse_structure(cache.A, cache.b)
        getfield(cache.cacheval, :QRFactorizationPivoted)
    else
        getfield(cache.cacheval, :SparseColumnPivotedQRFactorization)
    end
    y = _ldiv!(cache.u, qr_fact, cache.b)
    return SciMLBase.build_linear_solution(
        alg, y, nothing, nothing; retcode = ReturnCode.Success, iters = 0, stats = nothing
    )
end

"""
    _default_sparse_lu_solve_with_fallback(cache::LinearCache, alg::DefaultLinearSolver, sol)

Post-process a sparse-LU solve result: if the inner solver reported failure
(`ReturnCode.Failure` from Sparspak, `ReturnCode.Infeasible` from KLU/UMFPACK)
or produced non-finite values, fall back to SPQR. Otherwise return the original
solution.

This is the sparse-LU analogue of `_default_lu_solve_with_fallback`. The sparse
LU algorithms in LinearSolve return `ReturnCode.Infeasible` on `UMFPACK_OK`/
`KLU_OK` mismatch, whereas the dense LU path returns `ReturnCode.Failure`; we
accept both here so the wiring works uniformly.
"""
function _default_sparse_lu_solve_with_fallback(
        cache::LinearCache, alg::DefaultLinearSolver, sol
    )
    if alg.safetyfallback
        if sol.retcode === ReturnCode.Failure || sol.retcode === ReturnCode.Infeasible
            return _do_sparse_qr_fallback(cache, alg, sol, :lu_failure)
        end
        if sol.retcode === ReturnCode.Success && any(!isfinite, sol.u)
            @SciMLMessage(
                "Sparse LU solve produced non-finite values (NaN/Inf), falling back to SPQR. Matrix is likely near-singular.",
                cache.verbose, :default_lu_fallback
            )
            return _do_sparse_qr_fallback(cache, alg, sol, :lu_failure)
        end
        if sol.retcode === ReturnCode.APosterioriSafetyFailure
            return _do_sparse_qr_fallback(cache, alg, sol, :residual_check)
        end
    end
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, nothing;
        retcode = sol.retcode, iters = sol.iters, stats = nothing
    )
end

"""
    _default_lu_solve_with_fallback(cache::LinearCache, alg::DefaultLinearSolver, sol)

Post-process an LU solve result: if LU explicitly failed, the solution contains NaN/Inf,
or the residual check returned `APosterioriSafetyFailure`, fall back to column-pivoted QR.
Otherwise return the LU solution directly.

The NaN/Inf check catches floating-point-near-singular matrices where LU "succeeds"
(no exact zero pivot) but produces non-finite solution components from dividing by
near-zero pivots. This is O(n) and has zero false positives.
"""
function _default_lu_solve_with_fallback(
        cache::LinearCache, alg::DefaultLinearSolver, sol
    )
    if alg.safetyfallback
        if sol.retcode === ReturnCode.Failure
            return _do_qr_fallback(cache, alg, sol, :lu_failure)
        end
        if sol.retcode === ReturnCode.Success && any(!isfinite, sol.u)
            @SciMLMessage(
                "LU solve produced non-finite values (NaN/Inf), falling back to QR. Matrix is likely near-singular.",
                cache.verbose, :default_lu_fallback
            )
            return _do_qr_fallback(cache, alg, sol, :lu_failure)
        end
        if sol.retcode === ReturnCode.APosterioriSafetyFailure
            return _do_qr_fallback(cache, alg, sol, :residual_check)
        end
    end
    # Use cache directly for type-stable inference (see _do_qr_fallback).
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, nothing;
        retcode = sol.retcode, iters = sol.iters, stats = nothing
    )
end

"""
    _default_qr_solve_with_fallback(cache::LinearCache, alg::DefaultLinearSolver, sol)

Post-process an unpivoted-QR solve result: if the factorization failed, the
solution contains NaN/Inf, or the `R` diagonal says `A` is rank-deficient, redo
the solve with column-pivoted QR. Otherwise return the QR solution directly.

Unpivoted QR is the default for non-square (and for ill-conditioned square)
dense problems because it is up to ~3x cheaper than `geqp3`, but it cannot solve
a rank-deficient least-squares problem: the triangular solve divides by a zero
(or negligible) diagonal entry and returns garbage — all-zeros with
`ReturnCode.Failure` for an exact zero, an overflowing solution with
`ReturnCode.Success` for a nearly-zero one (issue #531). Column-pivoted QR
truncates the rank the same way LAPACK's `xGELSY` does, so the fallback
reproduces `A \\ b`.

The fallback only triggers for factorizations the check applies to, so sparse
(SPQR) and GPU defaults keep their current behavior.
"""
function _default_qr_solve_with_fallback(
        cache::LinearCache, alg::DefaultLinearSolver, sol
    )
    # `DenseMatrix` (not just "not GPU") because `fell_back_to_qr` reuse on the
    # next solve routes dense and sparse to different helpers. All of these are
    # decided by the cache's type, so the branch folds away at compile time.
    if alg.safetyfallback && cache.cacheval isa DefaultLinearSolverInit &&
            cache.A isa DenseMatrix && _qr_fallback_pivot(cache.A) isa ColumnNorm
        if sol.retcode === ReturnCode.Failure
            return _do_qr_fallback(cache, alg, sol, :qr_rank_deficient)
        end
        if sol.retcode === ReturnCode.Success &&
                (
                any(!isfinite, sol.u) ||
                    _qr_rank_deficient(getfield(cache.cacheval, :QRFactorization))
            )
            return _do_qr_fallback(cache, alg, sol, :qr_rank_deficient)
        end
    end
    # Use cache directly for type-stable inference (see _do_qr_fallback).
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, nothing;
        retcode = sol.retcode, iters = sol.iters, stats = nothing
    )
end

"""
    _algchoice_to_alg_with_safety(alg::Symbol)

Like `algchoice_to_alg`, but generates an expression that passes
`residualsafety = alg.residualsafety` at runtime. Used in the `@generated solve!`
so that the inner LU algorithm does its own residual check when the default solver
has `residualsafety=true`.
"""
function _algchoice_to_alg_with_safety(alg::Symbol)
    return if alg === :LUFactorization
        :(LUFactorization(residualsafety = alg.residualsafety))
    elseif alg === :GenericLUFactorization
        :(GenericLUFactorization(residualsafety = alg.residualsafety))
    elseif alg === :MKLLUFactorization
        :(MKLLUFactorization(residualsafety = alg.residualsafety))
    elseif alg === :AppleAccelerateLUFactorization
        :(AppleAccelerateLUFactorization(residualsafety = alg.residualsafety))
    elseif alg === :RFLUFactorization
        :(RFLUFactorization(throwerror = false, residualsafety = alg.residualsafety))
    elseif alg === :BLISLUFactorization
        :(BLISLUFactorization(throwerror = false, residualsafety = alg.residualsafety))
    elseif alg === :CudaOffloadLUFactorization
        :(CudaOffloadLUFactorization(throwerror = false, residualsafety = alg.residualsafety))
    elseif alg === :MetalLUFactorization
        :(MetalLUFactorization(throwerror = false, residualsafety = alg.residualsafety))
    else
        error("Algorithm $alg does not support residualsafety")
    end
end

# Generated body has the shape
#
#     if alg.alg === DefaultAlgorithmChoice.LUFactorization
#         SciMLBase.solve!(cache, LUFactorization(), args...; kwargs...)
#     elseif ...
#     end
#
# with one branch per DefaultAlgorithmChoice, so each branch calls solve! on a
# concrete algorithm type instead of going through a Symbol-to-algorithm lookup.
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
            # Pass residualsafety = alg.residualsafety so the inner algorithm does
            # its own residual check and returns APosterioriSafetyFailure if needed.
            inner_alg_expr = _algchoice_to_alg_with_safety(alg)
            newex = quote
                sol = SciMLBase.solve!(cache, $inner_alg_expr)
                _default_lu_solve_with_fallback(cache, alg, sol)
            end
        elseif alg == Symbol(DefaultAlgorithmChoice.RFLUFactorization)
            inner_alg_expr = _algchoice_to_alg_with_safety(alg)
            newex = quote
                if !userecursivefactorization(nothing)
                    error("Default algorithm calling solve on RecursiveFactorization without the package being loaded. This shouldn't happen.")
                end
                sol = SciMLBase.solve!(cache, $inner_alg_expr)
                _default_lu_solve_with_fallback(cache, alg, sol)
            end
        elseif alg == Symbol(DefaultAlgorithmChoice.BLISLUFactorization)
            inner_alg_expr = _algchoice_to_alg_with_safety(alg)
            newex = quote
                if !useblis(nothing)
                    error("Default algorithm calling solve on BLISLUFactorization without the extension being loaded. This shouldn't happen.")
                end
                sol = SciMLBase.solve!(cache, $inner_alg_expr)
                _default_lu_solve_with_fallback(cache, alg, sol)
            end
        elseif alg == Symbol(DefaultAlgorithmChoice.CudaOffloadLUFactorization)
            inner_alg_expr = _algchoice_to_alg_with_safety(alg)
            newex = quote
                if !usecuda(nothing)
                    error("Default algorithm calling solve on CudaOffloadLUFactorization without CUDA.jl being loaded. This shouldn't happen.")
                end
                sol = SciMLBase.solve!(cache, $inner_alg_expr)
                _default_lu_solve_with_fallback(cache, alg, sol)
            end
        elseif alg == Symbol(DefaultAlgorithmChoice.MetalLUFactorization)
            inner_alg_expr = _algchoice_to_alg_with_safety(alg)
            newex = quote
                if !usemetal(nothing)
                    error("Default algorithm calling solve on MetalLUFactorization without Metal.jl being loaded. This shouldn't happen.")
                end
                sol = SciMLBase.solve!(cache, $inner_alg_expr)
                _default_lu_solve_with_fallback(cache, alg, sol)
            end
        elseif alg == Symbol(DefaultAlgorithmChoice.QRFactorization)
            # Unpivoted QR (dense non-square, or ill-conditioned square): on a
            # rank-deficient `A` it cannot produce the least-squares solution, so
            # redo the solve with column-pivoted QR.
            newex = quote
                sol = SciMLBase.solve!(cache, $(algchoice_to_alg(alg)))
                _default_qr_solve_with_fallback(cache, alg, sol)
            end
        else
            if alg in LinearSolve._SPARSE_ONLY_ALGORITHMS
                if alg in LinearSolve._SPARSE_LU_FALLBACK_ALGORITHMS
                    # Sparse LU (KLU/UMFPACK/Sparspak): on failure, fall back to
                    # SPQR. Mirrors the dense LU → QR fallback path.
                    #
                    # When the persistent-nonstructural-zero reduction is active, the
                    # reduced matrix is the operand for the WHOLE sub-solve: we swap
                    # `cache.A` to it (raw `setfield!`, so no `isfresh`/`A_backup`
                    # side effects) around both the LU attempt and the QR fallback,
                    # so the fallback factorizes the same reduced matrix, then restore.
                    newex = quote
                        if !(cache.A isa Array)
                            _red = cache.cacheval.sparse_reduction
                            _Aop = reduce_operand!(_red, cache.A)
                            if _Aop === cache.A
                                sol = SciMLBase.solve!(cache, $(algchoice_to_alg(alg)))
                                _default_sparse_lu_solve_with_fallback(cache, alg, sol)
                            else
                                # Swap in the reduced operand for the whole sub-solve
                                # (LU attempt + QR fallback both read `cache.A`), then
                                # restore. The sparse solvers report status by return
                                # code rather than throwing, so no `try`/`finally` is
                                # needed. Separate variable for the raw sub-solve so
                                # `_result` only ever holds the (uniform
                                # `DefaultLinearSolver`-tagged) post-fallback solution,
                                # keeping the return type concrete.
                                _origA = getfield(cache, :A)
                                setfield!(cache, :A, _Aop)
                                _rawsol = SciMLBase.solve!(cache, $(algchoice_to_alg(alg)))
                                _result = _default_sparse_lu_solve_with_fallback(
                                    cache, alg, _rawsol
                                )
                                setfield!(cache, :A, _origA)
                                _result
                            end
                        else
                            error(
                                "Sparse algorithm " * $(string(alg)) *
                                    " called on non-sparse matrix. This shouldn't happen."
                            )
                        end
                    end
                elseif alg == Symbol(DefaultAlgorithmChoice.SparseColumnPivotedQRFactorization)
                    # Sparse column-pivoted QR (non-square / least-squares). Like the
                    # sparse-LU branch, drop persistent nonstructural zeros when active
                    # by swapping in the reduced operand for the solve, then restore.
                    # (CHOLMOD is handled in the `else` below WITHOUT reduction: dropping
                    # zeros asymmetrically would break the Cholesky structure.)
                    newex = quote
                        if !(cache.A isa Array)
                            _red = cache.cacheval.sparse_reduction
                            _Aop = reduce_operand!(_red, cache.A)
                            if _Aop === cache.A
                                sol = SciMLBase.solve!(cache, $(algchoice_to_alg(alg)))
                                SciMLBase.build_linear_solution(
                                    alg, cache.u, nothing, nothing;
                                    retcode = sol.retcode, iters = sol.iters, stats = nothing
                                )
                            else
                                _origA = getfield(cache, :A)
                                setfield!(cache, :A, _Aop)
                                _rawsol = SciMLBase.solve!(cache, $(algchoice_to_alg(alg)))
                                _result = SciMLBase.build_linear_solution(
                                    alg, cache.u, nothing, nothing;
                                    retcode = _rawsol.retcode, iters = _rawsol.iters,
                                    stats = nothing
                                )
                                setfield!(cache, :A, _origA)
                                _result
                            end
                        else
                            error(
                                "Sparse algorithm " * $(string(alg)) *
                                    " called on non-sparse matrix. This shouldn't happen."
                            )
                        end
                    end
                else
                    newex = quote
                        if !(cache.A isa Array)
                            sol = SciMLBase.solve!(cache, $(algchoice_to_alg(alg)))
                            SciMLBase.build_linear_solution(
                                alg, cache.u, nothing, nothing;
                                retcode = sol.retcode,
                                iters = sol.iters, stats = nothing
                            )
                        else
                            error(
                                "Sparse algorithm " * $(string(alg)) *
                                    " called on non-sparse matrix. This shouldn't happen."
                            )
                        end
                    end
                end
            else
                newex = quote
                    sol = SciMLBase.solve!(cache, $(algchoice_to_alg(alg)))
                    SciMLBase.build_linear_solution(
                        alg, cache.u, nothing, nothing;
                        retcode = sol.retcode,
                        iters = sol.iters, stats = nothing
                    )
                end
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
    alg_dispatch = Expr(:if, ex.args...)
    return quote
        if cache.cacheval isa DefaultLinearSolverInit &&
                cache.cacheval.fell_back_to_qr && !cache.isfresh
            if cache.A isa DenseMatrix
                _reuse_qr_fallback(cache, alg)
            else
                _reuse_sparse_qr_fallback(cache, alg)
            end
        else
            $alg_dispatch
        end
    end
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
        newex = if alg == Symbol(DefaultAlgorithmChoice.RFLUFactorization)
            quote
                getproperty(cache.cacheval, $(Meta.quot(alg)))[1]' \ dy
            end
        elseif alg == Symbol(DefaultAlgorithmChoice.GenericLUFactorization)
            quote
                getproperty(cache.cacheval, $(Meta.quot(alg))).fact' \ dy
            end
        elseif alg == Symbol(DefaultAlgorithmChoice.MKLLUFactorization)
            quote
                A = getproperty(cache.cacheval, $(Meta.quot(alg)))[1]
                getrs!('T', A.factors, A.ipiv, dy)
            end
        elseif alg == Symbol(DefaultAlgorithmChoice.AppleAccelerateLUFactorization)
            quote
                A = getproperty(cache.cacheval, $(Meta.quot(alg)))
                aa_getrs!('T', A.factors, A.ipiv, dy, A.info)
            end
        elseif alg in Symbol.(
                (
                    DefaultAlgorithmChoice.LUFactorization,
                    DefaultAlgorithmChoice.QRFactorization,
                    DefaultAlgorithmChoice.KLUFactorization,
                    DefaultAlgorithmChoice.SupernodalLUFactorization,
                    DefaultAlgorithmChoice.LDLtFactorization,
                    DefaultAlgorithmChoice.SparspakFactorization,
                    DefaultAlgorithmChoice.BunchKaufmanFactorization,
                    DefaultAlgorithmChoice.CHOLMODFactorization,
                    DefaultAlgorithmChoice.SVDFactorization,
                    DefaultAlgorithmChoice.CholeskyFactorization,
                    DefaultAlgorithmChoice.NormalCholeskyFactorization,
                    DefaultAlgorithmChoice.QRFactorizationPivoted,
                    DefaultAlgorithmChoice.SparseColumnPivotedQRFactorization,
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
                # `adjoint` rather than `transpose` to match the other branches, which
                # differ for a complex eltype, and `.u` because the caller wants the
                # solution vector the sibling branches return.
                invprob = LinearSolve.LinearProblem(adjoint(cache.A), dy)
                solve(
                    invprob, cache.alg;
                    abstol = cache.abstol,
                    reltol = cache.reltol,
                    verbose = cache.verbose
                ).u
            end
        else
            # Interpolate the algorithm name at generator time: inside `quote`, the
            # `$(alg)` of a string literal is left as a reference to a runtime binding
            # `alg`, which does not exist in the generated method.
            msg = "Default linear solver with algorithm $(alg) is currently not supported by Enzyme rules on LinearSolve.jl. Please open an issue on LinearSolve.jl detailing which algorithm is missing the adjoint handling"
            quote
                error($msg)
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
