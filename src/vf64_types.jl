# VF64 specialized types for reducing type parameter counts
# when A::Matrix{Float64} and b::Vector{Float64} with DefaultLinearSolver.
#
# This file defines DefaultLinearSolverInitVF64 which hardcodes all 24 factorization
# cache types for Matrix{Float64}, reducing the type parameter count from 25 to 1
# (only A_backup type remains parameterized).
#
# Together with LinearCacheVF64 (defined in common.jl), this reduces the total
# LinearCache type string from ~1500 chars to ~100 chars in stack traces.

# Type alias for the default LinearVerbosity (Standard preset)
const _DefaultLinearVerbosity = typeof(LinearVerbosity())

# Compute concrete factorization types for Matrix{Float64} at module load time
# and define DefaultLinearSolverInitVF64 with hardcoded types.
let
    _A = [1.0 0.0; 0.0 1.0]
    _b = [1.0, 1.0]
    _u = [0.0, 0.0]
    _Pl = IdentityOperator(2)
    _Pr = IdentityOperator(2)
    _verbose = LinearVerbosity()
    _assump = OperatorAssumptions(true)
    _alg = DefaultLinearSolver(DefaultAlgorithmChoice.LUFactorization)

    _cacheval = _init_default_cacheval(
        _alg, _A, _b, _u, _Pl, _Pr, 2, sqrt(eps()), sqrt(eps()),
        _verbose, _assump, _A
    )

    _T = typeof(_cacheval)
    _tparams = _T.parameters

    @eval begin
        """
            DefaultLinearSolverInitVF64{TA}

        VF64-specialized variant of `DefaultLinearSolverInit` for the common case of
        `Matrix{Float64}` linear systems. All 24 factorization cache slot types are
        hardcoded to their concrete types for `Matrix{Float64}`, reducing the type
        parameter count from 25 to 1 (only `A_backup::TA` remains parameterized).

        Field names match `DefaultLinearSolverInit` exactly for transparent dispatch.
        """
        mutable struct DefaultLinearSolverInitVF64{TA}
            LUFactorization::$(_tparams[1])
            QRFactorization::$(_tparams[2])
            DiagonalFactorization::$(_tparams[3])
            var"DirectLdiv!"::$(_tparams[4])
            SparspakFactorization::$(_tparams[5])
            KLUFactorization::$(_tparams[6])
            UMFPACKFactorization::$(_tparams[7])
            KrylovJL_GMRES::$(_tparams[8])
            GenericLUFactorization::$(_tparams[9])
            RFLUFactorization::$(_tparams[10])
            LDLtFactorization::$(_tparams[11])
            BunchKaufmanFactorization::$(_tparams[12])
            CHOLMODFactorization::$(_tparams[13])
            SVDFactorization::$(_tparams[14])
            CholeskyFactorization::$(_tparams[15])
            NormalCholeskyFactorization::$(_tparams[16])
            AppleAccelerateLUFactorization::$(_tparams[17])
            MKLLUFactorization::$(_tparams[18])
            QRFactorizationPivoted::$(_tparams[19])
            KrylovJL_CRAIGMR::$(_tparams[20])
            KrylovJL_LSMR::$(_tparams[21])
            BLISLUFactorization::$(_tparams[22])
            CudaOffloadLUFactorization::$(_tparams[23])
            MetalLUFactorization::$(_tparams[24])
            A_backup::TA
        end
    end
end

"""
    DefaultLinearSolverInitType

Union type for dispatch compatibility between `DefaultLinearSolverInit` and
`DefaultLinearSolverInitVF64`.
"""
const DefaultLinearSolverInitType = Union{DefaultLinearSolverInit, DefaultLinearSolverInitVF64}

# Extend the trait for VF64 variant
_is_default_linear_solver_init(::DefaultLinearSolverInitVF64) = true

# __setfield! for DefaultLinearSolverInitVF64 - same generated logic as DefaultLinearSolverInit
@generated function __setfield!(cache::DefaultLinearSolverInitVF64, alg::DefaultLinearSolver, v)
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
        cache::DefaultLinearSolverInitVF64,
        alg::DefaultLinearSolver, v::LinearAlgebra.QRPivoted
    )
    return setfield!(cache, :QRFactorizationPivoted, v)
end

"""
    _convert_to_vf64_cacheval(cache::DefaultLinearSolverInit)

Convert a generic `DefaultLinearSolverInit` to `DefaultLinearSolverInitVF64`
by copying all field values. This is called during `LinearCacheVF64` construction.
"""
function _convert_to_vf64_cacheval(cache::DefaultLinearSolverInit)
    return DefaultLinearSolverInitVF64(
        cache.LUFactorization,
        cache.QRFactorization,
        cache.DiagonalFactorization,
        getfield(cache, Symbol("DirectLdiv!")),
        cache.SparspakFactorization,
        cache.KLUFactorization,
        cache.UMFPACKFactorization,
        cache.KrylovJL_GMRES,
        cache.GenericLUFactorization,
        cache.RFLUFactorization,
        cache.LDLtFactorization,
        cache.BunchKaufmanFactorization,
        cache.CHOLMODFactorization,
        cache.SVDFactorization,
        cache.CholeskyFactorization,
        cache.NormalCholeskyFactorization,
        cache.AppleAccelerateLUFactorization,
        cache.MKLLUFactorization,
        cache.QRFactorizationPivoted,
        cache.KrylovJL_CRAIGMR,
        cache.KrylovJL_LSMR,
        cache.BLISLUFactorization,
        cache.CudaOffloadLUFactorization,
        cache.MetalLUFactorization,
        cache.A_backup,
    )
end

"""
    _try_build_vf64_cache(A, b, u, p, alg, cacheval, isfresh, precsisfresh,
        Pl, Pr, abstol, reltol, maxiters, verbose, assumptions, sensealg)

Attempt to construct a `LinearCacheVF64` if all types match the VF64 pattern:
- `A::Matrix{Float64}`, `b::Vector{Float64}`, `u::Vector{Float64}`
- `alg::DefaultLinearSolver`
- `cacheval::DefaultLinearSolverInit`
- `Pl::IdentityOperator`, `Pr::IdentityOperator`
- `abstol::Float64`, `reltol::Float64`
- `assumptions::OperatorAssumptions{Bool}`

Returns `nothing` if the types don't match, allowing fallback to generic `LinearCache`.
"""
function _try_build_vf64_cache(
        A::Matrix{Float64}, b::Vector{Float64}, u::Vector{Float64},
        p, alg::DefaultLinearSolver, cacheval::DefaultLinearSolverInit,
        isfresh::Bool, precsisfresh::Bool,
        Pl::IdentityOperator, Pr::IdentityOperator,
        abstol::Float64, reltol::Float64, maxiters::Int,
        verbose, assumptions::OperatorAssumptions{Bool},
        sensealg
    )
    vf64_cacheval = _convert_to_vf64_cacheval(cacheval)
    return LinearCacheVF64{typeof(p), typeof(vf64_cacheval), typeof(verbose), typeof(sensealg)}(
        A, b, u, p, alg, vf64_cacheval, isfresh, precsisfresh,
        Pl, Pr, abstol, reltol, maxiters, verbose, assumptions, sensealg
    )
end
