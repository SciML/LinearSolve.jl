module LinearSolve
if isdefined(Base, :Experimental) &&
   isdefined(Base.Experimental, Symbol("@max_methods"))
    @eval Base.Experimental.@max_methods 1
end

import PrecompileTools

PrecompileTools.@recompile_invalidations begin
    using ArrayInterface
    using RecursiveFactorization
    using Base: cache_dependencies, Bool
    using LinearAlgebra
    using SparseArrays
    using SparseArrays: AbstractSparseMatrixCSC, nonzeros, rowvals, getcolptr
    using SciMLBase: AbstractLinearAlgorithm
    using SciMLOperators
    using SciMLOperators: AbstractSciMLOperator, IdentityOperator
    using Setfield
    using UnPack
    using KLU
    using Sparspak
    using FastLapackInterface
    using DocStringExtensions
    using EnumX
    using Requires
    import InteractiveUtils

    using LinearAlgebra: BlasInt, LU
    using LinearAlgebra.LAPACK: require_one_based_indexing,
        chkfinite, chkstride1,
        @blasfunc, chkargsok

    import GPUArraysCore
    import Preferences
    import ConcreteStructs: @concrete

    # wrap
    import Krylov
    using SciMLBase
    import Preferences
end

if Preferences.@load_preference("TriggerMKLLBT", true)
   using MKL
end

if Preferences.@load_preference("LoadMKL_JLL", true)
    using MKL_jll
    const usemkl = MKL_jll.is_available()
else
    const usemkl = false
end

using Reexport
@reexport using SciMLBase
using SciMLBase: _unwrap_val

abstract type SciMLLinearSolveAlgorithm <: SciMLBase.AbstractLinearAlgorithm end
abstract type AbstractFactorization <: SciMLLinearSolveAlgorithm end
abstract type AbstractKrylovSubspaceMethod <: SciMLLinearSolveAlgorithm end
abstract type AbstractSolveFunction <: SciMLLinearSolveAlgorithm end

# Traits

needs_concrete_A(alg::AbstractFactorization) = true
needs_concrete_A(alg::AbstractKrylovSubspaceMethod) = false
needs_concrete_A(alg::AbstractSolveFunction) = false

# Util
is_underdetermined(x) = false
is_underdetermined(A::AbstractMatrix) = size(A, 1) < size(A, 2)
is_underdetermined(A::AbstractSciMLOperator) = size(A, 1) < size(A, 2)

_isidentity_struct(A) = false
_isidentity_struct(λ::Number) = isone(λ)
_isidentity_struct(A::UniformScaling) = isone(A.λ)
_isidentity_struct(::SciMLOperators.IdentityOperator) = true

# Dispatch Friendly way to check if an extension is loaded
__is_extension_loaded(::Val) = false

function _fast_sym_givens! end

# Code

const INCLUDE_SPARSE = Preferences.@load_preference("include_sparse", Base.USE_GPL_LIBS)

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
    RFLUFactorization
    LDLtFactorization
    BunchKaufmanFactorization
    CHOLMODFactorization
    SVDFactorization
    CholeskyFactorization
    NormalCholeskyFactorization
    AppleAccelerateLUFactorization
    MKLLUFactorization
    QRFactorizationPivoted
end

struct DefaultLinearSolver <: SciMLLinearSolveAlgorithm
    alg::DefaultAlgorithmChoice.T
end

const BLASELTYPES = Union{Float32, Float64, ComplexF32, ComplexF64}

include("common.jl")
include("factorization.jl")
include("appleaccelerate.jl")
include("mkl.jl")
include("simplelu.jl")
include("simplegmres.jl")
include("iterative_wrappers.jl")
include("preconditioners.jl")
include("solve_function.jl")
include("default.jl")
include("init.jl")
include("extension_algs.jl")
include("deprecated.jl")

@generated function SciMLBase.solve!(cache::LinearCache, alg::AbstractFactorization;
    kwargs...)
    quote
        if cache.isfresh
            fact = do_factorization(alg, cache.A, cache.b, cache.u)
            cache.cacheval = fact
            cache.isfresh = false
        end
        y = _ldiv!(cache.u, @get_cacheval(cache, $(Meta.quot(defaultalg_symbol(alg)))),
            cache.b)

        #=
        retcode = if LinearAlgebra.issuccess(fact)
            SciMLBase.ReturnCode.Success
        else
            SciMLBase.ReturnCode.Failure
        end
        SciMLBase.build_linear_solution(alg, y, nothing, cache; retcode = retcode)
        =#
        SciMLBase.build_linear_solution(alg, y, nothing, cache)
    end
end

@static if INCLUDE_SPARSE
    include("factorization_sparse.jl")
end

# Solver Specific Traits
## Needs Square Matrix
"""
    needs_square_A(alg)

Returns `true` if the algorithm requires a square matrix.
"""
needs_square_A(::Nothing) = false  # Linear Solve automatically will use a correct alg!
needs_square_A(alg::SciMLLinearSolveAlgorithm) = true
for alg in (:QRFactorization, :FastQRFactorization, :NormalCholeskyFactorization,
    :NormalBunchKaufmanFactorization)
    @eval needs_square_A(::$(alg)) = false
end
for kralg in (Krylov.lsmr!, Krylov.craigmr!)
    @eval needs_square_A(::KrylovJL{$(typeof(kralg))}) = false
end
for alg in (:LUFactorization, :FastLUFactorization, :SVDFactorization,
    :GenericFactorization, :GenericLUFactorization, :SimpleLUFactorization,
    :RFLUFactorization, :UMFPACKFactorization, :KLUFactorization, :SparspakFactorization,
    :DiagonalFactorization, :CholeskyFactorization, :BunchKaufmanFactorization,
    :CHOLMODFactorization, :LDLtFactorization, :AppleAccelerateLUFactorization,
    :MKLLUFactorization, :MetalLUFactorization)
    @eval needs_square_A(::$(alg)) = true
end

const IS_OPENBLAS = Ref(true)
isopenblas() = IS_OPENBLAS[]

PrecompileTools.@compile_workload begin
    A = rand(4, 4)
    b = rand(4)
    prob = LinearProblem(A, b)
    sol = solve(prob)
    sol = solve(prob, LUFactorization())
    sol = solve(prob, RFLUFactorization())
    sol = solve(prob, KrylovJL_GMRES())
end

@static if INCLUDE_SPARSE
    PrecompileTools.@compile_workload begin
        A = sprand(4, 4, 0.3) + I
        b = rand(4)
        prob = LinearProblem(A, b)
        sol = solve(prob, KLUFactorization())
        sol = solve(prob, UMFPACKFactorization())
    end
end

@static if VERSION > v"1.9-"
    PrecompileTools.@compile_workload begin
        A = sprand(4, 4, 0.3) + I
        b = rand(4)
        prob = LinearProblem(A * A', b)
        sol = solve(prob) # in case sparspak is used as default
        sol = solve(prob, SparspakFactorization())
    end
end

export LUFactorization, SVDFactorization, QRFactorization, GenericFactorization,
    GenericLUFactorization, SimpleLUFactorization, RFLUFactorization,
    NormalCholeskyFactorization, NormalBunchKaufmanFactorization,
    UMFPACKFactorization, KLUFactorization, FastLUFactorization, FastQRFactorization,
    SparspakFactorization, DiagonalFactorization, CholeskyFactorization,
    BunchKaufmanFactorization, CHOLMODFactorization, LDLtFactorization

export LinearSolveFunction, DirectLdiv!

export KrylovJL, KrylovJL_CG, KrylovJL_MINRES, KrylovJL_GMRES,
    KrylovJL_BICGSTAB, KrylovJL_LSMR, KrylovJL_CRAIGMR,
    IterativeSolversJL, IterativeSolversJL_CG, IterativeSolversJL_GMRES,
    IterativeSolversJL_BICGSTAB, IterativeSolversJL_MINRES, IterativeSolversJL_IDRS,
    KrylovKitJL, KrylovKitJL_CG, KrylovKitJL_GMRES

export SimpleGMRES

export HYPREAlgorithm
export CudaOffloadFactorization
export MKLPardisoFactorize, MKLPardisoIterate
export PardisoJL
export MKLLUFactorization
export AppleAccelerateLUFactorization
export MetalLUFactorization

export OperatorAssumptions, OperatorCondition

end
