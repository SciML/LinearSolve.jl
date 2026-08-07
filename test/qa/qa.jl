using SciMLTesting, LinearSolve, Test
using SparseArrays  # materializes the KLU submodule via LinearSolveSparseArraysExt

# ExplicitImports only analyzes an extension module once it exists, and an extension
# module only exists once every one of its triggers has been loaded. Loading the
# cheaply-resolvable weakdeps here is what puts LinearSolve's extensions under QA.
using AlgebraicMultigrid, ArnoldiMethod, Arpack, BandedMatrices, BlockDiagonals
using ChainRulesCore, CliqueTrees, EnzymeCore, FastAlmostBandedMatrices
using FastLapackInterface, ForwardDiff, IterativeSolvers, JacobiDavidson
using KernelAbstractions, KrylovKit, Mooncake, PureUMFPACK, RecursiveFactorization
using Sparspak, SpecializingFactorizations, TriangularSolve
using LAPACK_jll, blis_jll

# The GPU backends are loaded for their extension modules only. ExplicitImports needs
# the module to exist, not a device: these all precompile and import on a CPU-only
# host (Metal logs "only supported on Apple Silicon" at init and still loads), and the
# CUDA/ROCm toolkit artifacts stay lazy without a driver.
using AMDGPU, CUDSS, Metal, cuSOLVER

# Extensions deliberately left unscanned:
#   external libs - LinearSolveElementalExt, LinearSolveGinkgoExt, LinearSolveHSLExt,
#                   LinearSolveHYPREExt, LinearSolveMUMPSExt, LinearSolvePardisoExt,
#                   LinearSolvePETScExt, LinearSolvePETScMPIExt,
#                   LinearSolvePartitionedSolversExt, LinearSolveSTRUMPACKExt,
#                   LinearSolveSuperLUDISTExt. These need real system installs (MPI,
#                   vendor libraries), not just a package that precompiles.
#   julia >= 1.12 - LinearSolveParUExt (ParU_jll needs a newer SuiteSparse_jll than
#                   the LTS stdlib ships, and QA also runs on lts)
#   unresolvable  - LinearSolveCUSOLVERRFExt. CUSOLVERRF 0.2.6 pins CUDA to 5.7-5.11
#                   while CUDSS 0.7 requires CUDACore 6, so the root [compat] entries
#                   for the two cannot both be satisfied in one environment. CUDSS is
#                   loaded here; adding CUSOLVERRF back needs that conflict resolved
#                   upstream first.
loaded_extensions = (
    :LinearSolveAMDGPUExt, :LinearSolveAlgebraicMultigridExt,
    :LinearSolveArnoldiMethodExt, :LinearSolveArpackExt, :LinearSolveBLISExt,
    :LinearSolveBandedMatricesExt, :LinearSolveBlockDiagonalsExt,
    :LinearSolveCUDAExt, :LinearSolveCUDSSExt, :LinearSolveChainRulesCoreExt,
    :LinearSolveCliqueTreesExt, :LinearSolveEnzymeExt,
    :LinearSolveFastAlmostBandedMatricesExt, :LinearSolveFastLapackInterfaceExt,
    :LinearSolveForwardDiffExt, :LinearSolveIterativeSolversExt,
    :LinearSolveJacobiDavidsonExt, :LinearSolveKernelAbstractionsExt,
    :LinearSolveKrylovKitExt, :LinearSolveMetalExt, :LinearSolveMooncakeExt,
    :LinearSolvePureUMFPACKExt,
    :LinearSolveRecursiveFactorizationExt, :LinearSolveSparseArraysExt,
    :LinearSolveSparspakExt, :LinearSolveSpecializingFactorizationsExt,
)

# ExplicitImports silently skips an extension that fails to load -- `get_extension`
# returns `nothing` and the checks still report a clean pass -- so assert the modules
# exist rather than trusting a green `run_qa`.
@testset "Extensions loaded" begin
    for ext in loaded_extensions
        @test Base.get_extension(LinearSolve, ext) !== nothing
    end
end

# Extension submodules ExplicitImports cannot analyze; allow them to be unanalyzable.
klu_mod = try
    Base.get_extension(LinearSolve, :LinearSolveSparseArraysExt).KLU
catch
    nothing
end
unanalyzable_mods = (
    LinearSolve.OperatorCondition, LinearSolve.DefaultAlgorithmChoice,
    LinearSolve.NonstructuralZeros, LinearSolve.WarmStart,
)
if klu_mod !== nothing
    unanalyzable_mods = (unanalyzable_mods..., klu_mod)
end

# SciMLLogging names pulled in by the @verbosity_specifier macro expansion, plus
# @set! reached by extensions via LinearSolve.@set! — both look stale to EI because
# their only uses are through macro-generated / downstream-extension code.
sciml_logging_macro_imports = (
    :AbstractVerbositySpecifier, :AbstractVerbosityPreset,
    :None, :Minimal, :Standard, :Detailed, :All,
)
extension_imports = (Symbol("@set!"),)

# Algorithm types the BandedMatrices / FastAlmostBandedMatrices extensions reference
# only from inside `for alg in (...) @eval ... end` loops. They must be imported for
# the generated `init_cacheval` methods to resolve, but EI analyzes source text rather
# than expanded code, so it sees the imports as unused.
eval_generated_imports = (
    :AppleAccelerateLUFactorization, :BunchKaufmanFactorization,
    :CHOLMODFactorization, :CholeskyFactorization, :DiagonalFactorization,
    :GenericLUFactorization, :KLUFactorization, :LDLtFactorization,
    :LUFactorization, :MKLLUFactorization, :NormalCholeskyFactorization,
    :RFLUFactorization, :SVDFactorization, :SparspakFactorization,
    :UMFPACKFactorization,
)

# Non-public names the extensions import from the backend package they wrap, where the
# backend exposes no public spelling for what the extension has to do:
#   Mooncake      - the reverse-rule interface (@from_chainrules / @is_primitive and the
#                   CoDual / fdata / rdata tangent types)
#   EnzymeCore    - EnzymeRules, the submodule the rules are defined in
#   ForwardDiff   - Dual / Partials, needed to dispatch on dual-number problems
#   ArnoldiMethod - LM/LR/SR/LI/SI, its eigenvalue-target singletons
#   Sparspak      - sparspaklu/sparspaklu!, the only entry points to its factorization
backend_internal_imports = (
    Symbol("@from_chainrules"), Symbol("@is_primitive"), :CoDual, :MinimalCtx,
    :NoRData, :ReverseMode, :fdata, :primal, :rdata, :zero_fcodual,
    :EnzymeRules, :Dual, :Partials, :LI, :LM, :LR, :SI, :SR,
    :sparspaklu, Symbol("sparspaklu!"),
)

# LinearSolve's own non-public names, imported from LinearSolve by its extensions.
# Same class as the qualified accesses tracked in
# https://github.com/SciML/LinearSolve.jl/issues/1058; promoting them is a separate
# public-API decision.
linearsolve_internal_imports = (
    Symbol("@get_cacheval"), :AbstractFactorization, :AbstractKrylovSubspaceMethod,
    :BLISLUFactorization, :DEFAULT_PRECS, :DefaultAlgorithmChoice, :DefaultLinearSolver,
    :LinearCache, :SciMLLinearSolveAlgorithm, :__init, :blas_info_msg, :default_alias_A,
    :defaultalg, :do_factorization, :get_blas_operation_info, :init_cacheval,
)

docs_src = normpath(joinpath(@__DIR__, "..", "..", "docs", "src"))
scimlbase_reexports = Tuple(names(LinearSolve.SciMLBase; all = false, imported = false))

run_qa(
    LinearSolve;
    explicit_imports = true,
    reexports_allow = scimlbase_reexports,
    # `scimlbase_reexports` covers both checks: `names(SciMLBase)` includes the
    # module's own name, so re-exporting it puts `:SciMLBase` in LinearSolve's public
    # API, where the docstring check counts it as undocumented. It is SciMLBase's name
    # to document, not LinearSolve's, hence the same ignore list as the rendered check.
    api_docs_kwargs = (;
        rendered = true, docs_src,
        ignore = scimlbase_reexports, rendered_ignore = scimlbase_reexports,
    ),
    aqua_kwargs = (;
        # `MKL_jll` is not stale: `src/LinearSolve.jl` and `src/mkl.jl` load it
        # (`using MKL_jll: MKL_jll` / `libmkl_rt`) behind a `@static if` gated on a
        # `Preferences.@load_preference`, so MKL can be opted out of at build time.
        # Aqua analyzes the source without resolving that branch and cannot see the
        # use, so the dependency has to be ignored here rather than dropped.
        deps_compat = (; ignore = [:MKL_jll]),
        stale_deps = (; ignore = [:MKL_jll]),
        piracies = (; treat_as_own = [LinearProblem, EigenvalueProblem]),
    ),
    ei_kwargs = (;
        no_implicit_imports = (;
            skip = (Base, Core), allow_unanalyzable = unanalyzable_mods,
        ),
        no_stale_explicit_imports = (;
            allow_unanalyzable = unanalyzable_mods,
            ignore = (
                sciml_logging_macro_imports..., extension_imports...,
                eval_generated_imports...,
            ),
        ),
        # Names imported from a re-exporting module rather than their defining owner:
        #   @blasfunc/chkstride1 (LinearAlgebra.BLAS, via LinearAlgebra.LAPACK),
        #   AbstractSciMLOperator (SciMLOperators, via SciMLBase),
        #   ArrayInterface/UMFPACK_OK (re-exported), inv (Base, via LinearAlgebra),
        #   init/solve/solve! (CommonSolve, via SciMLBase) — CommonSolve is not a direct
        #   LinearSolve dependency and only the extensions use these names, so making it
        #   one would just make it a stale dep of the main module.
        all_explicit_imports_via_owners = (;
            ignore = (
                Symbol("@blasfunc"), :AbstractSciMLOperator, :ArrayInterface,
                :UMFPACK_OK, :chkstride1, :init, :inv, :solve, Symbol("solve!"),
            ),
        ),
        # CUDACore and cuSPARSE are reached through cuSOLVER / CUDSS, which are the
        # extensions' declared triggers. Adding them as extra triggers would be the
        # usual fix, but it would also stop the extension loading for anyone who does
        # a plain `using cuSOLVER`, so the aliases stay.
        all_qualified_accesses_via_owners = (; ignore = (:CUDACore, :cuSPARSE)),
        # Non-public names explicitly imported from stdlib / other packages
        # (LinearAlgebra(.BLAS/.LAPACK), SparseArrays, SciMLBase, SciMLOperators,
        # ArrayInterface, StaticArraysCore, Base) and needed by the solver bindings.
        all_explicit_imports_are_public = (;
            ignore = (
                Symbol("@blasfunc"), :AbstractSciMLOperator, :AbstractSparseMatrixCSC,
                :ArrayInterface, :BLASELTYPES, :BlasInt, :StaticArray, :UMFPACK_OK,
                :build_eigenvalue_solution, :chkargsok, :chkfinite, :chkstride1,
                :getcolptr, :inv, :pattern_changed, :require_one_based_indexing,
                backend_internal_imports..., linearsolve_internal_imports...,
            ),
        ),
    ),
    # ~90 qualified accesses of non-public names (LinearSolve's own internals reached
    # via LinearSolve.x from extensions, plus stdlib/SciMLBase/LinearAlgebra internals).
    # Making them public is a large cross-package effort tracked in
    # https://github.com/SciML/LinearSolve.jl/issues/1058
    ei_broken = (:all_qualified_accesses_are_public,),
)

if klu_mod !== nothing
    run_api_docs(klu_mod; rendered = true, docs_src)
end
