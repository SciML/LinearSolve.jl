using SciMLTesting, LinearSolve, Test
using SparseArrays

# ExplicitImports only analyzes an extension module once it exists, and an extension
# module only exists once every one of its triggers has been loaded. Loading the
# cheaply-resolvable weakdeps here is what puts LinearSolve's extensions under QA.
using AlgebraicMultigrid, ArnoldiMethod, Arpack, BandedMatrices, BlockDiagonals
using ConjugateGradients
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
    :LinearSolveConjugateGradientsExt,
    :LinearSolveCUDAExt, :LinearSolveCUDSSExt, :LinearSolveChainRulesCoreExt,
    :LinearSolveCliqueTreesExt, :LinearSolveEnzymeExt,
    :LinearSolveFastAlmostBandedMatricesExt, :LinearSolveFastLapackInterfaceExt,
    :LinearSolveForwardDiffExt, :LinearSolveIterativeSolversExt,
    :LinearSolveJacobiDavidsonExt, :LinearSolveKernelAbstractionsExt,
    :LinearSolveKrylovKitExt, :LinearSolveMetalExt, :LinearSolveMooncakeExt,
    :LinearSolvePureUMFPACKExt,
    :LinearSolveRecursiveFactorizationExt,
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

# Submodules ExplicitImports cannot analyze; allow them to be unanalyzable.
unanalyzable_mods = (
    LinearSolve.OperatorCondition, LinearSolve.DefaultAlgorithmChoice,
    LinearSolve.NonstructuralZeros, LinearSolve.WarmStart, LinearSolve.KLU,
)

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

# The names of UMFPACK's control-vector entries, plus the constructor for a
# control vector carrying SparseArrays' defaults. `UMFPACKFactorization`'s
# `control` keyword is defined in terms of these, and SparseArrays.UMFPACK
# marks none of them public.
umfpack_control_imports = (
    :get_umfpack_control,
    :JL_UMFPACK_PRL, :JL_UMFPACK_DENSE_ROW, :JL_UMFPACK_DENSE_COL,
    :JL_UMFPACK_BLOCK_SIZE, :JL_UMFPACK_ORDERING,
    :JL_UMFPACK_FIXQ, :JL_UMFPACK_AMD_DENSE, :JL_UMFPACK_AGGRESSIVE,
    :JL_UMFPACK_SINGLETONS, :JL_UMFPACK_ALLOC_INIT,
    :JL_UMFPACK_SYM_PIVOT_TOLERANCE, :JL_UMFPACK_SCALE,
    :JL_UMFPACK_FRONT_ALLOC_INIT, :JL_UMFPACK_DROPTOL, :JL_UMFPACK_IRSTEP,
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

# LinearSolve's own non-public names reached as `LinearSolve.x` (or
# `LinearSolve.SupernodalLU.x`) from its extension modules. Same class as
# `linearsolve_internal_imports` above: an extension is its own module root, so
# ExplicitImports counts these as external accesses. Promoting them is a separate
# public-API decision (https://github.com/SciML/LinearSolve.jl/issues/1058); this
# inventory keeps the strict check on so any new non-public access is caught.
linearsolve_internal_accesses = (
    Symbol("@get_cacheval"), :ALREADY_WARNED_CUDSS, :AbstractFactorization,
    :AbstractKrylovSubspaceMethod, :DefaultAlgorithmChoice, :DefaultLinearSolver,
    :DefaultLinearSolverInit, :GPUArraysCore, :LinearCache, :MinNormQR,
    :NONPERSISTENT_ZERO_FRACTION, :PERSISTENT_ZERO_FRACTION_THRESHOLD,
    :PrecompileTools, :SciMLLinearSolveAlgorithm, :SupernodalLU,
    :_SPARSE_LU_FALLBACK_ALGORITHMS, :_SPARSE_ONLY_ALGORITHMS,
    :__is_extension_loaded, :__nonstructural_zeros, :_adjoint_factorization_solve,
    :_adjoint_krylov_solve, :_adjoint_precs, :_adjoint_solve, :_can_reuse_cache_factorization,
    :_check_residual_safety,
    :_custom_adjoint_factorization_solve, :_custom_cache_factorization,
    :_custom_can_reuse_adjoint_factorization, Symbol("_direct_lu_factorize!"),
    Symbol("_direct_lu_solve!"), Symbol("_fast_sym_givens!"), :_init_cacheval,
    :_isidentity_struct, Symbol("_ldiv!"), :_select_eigenpairs, :_sym_givens,
    :cudss_loaded, :default_alias_A, :default_num_eigenpairs, :default_tol,
    :defaultalg, :defaultalg_adjoint_eval, :do_factorization, :error_no_cudss_lu,
    :handle_sparsematrixcsc_lu, :init_cacheval, :init_sparse_reduction, :is_cusparse,
    :is_cusparse_csc, :is_cusparse_csr, :is_underdetermined, :issparsematrix,
    :issparsematrixcsc, :make_SparseMatrixCSC, :makeempty_SparseMatrixCSC,
    :pattern_changed, Symbol("reduce_operand!"), :sparse_colpivqr_factorize,
    Symbol("update_tolerances_internal!"), :use_klulike_sparse_structure, :useblis,
    :usecuda, :usemetal, :userecursivefactorization,
    # LinearSolve.SupernodalLU
    :PANEL_BLAS_MIN_NP, :SupernodalLUFactor, :_costabs, Symbol("_panel_ldiv!"),
    Symbol("_panel_rdiv!"), Symbol("_panel_solve_unit_lower!"),
    Symbol("_panel_solve_upper!"), Symbol("_panel_unit_lower_trsm!"),
    Symbol("_panel_upper_trsm!"), Symbol("_unit_lower_solve!"),
    Symbol("_upper_solve!"), :nperturbed, :snlu, Symbol("snlu!"),
    Symbol("solve!"),
)

# Non-public names of stdlib / backend packages accessed with a qualified path,
# where the owner exposes no public spelling for what the solver bindings need.
# Grouped by owner; a name listed once is ignored for every owner (the check's
# `ignore` is name-based).
external_internal_accesses = (
    # Base / Base.Experimental
    Symbol("@_inline_meta"), :Experimental, :RefValue, :USE_BLAS64, :USE_GPL_LIBS,
    :return_types, :structdiff, :typename, Symbol("@max_methods"),
    # LinearAlgebra (+ .BLAS / .LAPACK)
    :AdjointFactorization, :BlasFloat, :BlasInt, :PivotingStrategy, :QRCompactWY,
    :QRIteration, :TransposeFactorization, :_check_lu_success, Symbol("_ipiv_rows!"),
    :checknonsingular, :generic_lufact!, :lupivottype, :lutype, :get_config,
    :get_num_threads, :set_num_threads, :chkfinite, :chklapackerror, :chktrans,
    Symbol("geqp3!"), Symbol("geqrt!"), Symbol("getrf!"),
    # SparseArrays (+ .SPQR)
    :AbstractSparseMatrixCSC, :CHOLMOD, :SPQR, :UMFPACK, :getcolptr, :QRSparse,
    # AMDGPU (+ .rocBLAS / .rocSOLVER)
    :rocBLAS, :rocSOLVER, Symbol("trsv!"), Symbol("geqrf!"), Symbol("getrs!"),
    Symbol("ormqr!"),
    # CUDSS / cuSOLVER
    :cuSPARSE, :CUDACore,
    # EnzymeCore (+ .EnzymeRules)
    :EnzymeRules, :augmented_primal, :forward, :inactive_type, :reverse,
    # ForwardDiff
    :Dual, :npartials, :partials, :valtype, :value,
    # IterativeSolvers
    :GMRESIterable, :IDRSIterable, :MINRESIterable, :Residual,
    Symbol("gmres_iterable!"), Symbol("idrs_iterable!"), Symbol("init!"),
    Symbol("init_residual!"), Symbol("minres_iterable!"),
    # Krylov / Mooncake / EnumX / PureKLU / MKL_jll / OpenBLAS_jll
    Symbol("warm_start!"), Symbol("increment_and_get_rdata!"), Symbol("rrule!!"),
    :symbol_map, :KLU_OK, :is_available,
    # RecursiveFactorization / TriangularSolve
    Symbol("lu!"), Symbol("🦋mul!"), Symbol("🦋workspace"), Symbol("ldiv!"),
    Symbol("rdiv!"),
)

# `@reexport using SciMLBase` puts every SciMLBase export in LinearSolve's public API,
# so the reexport audit needs them allowed; the list is computed so it tracks
# SciMLBase. The API-docs checks need no matching ignore: SciMLTesting follows each
# binding to SciMLBase's docstring, exempts the reexported module name, and only
# requires a local `@docs` entry for names LinearSolve owns.
scimlbase_reexports = Tuple(names(LinearSolve.SciMLBase; all = false, imported = false))

run_qa(
    LinearSolve;
    explicit_imports = true,
    reexports_allow = scimlbase_reexports,
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
                :ArrayInterface, :BLASELTYPES, :BlasInt,
                :StaticArray, :UMFPACK_OK,
                :build_eigenvalue_solution, :chkargsok, :chkfinite, :chkstride1,
                :getcolptr, :inv, :pattern_changed,
                :require_one_based_indexing,
                umfpack_control_imports...,
                backend_internal_imports..., linearsolve_internal_imports...,
            ),
        ),
        # Inventory of today's non-public qualified accesses (see the two constants
        # above). Replaces the blanket `ei_broken` marker from
        # https://github.com/SciML/LinearSolve.jl/issues/1058: the strict check now
        # runs, so a qualified access of any name not listed here fails QA.
        all_qualified_accesses_are_public = (;
            ignore = (
                linearsolve_internal_accesses..., external_internal_accesses...,
            ),
        ),
    ),
)

run_api_docs(LinearSolve.KLU)
