using SciMLTesting, LinearSolve, Test
using SparseArrays  # materializes the KLU submodule via LinearSolveSparseArraysExt

# Extension submodules ExplicitImports cannot analyze; allow them to be unanalyzable.
klu_mod = try
    Base.get_extension(LinearSolve, :LinearSolveSparseArraysExt).KLU
catch
    nothing
end
unanalyzable_mods = (
    LinearSolve.OperatorCondition, LinearSolve.DefaultAlgorithmChoice,
    LinearSolve.NonstructuralZeros,
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
docs_src = normpath(joinpath(@__DIR__, "..", "..", "docs", "src"))
scimlbase_reexports = Tuple(names(LinearSolve.SciMLBase; all = false, imported = false))

run_qa(
    LinearSolve;
    explicit_imports = true,
    api_docs_kwargs = (; rendered = true, docs_src, rendered_ignore = scimlbase_reexports),
    # Recursive ambiguities are tracked separately; placeholder until resolved.
    aqua_broken = (:ambiguities,),
    aqua_kwargs = (;
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
            ignore = (sciml_logging_macro_imports..., extension_imports...),
        ),
        # Names imported from a re-exporting module rather than their defining owner:
        #   @blasfunc/chkstride1 (LinearAlgebra.BLAS, via LinearAlgebra.LAPACK),
        #   AbstractSciMLOperator (SciMLOperators, via SciMLBase),
        #   ArrayInterface/UMFPACK_OK (re-exported), inv (Base, via LinearAlgebra).
        all_explicit_imports_via_owners = (;
            ignore = (
                Symbol("@blasfunc"), :AbstractSciMLOperator, :ArrayInterface,
                :UMFPACK_OK, :chkstride1, :inv,
            ),
        ),
        # Non-public names explicitly imported from stdlib / other packages
        # (LinearAlgebra(.BLAS/.LAPACK), SparseArrays, SciMLBase, SciMLOperators,
        # ArrayInterface, StaticArraysCore, Base) and needed by the solver bindings.
        all_explicit_imports_are_public = (;
            ignore = (
                Symbol("@blasfunc"), :AbstractSciMLOperator, :AbstractSparseMatrixCSC,
                :ArrayInterface, :BLASELTYPES, :BlasInt, :StaticArray, :UMFPACK_OK,
                :build_eigenvalue_solution, :chkargsok, :chkfinite, :chkstride1,
                :getcolptr, :inv, :pattern_changed, :require_one_based_indexing,
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
