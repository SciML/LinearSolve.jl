using SciMLTesting, LinearSolvePyAMG, Test
using JET

docs_src = normpath(joinpath(pkgdir(LinearSolvePyAMG), "..", "..", "docs", "src"))

run_qa(
    LinearSolvePyAMG;
    explicit_imports = true,
    api_docs_kwargs = (; rendered = true, docs_src),
    jet_kwargs = (; target_defined_modules = true),
    ei_kwargs = (;
        # Non-public names accessed qualified: SciMLBase / LinearSolve internals the
        # solver wrapper relies on (build_linear_solution, default_alias_A/b, ...).
        all_qualified_accesses_are_public = (;
            ignore = (
                :SciMLLinearSolveAlgorithm, :Success, :build_linear_solution,
                :default_alias_A, :default_alias_b, :init_cacheval,
                :needs_concrete_A, :update_tolerances_internal!,
            ),
        ),
        # LinearCache is not declared public in LinearSolve.
        all_explicit_imports_are_public = (; ignore = (:LinearCache,)),
    ),
    # `using LinearAlgebra, SparseArrays, PythonCall, CondaPkg` brings several names
    # in implicitly; making them explicit is a source refactor tracked in
    # https://github.com/SciML/LinearSolve.jl/issues/1058
    ei_broken = (:no_implicit_imports,),
)
