using SciMLTesting, LinearSolvePyAMG, Test
using JET

run_qa(
    LinearSolvePyAMG;
    explicit_imports = true,
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
)
