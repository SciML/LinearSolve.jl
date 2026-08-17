using SciMLTesting, LinearSolveAutotune, Test
using JET

run_qa(
    LinearSolveAutotune;
    explicit_imports = true,
    # `plot` is deliberately re-exported from Plots: the package extends it with
    # AutotuneResults methods and exports it as the plotting entry point.
    reexports_allow = (:plot,),
    ei_kwargs = (;
        # BlasFloat (LinearAlgebra.BLAS, reached via LinearAlgebra) and Base.run.
        all_qualified_accesses_via_owners = (; ignore = (:BlasFloat, :run)),
        # Non-public names accessed qualified: LinearSolve internals
        # (get_config/get_extension/is_available/userecursivefactorization/...),
        # Base/Pkg internals (PkgId, UUID, dependencies, loaded_modules, run, ...),
        # and CPUSummary/blis names used by the benchmarking harness.
        # LinearAlgebra.BLAS.set_num_threads is the only stdlib API that pins
        # panel benchmarks to one BLAS thread, but it is not public on Julia 1.12.
        all_qualified_accesses_are_public = (;
            ignore = (
                :BLISLUFactorization, :BlasFloat, :GIT_VERSION_INFO, :Parameters,
                :PkgId, :UUID, :appleaccelerate_isavailable, :dependencies, :format,
                :functional, :get_config, :get_extension, :get_num_threads,
                :is_available, :libm_name, :loaded_modules, :run,
                :set_num_threads, :userecursivefactorization, :vendor,
            ),
        ),
    ),
)
