using LinearSolve, Aqua
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(LinearSolve)
    Aqua.test_ambiguities(LinearSolve, recursive = false, broken = true)
    Aqua.test_deps_compat(LinearSolve, ignore = [:MKL_jll])
    Aqua.test_piracies(LinearSolve,
        treat_as_own = [LinearProblem])
    Aqua.test_project_extras(LinearSolve)
    Aqua.test_stale_deps(LinearSolve, ignore = [:MKL_jll])
    Aqua.test_unbound_args(LinearSolve)
    Aqua.test_undefined_exports(LinearSolve)
end
