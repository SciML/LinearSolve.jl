using SafeTestsets

const TEST_GROUP = get(ENV, "LINEARSOLVE_TEST_GROUP", "All")

if TEST_GROUP == "Core" || TEST_GROUP == "All"
    @time @safetestset "LinearSolveAutotune Tests" include("autotune_tests.jl")
end
