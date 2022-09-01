using Pkg
using SafeTestsets
const LONGER_TESTS = false

const GROUP = get(ENV, "GROUP", "All")

function dev_subpkg(subpkg)
    subpkg_path = joinpath(dirname(@__DIR__), "lib", subpkg)
    Pkg.develop(PackageSpec(path = subpkg_path))
end

function activate_subpkg_env(subpkg)
    subpkg_path = joinpath(dirname(@__DIR__), "lib", subpkg)
    Pkg.activate(subpkg_path)
    Pkg.develop(PackageSpec(path = subpkg_path))
    Pkg.instantiate()
end

@show GROUP, GROUP == "LinearSolvePardiso"

if GROUP == "All" || GROUP == "Core"
    @time @safetestset "Basic Tests" begin include("basictests.jl") end
    @time @safetestset "Zero Initialization Tests" begin include("zeroinittests.jl") end
    @time @safetestset "Non-Square Tests" begin include("nonsquare.jl") end
end

if GROUP == "LinearSolveCUDA"
    dev_subpkg("LinearSolveCUDA")
    @time @safetestset "CUDA" begin include("../lib/LinearSolveCUDA/test/runtests.jl") end
end

if GROUP == "LinearSolvePardiso"
    dev_subpkg("LinearSolvePardiso")
    @time @safetestset "Pardiso" begin include("../lib/LinearSolvePardiso/test/runtests.jl") end
end
