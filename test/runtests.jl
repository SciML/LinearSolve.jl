using Pkg
using SafeTestsets
const LONGER_TESTS = false

const GROUP = get(ENV, "GROUP", "All")

const HAS_EXTENSIONS = isdefined(Base, :get_extension)

if GROUP == "All" || GROUP == "Core"
    @time @safetestset "Basic Tests" begin include("basictests.jl") end
    @time @safetestset "Re-solve" begin include("resolve.jl") end
    @time @safetestset "Zero Initialization Tests" begin include("zeroinittests.jl") end
    @time @safetestset "Non-Square Tests" begin include("nonsquare.jl") end
    @time @safetestset "SparseVector b Tests" begin include("sparse_vector.jl") end
    @time @safetestset "Default Alg Tests" begin include("default_algs.jl") end
    @time @safetestset "Traits" begin include("traits.jl") end
end

if GROUP == "LinearSolveCUDA"
    Pkg.activate("gpu")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
    @time @safetestset "CUDA" begin include("gpu/cuda.jl") end
end

if GROUP == "LinearSolvePardiso"
    Pkg.activate("pardiso")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
    @time @safetestset "Pardiso" begin include("pardiso/pardiso.jl") end
end

if (GROUP == "All" || GROUP == "LinearSolveHYPRE") && HAS_EXTENSIONS
    @time @safetestset "LinearSolveHYPRE" begin include("hypretests.jl") end
end
