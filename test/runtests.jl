using Pkg
using SafeTestsets
const LONGER_TESTS = false

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV,"APPVEYOR")

function activate_downstream_env()
    Pkg.activate("downstream")
    Pkg.develop(PackageSpec(path=dirname(@__DIR__)))
    Pkg.instantiate()
end

if GROUP == "All" || GROUP == "Core"
  @time @safetestset "Basic Tests" begin include("basictests.jl") end
end

if !is_APPVEYOR && GROUP == "GPU"
  activate_downstream_env()
  @time @safetestset "CUDA" begin include("cuda.jl") end
end
