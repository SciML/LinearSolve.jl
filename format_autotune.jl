using Pkg
Pkg.add("JuliaFormatter")
using JuliaFormatter

# Format only the changed files with SciMLStyle
format("lib/LinearSolveAutotune/src/gpu_detection.jl", SciMLStyle())
format("lib/LinearSolveAutotune/src/telemetry.jl", SciMLStyle())
format("lib/LinearSolveAutotune/src/benchmarking.jl", SciMLStyle())

println("Formatting complete!")