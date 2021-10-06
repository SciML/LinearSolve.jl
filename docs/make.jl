using LinearSolve
using Documenter

DocMeta.setdocmeta!(LinearSolve, :DocTestSetup, :(using LinearSolve); recursive=true)

makedocs(;
    modules=[LinearSolve],
    authors="Jonathan <edelman.jonathan.s@gmail.com> and contributors",
    repo="https://github.com/SciML/LinearSolve.jl/blob/{commit}{path}#{line}",
    sitename="LinearSolvers.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://linearsolve.sciml.ai/",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/SciML/LinearSolve.jl",
)
