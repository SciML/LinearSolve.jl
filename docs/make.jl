using LinearSolvers
using Documenter

DocMeta.setdocmeta!(LinearSolvers, :DocTestSetup, :(using LinearSolvers); recursive=true)

makedocs(;
    modules=[LinearSolvers],
    authors="Jonathan <edelman.jonathan.s@gmail.com> and contributors",
    repo="https://github.com/SciML/LinearSolvers.jl/blob/{commit}{path}#{line}",
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
    repo="github.com/SciML/LinearSolvers.jl",
)
