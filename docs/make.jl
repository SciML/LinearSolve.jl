using LinearSolve
using Documenter

DocMeta.setdocmeta!(LinearSolve, :DocTestSetup, :(using LinearSolve); recursive=true)

makedocs(
    sitename="LinearSolve.jl",
    authors="Chris Rackauckas",
    modules=[LinearSolve],
    clean=true,doctest=false,
    format = Documenter.HTML(#analytics = "UA-90474609-3",
                             assets = ["assets/favicon.ico"],
                             canonical="https://linearsolve.sciml.ai/stable/"),
    pages=[
        "Home" => "index.md",
        "Tutorials" => Any[
            "tutorials/linear.md"
            "tutorials/caching_interface.md"
        ],
        "Basics" => Any[
            "basics/LinearProblem.md",
            "basics/CachingAPI.md",
            "basics/FAQ.md"
        ],
        "Solvers" => Any[
            "solvers/solvers.md"
        ]
    ]
)

deploydocs(;
    repo="github.com/SciML/LinearSolve.jl",
)
