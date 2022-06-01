using LinearSolve
using Documenter

DocMeta.setdocmeta!(LinearSolve, :DocTestSetup, :(using LinearSolve); recursive=true)

makedocs(
    sitename="LinearSolve.jl",
    authors="Chris Rackauckas",
    modules=[LinearSolve,LinearSolve.SciMLBase],
    clean=true,doctest=false,
    format = Documenter.HTML(analytics = "UA-90474609-3",
                             assets = ["assets/favicon.ico"],
                             canonical="https://linearsolve.sciml.ai/stable/"),
    pages=pages
)

deploydocs(;
    repo="github.com/SciML/LinearSolve.jl",
    devbranch="main",
)
