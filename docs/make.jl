using LinearSolve
using Documenter

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

DocMeta.setdocmeta!(LinearSolve, :DocTestSetup, :(using LinearSolve); recursive = true)

include("pages.jl")

makedocs(
    sitename = "LinearSolve.jl",
    authors = "Chris Rackauckas",
    modules = [LinearSolve],
    clean = true, doctest = false, linkcheck = true,
    linkcheck_ignore = [
        "https://cli.github.com/manual/installation",
    ],
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/LinearSolve/stable/"
    ),
    pages = pages
)

deploydocs(;
    repo = "github.com/SciML/LinearSolve.jl",
    push_preview = true
)
