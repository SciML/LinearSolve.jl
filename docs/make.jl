using LinearSolve
using LinearSolveAutotune
using LinearSolvePyAMG
using SparseArrays
using Documenter

const LinearSolveSparseArraysExt = Base.get_extension(LinearSolve, :LinearSolveSparseArraysExt)

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

DocMeta.setdocmeta!(LinearSolve, :DocTestSetup, :(using LinearSolve); recursive = true)

include("pages.jl")

makedocs(
    sitename = "LinearSolve.jl",
    authors = "Chris Rackauckas",
    modules = [
        LinearSolve,
        LinearSolve.SciMLBase,
        LinearSolveAutotune,
        LinearSolvePyAMG,
        LinearSolveSparseArraysExt.KLU,
    ],
    clean = true, doctest = false, linkcheck = true,
    warnonly = [:docs_block, :missing_docs],
    linkcheck_ignore = [
        "https://cli.github.com/manual/installation",
    ],
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/LinearSolve/stable/",
        # solvers.md is a long auto-listed solver catalog that exceeds the
        # default 200 KiB per-page HTML limit; it is reference material meant
        # to be read in one page.
        size_threshold = 500 * 1024
    ),
    pages = pages
)

deploydocs(;
    repo = "github.com/SciML/LinearSolve.jl",
    push_preview = true
)
