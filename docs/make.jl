using LinearSolve
using Documenter

DocMeta.setdocmeta!(LinearSolve, :DocTestSetup, :(using LinearSolve); recursive = true)

include("pages.jl")

makedocs(sitename = "LinearSolve.jl",
         authors = "Chris Rackauckas",
         modules = [LinearSolve, LinearSolve.SciMLBase],
         clean = true, doctest = false,
         strict = [
             :doctest,
             :linkcheck,
             :parse_error,
             :example_block,
             # Other available options are
             # :autodocs_block, :cross_references, :docs_block, :eval_block, :example_block, :footnote, :meta_block, :missing_docs, :setup_block
         ],
         format = Documenter.HTML(analytics = "UA-90474609-3",
                                  assets = ["assets/favicon.ico"],
                                  canonical = "https://docs.sciml.ai/LinearSolve/stable/"),
         pages = pages)

deploydocs(;
           repo = "github.com/SciML/LinearSolve.jl",
           devbranch = "main")
