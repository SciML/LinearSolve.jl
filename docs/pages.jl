# Put in a separate page so it can be used by SciMLDocs.jl

pages = [
    "index.md",
    "tutorials/linear.md",
    "Tutorials" => Any[
        "tutorials/caching_interface.md",
        "tutorials/accelerating_choices.md",
        "tutorials/gpu.md",
        "tutorials/autotune.md",
    ],
    "Basics" => Any[
        "basics/LinearProblem.md",
        "basics/algorithm_selection.md",
        "basics/common_solver_opts.md",
        "basics/OperatorAssumptions.md",
        "basics/Preconditioners.md",
        "basics/FAQ.md",
    ],
    "Solvers" => Any["solvers/solvers.md"],
    "Advanced" => Any[
        "advanced/developing.md",
        "advanced/custom.md",
        "advanced/internal_api.md",
    ],
    "Release Notes" => "release_notes.md",
]
