# Put in a separate page so it can be used by SciMLDocs.jl

pages = ["index.md",
    "Tutorials" => Any["tutorials/linear.md",
        "tutorials/caching_interface.md",
        "tutorials/accelerating_choices.md"],
    "Basics" => Any["basics/LinearProblem.md",
        "basics/common_solver_opts.md",
        "basics/OperatorAssumptions.md",
        "basics/Preconditioners.md",
        "basics/FAQ.md"],
    "Solvers" => Any["solvers/solvers.md"],
    "Advanced" => Any["advanced/developing.md"
                      "advanced/custom.md"],
    "Release Notes" => "release_notes.md"
]
