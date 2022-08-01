# Overview

LinearSolve.jl is a high-performance unified interface for the linear solving packages of
Julia. It interfaces with other packages of the Julia ecosystem
to make it easy to test alternative solver packages and pass small types to
control algorithm swapping. It also interfaces with the
[ModelingToolkit.jl](https://mtk.sciml.ai/dev/) world of symbolic modeling to
allow for automatically generating high-performance code.

## Roadmap

Wrappers for every linear solver in the Julia language is on the roadmap. If
there are any important ones that are missing that you would like to see added,
please open an issue. The current algorithms should support automatic differentiation.
Pre-defined preconditioners would be a welcome addition.
