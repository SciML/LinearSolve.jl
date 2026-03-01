---
title: 'LinearSolve.jl: High-Performance Unified Interface for Linear System Solvers'
tags:
  - julia
  - linear algebra
  - numerical methods
  - high-performance computing
authors:
  - name: Christopher Rackauckas
    orcid: 0000-0001-5850-0663
    corresponding: true
    affiliation: "1, 2, 3"
affiliations:
 - name: JuliaHub
   index: 1
 - name: Pumas-AI
   index: 2
 - name: Massachusetts Institute of Technology
   index: 3
date: 25 October 2025
bibliography: paper.bib
---

# Summary

Solving linear systems of the form $Ax = b$ is a fundamental operation in scientific computing, appearing in applications ranging from machine learning to differential equations. LinearSolve.jl is a Julia [@Bezanson2017] package that provides a unified, high-performance interface for linear system solvers across the Julia ecosystem. The package implements the SciML common interface, enabling users to easily swap between different solver algorithms while maintaining maximum efficiency. LinearSolve.jl includes:

  - Fast pure Julia LU factorizations which can outperform standard BLAS implementations
  - Specialized sparse matrix solvers including KLU [@Davis2010KLU], UMFPACK [@Davis2004UMFPACK], and Pardiso [@Schenk2004Pardiso]
  - Sparspak.jl [@Sparspak] for pure Julia sparse LU factorization supporting generic number types
  - GPU-offloading capabilities for large dense and sparse matrices via CUDA.jl [@besard2018juliagpu], Metal.jl, and AMDGPU.jl
  - Comprehensive wrappers for Krylov methods from Krylov.jl [@Krylov], IterativeSolvers.jl [@IterativeSolvers], and KrylovKit.jl [@KrylovKit]
  - A polyalgorithm that intelligently selects optimal methods based on problem characteristics
  - An advanced caching interface that optimizes symbolic and numerical factorizations
  - Mixed precision solvers for memory-bandwidth-limited problems
  - Integration with ModelingToolkit.jl [@ma2021modelingtoolkit] for symbolic modeling

# Statement of need

Linear system solving is ubiquitous in scientific computing, yet choosing the right solver for a given problem requires significant expertise. Different algorithms excel in different regimes: direct methods like LU factorization are efficient for small to medium dense systems, sparse factorizations leverage structure in large sparse systems, and iterative Krylov methods scale to very large systems when lower tolerance is acceptable. Furthermore, modern hardware diversity (CPUs with various BLAS implementations, NVIDIA GPUs, AMD GPUs, Apple Silicon) means that optimal performance often requires hardware-specific implementations.

LinearSolve.jl addresses these challenges by providing a unified interface that abstracts away implementation details while maintaining high performance. Users define a `LinearProblem` once and can easily experiment with different solvers by changing a single argument. The package automatically handles API differences between underlying solver libraries, particularly in preconditioner definitions and matrix format requirements. This design philosophy aligns with the broader SciML ecosystem's goal of composable, high-performance scientific computing tools.

The package distinguishes itself from Base Julia's LinearAlgebra in several ways:

  - **Performance optimizations**: Pure Julia implementations like `RFLUFactorization` [@RecursiveFactorization] can outperform BLAS for small to medium matrices, while hardware-specific wrappers (`AppleAccelerateLUFactorization`, `MKLLUFactorization`) provide optimal performance across different platforms
  - **Caching infrastructure**: Automatic reuse of symbolic and numerical factorizations when solving multiple systems with the same structure
  - **Unified Krylov interface**: Seamless access to multiple Krylov implementations with consistent preconditioner handling
  - **GPU acceleration**: First-class support for offloading to CUDA, Metal, and AMD GPUs with automatic data transfer
  - **Polyalgorithm selection**: Intelligent algorithm choice based on matrix properties and problem size
  - **Sparse solver diversity**: Access to KLU, UMFPACK, Pardiso, Sparspak, and CliqueTrees solvers through a common interface

LinearSolve.jl is a critical dependency in the SciML ecosystem, particularly for DifferentialEquations.jl [@rackauckas2017differentialequations] where linear solves appear in implicit time integration and nonlinear system solving. It enables downstream packages and users to easily experiment with different linear solvers to find optimal performance for their specific applications.

# Example

The package provides comprehensive tutorials in the documentation:

  - [Tutorial I: Basics](https://docs.sciml.ai/LinearSolve/stable/tutorials/linear/) demonstrates defining linear problems and solving with different algorithms
  - [Tutorial II: Caching Interface](https://docs.sciml.ai/LinearSolve/stable/tutorials/caching_interface/) explains how to efficiently reuse factorizations
  - [Tutorial III: GPU Computing](https://docs.sciml.ai/LinearSolve/stable/tutorials/gpu/) shows GPU-accelerated solving for large matrices
  - [Tutorial IV: Autotuning](https://docs.sciml.ai/LinearSolve/stable/tutorials/autotune/) demonstrates automatic algorithm selection

A simple demonstration showing the unified interface:

```julia
using LinearSolve

# Define the linear problem
n = 4
A = rand(n, n)
b = rand(n)
prob = LinearProblem(A, b)

# Initialize with default algorithm
linsolve = init(prob)
sol = solve!(linsolve)
sol.u  # Solution vector

# Reuse for a different right-hand side
b2 = rand(n)
linsolve.b = b2
sol2 = solve!(linsolve)

# Switch to an iterative solver (GMRES)
using IterativeSolvers  # Load extension
linsolve = init(prob, IterativeSolversJL_GMRES())
sol3 = solve!(linsolve)

# Update the matrix and resolve
A2 = rand(n, n)
linsolve.A = A2
sol4 = solve!(linsolve)
```

This example demonstrates the key features: defining a problem once, efficiently solving multiple related systems through caching, and seamlessly switching between solver algorithms.

# Acknowledgements

We acknowledge contributions from the entire SciML community and the JuliaLang community for their development of the underlying solver packages that LinearSolve.jl interfaces with.

# References
