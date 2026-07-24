# SPDX-FileCopyrightText: 2026 Chris Rackauckas <accounts@chrisrackauckas.com> and contributors
# SPDX-License-Identifier: MIT
#
# SupernodalLU — a pure-Julia implementation of the supernodal
# left-right-looking sparse LU method of O. Schenk and K. Gärtner
# (FGCS 20(3), 2004; ETNA 23, 2006): supernodal BLAS-3 LU on the symmetrized
# nonzero pattern of A + Aᵀ, pivoting restricted to the supernode diagonal
# block with static pivot perturbation (ε‖A‖) compensated by iterative
# refinement, and maximum-weight matching + scaling preprocessing.
# Implemented from the papers; see NOTICE.md for the full per-component
# lineage (no code from the proprietary PARDISO library, which popularized
# the method, nor from Intel MKL PARDISO or HSL; `amd.jl` is a BSD-3-Clause
# SuiteSparse AMD port).
#
# Internal to LinearSolve: the public surface is `SupernodalLUFactorization`.
# Entry points here are `snlu` (analyze + factor), `snlu!` (refactorize, same
# pattern), `solve!`, and the `snlu_symbolic` analysis.
# Each supernode's dense diagonal block is factored through LinearSolve's own
# dense `init`/`solve!` machinery (a per-supernode dense `LinearCache` over a
# view of the panel), so the sparse solver automatically uses whatever
# LinearSolve's dense default resolves to — RFLUFactorization when
# RecursiveFactorization is loaded, MKL/LAPACK/generic otherwise — with no
# backend-specific wiring here.  Static pivoting is preserved by an
# optimistic fast path: accept the dense factorization iff every pivot
# clears the ε‖A‖ threshold, else restore the block and rerun the built-in
# static-perturbation kernel.

module SupernodalLU

using SparseArrays: SparseMatrixCSC, sparse, getcolptr, rowvals, nonzeros, nnz, spzeros
using LinearAlgebra: LinearAlgebra, SingularException, UpperTriangular,
    UnitLowerTriangular, ldiv!, rdiv!, mul!, norm, BLAS
using ..LinearSolve: LinearSolve
using SciMLBase: SciMLBase

include("amd.jl")          # vendored BSD-3 SuiteSparse AMD port (module AMD)
include("ordering.jl")
include("symbolic.jl")
include("matching.jl")
include("numeric.jl")
include("solve.jl")
include("interface.jl")

end # module
