# SPDX-FileCopyrightText: 2026 Chris Rackauckas <accounts@chrisrackauckas.com> and contributors
# SPDX-License-Identifier: MIT
#
# SupernodalQR — a pure-Julia multifrontal sparse Householder QR in the
# George–Heath / Matstoms tradition: the column elimination tree of A
# (the elimination tree of AᵀA) drives a supernodal assembly tree whose
# dense frontal matrices are factored by BLAS-3 Householder QR (LAPACK
# `geqrf!`), with Heath's dead-column rule for rank deficiency.  Rectangular
# systems are supported: m ≥ n gives the least-squares solution, m < n
# factors Aᴴ and gives the minimum-norm solution.  Implemented from the
# papers; see NOTICE.md for the per-component lineage (no code from
# SuiteSparseQR/SPQR, which is GPL).
#
# The symbolic phase reuses SupernodalLU's machinery wholesale: the R factor
# of A has the structure of the Cholesky factor of AᵀA, which is computed
# here on a "star" reduction of AᵀA (each row of A contributes edges from its
# leftmost column only — same filled graph, O(nnz(A)) size) and then handed
# to SupernodalLU's etree/postorder/row-subtree/supernode/amalgamation code.
#
# Internal to LinearSolve: the public surface is `SupernodalQRFactorization`.
# Entry points here are `snqr` (analyze + factor), `snqr!` (refactorize, same
# pattern), `solve!`, and the `snqr_symbolic` analysis.

module SupernodalQR

using SparseArrays: SparseMatrixCSC, sparse, getcolptr, rowvals, nonzeros, nnz
using LinearAlgebra: LinearAlgebra, I, UpperTriangular, ldiv!, lmul!, mul!, norm, LAPACK
using ..SupernodalLU: AMD, etree_sym, postorder_tree,
    _children_csr, permute_pattern, fundamental_supernodes

include("symbolic.jl")
include("numeric.jl")
include("solve.jl")
include("interface.jl")

end # module
