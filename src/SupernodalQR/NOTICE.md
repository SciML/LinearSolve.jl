SupernodalQR (LinearSolve src/SupernodalQR) — code lineage and licensing notice
==================================================

Summary: MIT.  No GPL/LGPL-licensed code and no proprietary code is
included.  Per-component provenance:

1. The method — multifrontal sparse Householder QR driven by the column
   elimination tree, with rows of A assembled into the front of their
   leftmost column and contribution blocks passed up the assembly tree — is
   implemented from the published literature:

     - A. George, M. T. Heath: "Solution of sparse linear least squares
       problems using Givens rotations", Linear Algebra Appl. 34, 1980.
     - J. W. H. Liu: "On general row merging schemes for sparse Givens
       transformations", SIAM J. Sci. Stat. Comput. 7(4), 1986.
     - P. Matstoms: "Sparse QR factorization in MATLAB", ACM TOMS 20(1), 1994.
     - T. A. Davis: "Algorithm 915, SuiteSparseQR: Multifrontal multithreaded
       rank-revealing sparse QR factorization", ACM TOMS 38(1), 2011
       (method description only).

   SuiteSparseQR/SPQR is GPL-2.0+; none of its source code was used.

2. Rank handling follows Heath's dead-column rule (M. T. Heath, "Some
   extensions of an algorithm for sparse linear least squares problems",
   SIAM J. Sci. Stat. Comput. 3(2), 1982), with the default tolerance
   formula `20 (m + n) eps max_j ‖A[:, j]‖` as published in Davis (2011).

3. The symbolic analysis never forms AᵀA.  It runs on a star reduction of
   AᵀA (each row of A contributes edges from its leftmost column only).
   Eliminating that column first turns the star into the row's clique, so
   the filled graph equals that of AᵀA; this is a direct consequence of the
   elimination-graph model and is implemented here from that argument.
   Column counts of R use the row-subtree counting characterization of
   J. R. Gilbert, E. G. Ng, B. W. Peyton, "An efficient algorithm to
   compute row and column counts for sparse Cholesky factorization", SIMAX
   15(4), 1994, implemented from the paper.

4. The column ordering is approximate minimum degree on the quotient graph
   whose initial elements are the rows of A (the idea behind COLAMD, which
   was not used).  It runs on the vendored BSD-3-Clause SuiteSparse AMD port
   in `src/SupernodalLU/amd.jl`, extended for this purpose with a `preset`
   start from a caller-built quotient graph; that modification remains
   under BSD-3-Clause.  Column singletons are taken as described in Davis
   (2011).

5. Elimination tree, postorder, and fundamental supernodes are reused from
   `src/SupernodalLU` (see its NOTICE.md).  Relaxed amalgamation follows the
   CHOLMOD scheme of merging a supernode into its adjacent parent (Chen,
   Davis, Hager & Rajamanickam, ACM TOMS 35(3), 2008) but decides with an
   original QR front cost model (Householder work plus stored entries).

6. Everything else (`symbolic.jl`, `numeric.jl`, `solve.jl`,
   `interface.jl`) is original work of the same authors, MIT-licensed.
