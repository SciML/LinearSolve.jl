SupernodalLU (LinearSolve src/SupernodalLU) — code lineage and licensing notice
==================================================

Summary: MIT for this folder, BSD-3-Clause for the vendored `amd.jl`.
No GPL/LGPL-licensed code and no proprietary code is included.  The full
license texts are in LICENSE.  Per-component provenance:

1. The PARDISO method itself (supernodal left-looking LU on the symmetrized
   pattern, restricted-block pivoting with static perturbation, weighted
   matching preprocessing) is implemented from the published literature:

     - O. Schenk, K. Gärtner: "Solving unsymmetric sparse systems of linear
       equations with PARDISO", FGCS 20(3), 2004.
     - O. Schenk, K. Gärtner: "On fast factorization pivoting methods for
       sparse symmetric indefinite systems", ETNA 23, 2006.

   PARDISO (pardiso-project.org) and Intel MKL PARDISO are closed-source;
   no source code of either exists publicly and none was used.

2. `amd.jl` is a pure-Julia port of SuiteSparse AMD, vendored from
   PureKLU.jl.  SuiteSparse AMD is BSD-3-Clause, (c) 1996-2025 Timothy A.
   Davis, Patrick R. Amestoy, and Iain S. Duff; the file remains under
   BSD-3-Clause (text in LICENSE).

3. `src/matching.jl` implements maximum-weight (max-product) bipartite
   matching with dual-variable scalings from the published algorithm
   descriptions (I. S. Duff, J. Koster, SIMAX 22(4), 2001; M. Olschowka,
   A. Neumaier, Linear Algebra Appl. 240, 1996), as shortest augmenting
   paths with potentials (Jonker–Volgenant style).  HSL MC64 is proprietary
   and its source was not used.

4. `src/symbolic.jl` implements standard published symbolic-analysis
   algorithms — the elimination tree via union-find with path halving
   (J. W. H. Liu, "The role of elimination trees in sparse factorization",
   SIMAX 11(1), 1990), L-structure prediction by row subtrees (J. R.
   Gilbert, E. Ng, B. W. Peyton, SIMAX 15(1), 1994), and relaxed supernode
   amalgamation in the style of Ashcraft–Grimes with the relaxation
   constants published for CHOLMOD (Y. Chen, T. A. Davis, W. W. Hager,
   S. Rajamanickam, ACM TOMS 35(3), 2008).  The implementations were
   written from the algorithm descriptions and deliberately use different
   formulations from the CSparse/SuiteSparse (LGPL/GPL) implementations of
   the same algorithms: union-find etree instead of ancestor stamping,
   cursor-stack DFS over counting-sorted child arrays instead of
   destructive sibling lists, row-subtree column construction instead of
   child-set merging, and single bucket-pass pattern permutation instead
   of per-column sorting.

5. The banded-detection acceptance thresholds in `symbolic.jl` follow the
   heuristic introduced in PureKLU.jl by the same copyright holder.

6. Everything else (`numeric.jl`, `solve.jl`, `interface.jl`,
   `ordering.jl` apart from the AMD calls, and the
   RecursiveFactorization/TriangularSolve kernel overrides in
   LinearSolveRecursiveFactorizationExt) is original work of the same
   authors, MIT-licensed.

For comparison, the related packages carry different licenses because they
ARE ports: PureKLU.jl is LGPL-2.1+ (KLU/BTF translation) and PureUMFPACK.jl
is GPL-2.0+ (UMFPACK/CSparse translation).  This folder is not a
translation of any existing codebase, which is why MIT (+ BSD-3 for the
vendored `amd.jl`) is the appropriate license, matching LinearSolve's.
