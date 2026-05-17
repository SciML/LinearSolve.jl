module LinearSolveCUDSSExt

using LinearSolve: LinearSolve, cudss_loaded
using CUDSS

LinearSolve.cudss_loaded(A::CUDSS.cuSPARSE.CuSparseMatrixCSR) = true

end
