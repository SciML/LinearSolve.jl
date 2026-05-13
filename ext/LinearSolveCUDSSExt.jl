module LinearSolveCUDSSExt

using LinearSolve: LinearSolve, cudss_loaded
using CUDSS

LinearSolve.cudss_loaded(::CUDSS.cuSPARSE.CuSparseMatrixCSR) = true

end
