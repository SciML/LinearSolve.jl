module LinearSolveCUDSSExt

using LinearSolve: LinearSolve
using CUDSS: CUDSS

LinearSolve.cudss_loaded(A::CUDSS.cuSPARSE.CuSparseMatrixCSR) = true

end
