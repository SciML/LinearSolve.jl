module LinearSolveCUDSSExt

using LinearSolve
using CUDSS

LinearSolve.cudss_loaded(A::CUDSS.CUDA.CUSPARSE.CuSparseMatrixCSR) = true

end
