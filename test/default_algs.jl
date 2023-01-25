using LinearSolve, SparseArrays
LinearSolve.defaultalg(nothing, zeros(3)) isa GenericLUFactorization
LinearSolve.defaultalg(nothing, zeros(50)) isa RFLUFactorization
LinearSolve.defaultalg(nothing, zeros(600)) isa LUFactorization
LinearSolve.defaultalg(Diagonal(zeros(5)), zeros(5)) isa DiagonalFactorization

LinearSolve.defaultalg(nothing, zeros(5), OperatorAssumptions{false}) isa QRFactorization

LinearSolve.defaultalg(sprand(1000,1000,0.01), zeros(1000)) isa KLUFactorization
LinearSolve.defaultalg(sprand(11000,11000,0.001), zeros(11000)) isa UMFPACKFactorization