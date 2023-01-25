using LinearSolve, LinearAlgebra, SparseArrays, Test
@test LinearSolve.defaultalg(nothing, zeros(3)) isa GenericLUFactorization
@test LinearSolve.defaultalg(nothing, zeros(50)) isa RFLUFactorization
@test LinearSolve.defaultalg(nothing, zeros(600)) isa LUFactorization
@test LinearSolve.defaultalg(LinearAlgebra.Diagonal(zeros(5)), zeros(5)) isa
      DiagonalFactorization

@test LinearSolve.defaultalg(nothing, zeros(5),
                             LinearSolve.OperatorAssumptions{false}()) isa QRFactorization

@test LinearSolve.defaultalg(sprand(1000, 1000, 0.01), zeros(1000)) isa KLUFactorization
@test LinearSolve.defaultalg(sprand(11000, 11000, 0.001), zeros(11000)) isa
      UMFPACKFactorization
