using FastAlmostBandedMatrices, BandedMatrices, LinearAlgebra, LinearSolve, Test

# Square Case
n = 8
A = BandedMatrix(Matrix(I, n, n), (2, 2))
b = ones(n)
A1 = A / 1;
b1 = rand(n);
x1 = zero(b);
A2 = A / 2;
b2 = rand(n);
x2 = zero(b);

sol1 = solve(LinearProblem(A1, b1; u0 = x1))
@test sol1.u ≈ A1 \ b1
sol2 = solve(LinearProblem(A2, b2; u0 = x2))
@test sol2.u ≈ A2 \ b2

A = AlmostBandedMatrix(BandedMatrix(fill(2.0, n, n), (1, 1)), fill(3.0, 2, n))
A[band(0)] .+= 1:n

sol1ab = solve(LinearProblem(A, b; u0 = x1))
@test sol1ab.u ≈ Matrix(A) \ b

# Square Symmetric
A1s = Symmetric(A1)
A2s = Symmetric(A2)

sol1s = solve(LinearProblem(A1s, b1; u0 = x1))
@test sol1s.u ≈ A1s \ b1
sol2s = solve(LinearProblem(A2s, b2; u0 = x2))
@test sol2s.u ≈ A2s \ b2

# Underdetermined
A = BandedMatrix(rand(8, 10), (2, 2))
b = rand(8)

@test_throws ErrorException solve(LinearProblem(A, b)).u

A = AlmostBandedMatrix(BandedMatrix(fill(2.0, n - 2, n), (1, 1)), fill(3.0, 2, n))
A[band(0)] .+= 1:(n - 2)

@test_throws ErrorException solve(LinearProblem(A, b)).u

# Overdetermined
A = BandedMatrix(ones(10, 8), (2, 0))
b = rand(10)

@test_nowarn solve(LinearProblem(A, b))

A = AlmostBandedMatrix(BandedMatrix(fill(2.0, n + 2, n), (1, 1)), fill(3.0, 2, n))
A[band(0)] .+= 1:n

@test_nowarn solve(LinearProblem(A, b))

# Workaround for no lu from BandedMatrices
A = BandedMatrix{BigFloat}(ones(3, 3), (0, 0))
b = BigFloat[1, 2, 3]
prob = LinearProblem(A, b)
@test_nowarn solve(prob)
