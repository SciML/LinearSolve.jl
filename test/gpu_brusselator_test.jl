using OrdinaryDiffEq, LinearSolve, CUDA, CUDSS, Symbolics, SparseDiffTools, SparseArrays, Test

# --- CONFIGURAZIONE TEST ---
# Disabilitiamo lo scalar indexing: se il codice prova a farlo, il test fallisce.
# Questo è esattamente ciò che vogliamo per il debug.
CUDA.allowscalar(false)

println("--- INIZIO TEST BRUSSELATOR GPU CON CUDSS ---")

# Parametri della Brusselator 2D
const N = 32
const xyd_brusselator = range(0, stop=1, length=N)
brusselator_f(x, y, t) = (((x-0.3)^2 + (y-0.6)^2) <= 0.1^2) * (t >= 1.1) * 5.0
limit(a, N) = a == N+1 ? 1 : a == 0 ? N : a

function brusselator_2d(du, u, p, t)
    A, B, α = p[1], p[2], p[3]/step(xyd_brusselator)^2
    II = LinearIndices((N, N, 2))
    for I in CartesianIndices((N, N))
        i, j = Tuple(I)
        x, y = xyd_brusselator[i], xyd_brusselator[j]
        ip1, im1 = limit(i+1, N), limit(i-1, N)
        jp1, jm1 = limit(j+1, N), limit(j-1, N)
        du[II[i,j,1]] = α*(u[II[im1,j,1]] + u[II[ip1,j,1]] + u[II[i,jp1,1]] + u[II[i,jm1,1]] - 4u[II[i,j,1]]) +
                        B + u[II[i,j,1]]^2*u[II[i,j,2]] - (A + 1)*u[II[i,j,1]] + brusselator_f(x, y, t)
        du[II[i,j,2]] = α*(u[II[im1,j,2]] + u[II[ip1,j,2]] + u[II[i,jp1,2]] + u[II[i,jm1,2]] - 4u[II[i,j,2]]) +
                        A*u[II[i,j,1]] - u[II[i,j,1]]^2*u[II[i,j,2]]
    end
end

# Inizializzazione
u0 = zeros(N, N, 2)
for I in CartesianIndices((N, N))
    x, y = xyd_brusselator[I[1]], xyd_brusselator[I[2]]
    u0[I,1] = 22*(y*(1-y))^(3/2); u0[I,2] = 27*(x*(1-x))^(3/2)
end

p = (3.4, 1.0, 10.0)
u0_gpu = cu(u0)

# Generazione Sparsità del Jacobiano
println("Calcolo sparsità e colorazione...")
du0 = copy(u0)
jac_sparsity = float.(Symbolics.jacobian_sparsity((du,u)->brusselator_2d(du,u,p,0.0), du0, u0))
jac_cusparse = CUDA.CUSPARSE.CuSparseMatrixCSR(jac_sparsity)
colorvec = matrix_colors(jac_sparsity)

# Definizione del problema ODE
f = ODEFunction(brusselator_2d; jac_prototype=jac_cusparse, colorvec=colorvec)
prob = ODEProblem(f, u0_gpu, (0.0f0, 11.5f0), p)

# --- IL TEST VERO E PROPRIO ---
@testset "Brusselator GPU with CUDSS" begin
    println("Lancio solve()...")
    # Se questo fallisce per scalar indexing, GitHub Actions ci darà il log
    sol = solve(prob, Rosenbrock23(linsolve=CUDSSFactorization()), save_everystep=false)
    @test sol.retcode == ReturnCode.Success
end

println("--- TEST CONCLUSO ---")
