using BenchmarkTools, LinearAlgebra, LinearSolve, Random, RecursiveFactorization
using TriangularSolve

const SNLU = LinearSolve.SupernodalLU

machine_load() = Sys.islinux() ? first(split(read("/proc/loadavg", String))) : "unavailable"

function panel_matrix(np)
    W = Matrix{Float64}(I, np, np)
    scale = inv(sqrt(np))
    for j in 1:np
        for i in 1:(j - 1)
            W[i, j] = scale * randn()
        end
        for i in (j + 1):np
            W[i, j] = scale * randn()
        end
    end
    return W
end

function lower_times(W, Y0, np)
    kernel = @belapsed SNLU._unit_lower_solve!($W, Y, $np) setup = (Y = copy($Y0)) evals = 1
    triangularsolve = @belapsed TriangularSolve.ldiv!(
        UnitLowerTriangular(view($W, 1:$np, 1:$np)), Y, Val(false)
    ) setup = (Y = copy($Y0)) evals = 1
    blas = @belapsed SNLU._panel_unit_lower_trsm!($W, Y, $np) setup = (Y = copy($Y0)) evals = 1
    return kernel, triangularsolve, blas
end

function upper_times(W, Y0, np)
    kernel = @belapsed SNLU._upper_solve!($W, Y, $np) setup = (Y = copy($Y0)) evals = 1
    triangularsolve = @belapsed TriangularSolve.ldiv!(
        UpperTriangular(view($W, 1:$np, 1:$np)), Y, Val(false)
    ) setup = (Y = copy($Y0)) evals = 1
    blas = @belapsed SNLU._panel_upper_trsm!($W, Y, $np) setup = (Y = copy($Y0)) evals = 1
    return kernel, triangularsolve, blas
end

function benchmark_panel_cutoffs(; nps = (8, 16, 32, 64, 128, 256, 512, 768, 1024, 1280, 1536, 1792, 2048), nrhss = (1, 2, 4, 8, 16, 32))
    BLAS.set_num_threads(1)
    Random.seed!(1172)
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.25
    BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
    println("Julia $(VERSION), TriangularSolve $(Base.pkgversion(TriangularSolve)), BLAS threads $(BLAS.get_num_threads()), load $(machine_load())")
    println("np nrhs sweep kernel_ns triangularsolve_ns blas_ns")
    for np in nps, nrhs in nrhss
        W = panel_matrix(np)
        Y0 = randn(np, nrhs)
        for (sweep, times) in (("lower", lower_times(W, Y0, np)), ("upper", upper_times(W, Y0, np)))
            kernel, triangularsolve, blas = times
            println("$np $nrhs $sweep $(round(Int, kernel * 1.0e9)) $(round(Int, triangularsolve * 1.0e9)) $(round(Int, blas * 1.0e9))")
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    benchmark_panel_cutoffs()
end
