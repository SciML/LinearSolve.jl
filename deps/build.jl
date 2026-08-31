using CxxWrap, Trilinos_jll
cxx_inc = joinpath(dirname(dirname(pathof(CxxWrap))), "deps", "usr", "include")
trilinos_inc = joinpath(Trilinos_jll.artifact_dir, "include")
julia_inc = joinpath(Sys.BINDIR, "..", "include", "julia")
run(`clang++ -shared -fPIC -std=c++17 -undefined dynamic_lookup -I$cxx_inc -I$trilinos_inc -I$julia_inc deps/amesos2_wrap.cpp -o deps/libamesos2_wrap.dylib`)
