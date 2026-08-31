#include <jlcxx/jlcxx.hpp>
#include <iostream>

void dummy_solve() {
    std::cout << "Amesos2 dummy solve hit\n";
}

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
    mod.method("dummy_solve", &dummy_solve);
}
