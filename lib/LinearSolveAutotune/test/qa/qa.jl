using LinearSolveAutotune, Aqua, JET

Aqua.test_all(LinearSolveAutotune)
JET.test_package(LinearSolveAutotune; target_defined_modules = true)
