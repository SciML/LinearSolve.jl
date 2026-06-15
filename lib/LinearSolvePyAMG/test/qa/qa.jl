using LinearSolvePyAMG, Aqua, JET

Aqua.test_all(LinearSolvePyAMG)
JET.test_package(LinearSolvePyAMG; target_defined_modules = true)
