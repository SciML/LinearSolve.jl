module LinearSolveSLATEExt

using LinearSolve: LinearSolve
using SLATE_jll: SLATE_jll

# SLATE_jll only builds for some platforms, so the path is registered only where the
# artifact actually exists. `_slate_library_candidates` puts this after any explicit
# `libpath` or `ENV` setting, so loading the JLL never overrides a hand-built SLATE.
function __init__()
    if SLATE_jll.is_available()
        LinearSolve._SLATE_JLL_LIBPATH[] = SLATE_jll.libslate_lapack_api
    end
    return nothing
end

end
