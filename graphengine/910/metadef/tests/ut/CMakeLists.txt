if (ENABLE_METADEF_COV)
    set(COVERAGE_COMPILER_FLAGS "-g --coverage -fprofile-arcs -fPIC -O0 -ftest-coverage")
    set(CMAKE_CXX_FLAGS "${COVERAGE_COMPILER_FLAGS}")
endif()

add_subdirectory(graph)
add_subdirectory(register)
add_subdirectory(error_manager)
add_subdirectory(exe_graph)