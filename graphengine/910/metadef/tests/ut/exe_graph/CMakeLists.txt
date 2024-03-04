file(GLOB_RECURSE TEST_SRCS CONFIGURE_DEPENDS "${METADEF_DIR}/tests/ut/exe_graph/*.cc")
file(GLOB_RECURSE FAKER_SRCS CONFIGURE_DEPENDS "${METADEF_DIR}/tests/depends/faker/*.cc")
file(GLOB_RECURSE GE_RUNTIME_STUB_SRCS CONFIGURE_DEPENDS "${METADEF_DIR}/tests/depends/ge_runtime_stub/*.cc")

add_executable(ut_exe_graph ${TEST_SRCS} ${FAKER_SRCS} ${GE_RUNTIME_STUB_SRCS})

target_include_directories(ut_exe_graph PRIVATE
        ${METADEF_DIR}
        ${METADEF_DIR}/tests/depends
        ${METADEF_DIR}/exe_graph
        ${CMAKE_BINARY_DIR}
        ${CMAKE_BINARY_DIR}/proto/metadef_protos
        )

target_compile_options(ut_exe_graph PRIVATE
        -g --coverage -fprofile-arcs -ftest-coverage
        )

target_compile_definitions(ut_exe_graph PRIVATE
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
        google=ascend_private
        )

target_link_libraries(ut_exe_graph PRIVATE
        intf_pub
        -Wl,--no-as-needed
        slog_headers
        platform_stub
        msprof_headers
        runtime_headers
        exe_graph register error_manager
        mmpa
        GTest::gtest GTest::gtest_main slog_stub ascend_protobuf c_sec mmpa_stub -lrt -ldl -lgcov
        metadef_headers
        graph
        graph_base
        -Wl,--as-needed
        )
