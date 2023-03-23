project(ExeGraphUT)

file(GLOB_RECURSE TEST_SRCS CONFIGURE_DEPENDS "${METADEF_DIR}/tests/ut/exe_graph/*.cc")
file(GLOB_RECURSE FAKER_SRCS CONFIGURE_DEPENDS "${METADEF_DIR}/tests/depends/faker/*.cc")

add_executable(ut_exe_graph ${TEST_SRCS} ${FAKER_SRCS})

target_include_directories(ut_exe_graph PRIVATE
        ${METADEF_DIR}/tests/depends
        ${METADEF_DIR}/third_party/
        ${METADEF_DIR}/exe_graph
        ${METADEF_DIR}/third_party/graphengine/inc/
        ${METADEF_DIR}/third_party/graphengine/inc/external
        ${METADEF_DIR}/third_party/transformer/inc
        ${METADEF_DIR}/third_party/fwkacllib/inc
        ${CMAKE_BINARY_DIR}
        ${CMAKE_BINARY_DIR}/proto/metadef_protos
        )

target_compile_options(ut_exe_graph PRIVATE
        -g --coverage -fprofile-arcs -ftest-coverage
        )

target_compile_definitions(ut_exe_graph PRIVATE
        google=ascend_private
        )

target_link_libraries(ut_exe_graph PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        exe_graph error_manager
        gtest gtest_main slog_stub ascend_protobuf c_sec mmpa_stub -lrt -ldl -lgcov
        metadef_headers
        fwk_mmpa_headers
        graphengine_headers
        graph
        )
