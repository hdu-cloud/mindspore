include(${METADEF_DIR}/cmake/build_type.cmake)
file(GLOB_RECURSE EXE_GRAPH_SRCS CONFIGURE_DEPENDS "${METADEF_DIR}/exe_graph/*.cc")

add_library(exe_graph SHARED ${EXE_GRAPH_SRCS})

target_include_directories(exe_graph PRIVATE
    ${METADEF_DIR}
)

target_include_directories(exe_graph PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
)

target_link_libraries(exe_graph PRIVATE
    intf_pub
    mmpa_headers
    metadef_headers
    c_sec
    slog
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
)

target_compile_definitions(exe_graph PRIVATE
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
)

target_compile_options(exe_graph PRIVATE
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal -fno-common>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)

if (${ENABLE_OPEN_SRC} STREQUAL "True")
    target_link_libraries(exe_graph PRIVATE ascend_protobuf graph register)
    target_compile_options(exe_graph PRIVATE -O2)
else()
    target_clone(exe_graph exe_graph_static STATIC)

    target_link_libraries(exe_graph PRIVATE ascend_protobuf graph register)
    target_link_libraries(exe_graph_static PRIVATE
            ascend_protobuf_static graph_static register_static
            )

    target_compile_options(exe_graph PRIVATE -O2)
    target_compile_options(exe_graph_static PRIVATE $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal>)
endif()

##############################################################
set(STUB_SRC_LIST
    ${CMAKE_CURRENT_BINARY_DIR}/stub_compute_node_info.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_runtime_attrs.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_tiling_data.cc
)

add_custom_command(
    OUTPUT ${STUB_SRC_LIST}
    COMMAND echo "Generating stub files."
            && ${HI_PYTHON} ${CMAKE_CURRENT_LIST_DIR}/stub/gen_stubapi.py ${METADEF_DIR}/inc/external ${CMAKE_CURRENT_BINARY_DIR}
            && echo "Generating stub files end."
)

add_custom_target(exe_graph_stub DEPENDS ${STUB_SRC_LIST})

############ stub/libexe_graph.so ############
add_library(stub_exe_graph SHARED ${STUB_SRC_LIST})

add_dependencies(stub_exe_graph exe_graph_stub)

target_include_directories(stub_exe_graph PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
)

target_compile_options(stub_exe_graph PRIVATE
    -Wfloat-equal
    -fno-common
)

target_link_libraries(stub_exe_graph
    PRIVATE
        intf_pub
        air_headers
        c_sec
    PUBLIC
        metadef_headers
)

set_target_properties(stub_exe_graph PROPERTIES
    OUTPUT_NAME exe_graph
    LIBRARY_OUTPUT_DIRECTORY stub
)

############ install ############
install(TARGETS stub_exe_graph OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/${CMAKE_SYSTEM_PROCESSOR}/stub
)
