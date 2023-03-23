project(ExeGraph)

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
    ${METADEF_DIR}/third_party/transformer/inc
)

target_link_libraries(exe_graph PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:slog_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:mmpa_headers>>
    metadef_headers
    fwk_mmpa_headers
    graphengine_headers
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
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)

target_clone(exe_graph exe_graph_static STATIC)

target_link_libraries(exe_graph PRIVATE ascend_protobuf graph)
target_link_libraries(exe_graph_static PRIVATE
        ascend_protobuf_static graph_static
        )

target_compile_options(exe_graph PRIVATE -O2)
target_compile_options(exe_graph_static PRIVATE $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal>)

