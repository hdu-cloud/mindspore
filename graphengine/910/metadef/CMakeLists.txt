cmake_minimum_required(VERSION 3.14)
project (MetaDef[CXX])
if (NOT DEFINED CMAKE_MODULE_PATH)
    set(CMAKE_MODULE_PATH ${ASCEND_OPENSDK_DIR}/cmake/modules)
endif()

if (NOT DEFINED CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH ${ASCEND_OPENSDK_DIR}/cmake;
            ${ASCEND_OPENSDK_DIR}/c_sec;
            ${ASCEND_OPENSDK_DIR}/json;
            ${ASCEND_OPENSDK_DIR}/openssl;
            ${ASCEND_OPENSDK_DIR}/zlib;
            ${ASCEND_OPENSDK_DIR}/protoc;
            ${ASCEND_OPENSDK_DIR}/protoc_grpc;
            ${ASCEND_OPENSDK_DIR}/grpc;
            ${ASCEND_OPENSDK_DIR}/protobuf_static;
            ${ASCEND_OPENSDK_DIR}/ascend_protobuf;
            ${ASCEND_OPENSDK_DIR}/ascend_protobuf_static;
            ${ASCEND_OPENSDK_DIR}/gtest_shared/lib/cmake/GTest;
            ${ASCEND_OPENSDK_DIR}/gtest_shared/lib64/cmake/GTest)
endif()

if (NOT DEFINED protoc_ROOT)
    set(protoc_ROOT ${ASCEND_OPENSDK_DIR}/protoc)
endif()

if (NOT DEFINED protobuf_grpc_ROOT)
    set(protobuf_grpc_ROOT ${ASCEND_OPENSDK_DIR}/grpc)
endif()

if (NOT DEFINED protobuf_static_ROOT)
    set(protobuf_static_ROOT ${ASCEND_OPENSDK_DIR}/protobuf_static)
endif()

if (NOT DEFINED ascend_protobuf_shared_ROOT)
    set(ascend_protobuf_shared_ROOT ${ASCEND_OPENSDK_DIR}/ascend_protobuf)
endif()

if (NOT DEFINED ascend_protobuf_static_ROOT)
    set(ascend_protobuf_static_ROOT ${ASCEND_OPENSDK_DIR}/ascend_protobuf_static)
endif()

include(CMakePrintHelpers)
message(STATUS "Variables in metadef project :")
cmake_print_variables(ASCEND_OPENSDK_DIR)
cmake_print_variables(CMAKE_BUILD_TYPE)
cmake_print_variables(CMAKE_INSTALL_PREFIX)
cmake_print_variables(CMAKE_PREFIX_PATH)
cmake_print_variables(CMAKE_MODULE_PATH)
cmake_print_variables(protoc_ROOT)
cmake_print_variables(protobuf_grpc_ROOT)
cmake_print_variables(protobuf_static_ROOT)
cmake_print_variables(ascend_protobuf_shared_ROOT)
cmake_print_variables(ascend_protobuf_static_ROOT)
cmake_print_variables(ENABLE_OPEN_SRC ENABLE_GITEE ENABLE_METADEF_UT ENABLE_METADEF_ST ENABLE_METADEF_COV ENABLE_BENCHMARK GE_ONLY)

set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)
set(INSTALL_RUNTIME_DIR bin)
set(INSTALL_INCLUDE_DIR include)
set(INSTALL_CONFIG_DIR  cmake)
include(CMakePackageConfigHelpers)

set(METADEF_DIR ${CMAKE_CURRENT_LIST_DIR})

option(ENABLE_OPEN_SRC "Enable graphengine compile in opensource." FALSE)

if (ENABLE_OPEN_SRC)
    set(HI_PYTHON python3)
    include(cmake/test_funcs.cmake)    
    # 开源软件包
    find_package(json MODULE)
    find_package(protoc MODULE)
    find_package(protobuf_static MODULE)
    find_package(ascend_protobuf_shared MODULE)
    find_package(ascend_protobuf_static MODULE)
    find_package(GTest CONFIG REQUIRED)

    # 自研软件包
    find_package(securec MODULE REQUIRED)
    find_package(slog CONFIG REQUIRED)
    if (ENABLE_METADEF_UT OR ENABLE_METADEF_ST)
        add_library(mmpa_headers INTERFACE)
        target_include_directories(mmpa_headers INTERFACE
                ${ASCEND_OPENSDK_DIR}/include
                ${ASCEND_OPENSDK_DIR}/include/mmpa
                ${ASCEND_OPENSDK_DIR}/include/sub_inc
                )
    else()
        find_package(mmpa CONFIG REQUIRED)
    endif()
    find_package(cce CONFIG REQUIRED)
    find_package(msprof CONFIG REQUIRED)
    find_package(runtime CONFIG REQUIRED)
    find_package(platform CONFIG REQUIRED)
    if (BUILD_WITHOUT_AIR)
        find_package(air CONFIG REQUIRED)
    endif()

    include(cmake/intf_pub_linux.cmake)

endif()

include(cmake/common_funcs.cmake)
add_subdirectory(inc)
add_subdirectory(proto)
add_subdirectory(graph)
add_subdirectory(exe_graph)
add_subdirectory(error_manager)
add_subdirectory(register)
if (ENABLE_METADEF_UT OR ENABLE_METADEF_ST)
    include(cmake/external_libs/gbenchmark.cmake)
    add_subdirectory(tests)
endif()

install(TARGETS exe_graph error_manager graph graph_base flow_graph register rt2_registry_static metadef_headers
    EXPORT metadef-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL COMPONENT opensdk
    ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL COMPONENT opensdk
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR} OPTIONAL COMPONENT opensdk
)

# 下列头文件发布是非法的，需要在后续整改中删掉
# --------------------start------------------------
install(FILES   ${METADEF_DIR}/third_party/transformer/inc/axis_util.h
                ${METADEF_DIR}/third_party/transformer/inc/expand_dimension.h
                ${METADEF_DIR}/third_party/transformer/inc/transfer_shape_utils.h
                ${METADEF_DIR}/third_party/transformer/inc/transfer_range_according_to_format.h
                ${METADEF_DIR}/third_party/transformer/inc/transfer_shape_according_to_format.h
        DESTINATION ${INSTALL_INCLUDE_DIR}/metadef/transformer COMPONENT opensdk EXCLUDE_FROM_ALL
)
install(FILES   ${METADEF_DIR}/register/op_tiling/op_tiling_constants.h
                ${METADEF_DIR}/register/op_tiling/op_compile_info_manager.h
                ${METADEF_DIR}/register/op_tiling/op_tiling_utils.h
        DESTINATION ${INSTALL_INCLUDE_DIR}/metadef/register/op_tiling COMPONENT opensdk EXCLUDE_FROM_ALL
)
install(FILES   ${METADEF_DIR}/graph/operator_impl.h
                ${METADEF_DIR}/graph/op_io.h
        DESTINATION ${INSTALL_INCLUDE_DIR}/metadef/graph COMPONENT opensdk EXCLUDE_FROM_ALL
)
install(FILES   ${METADEF_DIR}/graph/debug/ge_log.h
                ${METADEF_DIR}/graph/debug/ge_util.h # 被air使用
                ${METADEF_DIR}/graph/debug/ge_op_types.h # 被parser使用
        DESTINATION ${INSTALL_INCLUDE_DIR}/metadef/graph/debug COMPONENT opensdk EXCLUDE_FROM_ALL
)
install(FILES   ${METADEF_DIR}/graph/utils/dumper/ge_graph_dumper.h
        DESTINATION ${INSTALL_INCLUDE_DIR}/metadef/graph/utils/dumper COMPONENT opensdk EXCLUDE_FROM_ALL
)
# ---------------------end-------------------------

install(DIRECTORY ${METADEF_DIR}/inc/ DESTINATION ${INSTALL_INCLUDE_DIR}/metadef
    COMPONENT opensdk EXCLUDE_FROM_ALL FILES_MATCHING PATTERN "*.h"
)
install(DIRECTORY ${METADEF_DIR}/proto/ DESTINATION proto
    COMPONENT opensdk EXCLUDE_FROM_ALL FILES_MATCHING PATTERN "*.proto"
)

if (PACKAGE STREQUAL "opensdk")
    install(EXPORT metadef-targets DESTINATION ${INSTALL_CONFIG_DIR}
        FILE metadef-targets.cmake COMPONENT opensdk EXCLUDE_FROM_ALL
    )
    set(PKG_NAME metadef)
    configure_package_config_file(${TOP_DIR}/cmake/config/pkg_config_template.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/metadef-config.cmake
        INSTALL_DESTINATION ${INSTALL_CONFIG_DIR}
        PATH_VARS INSTALL_BASE_DIR INSTALL_INCLUDE_DIR INSTALL_LIBRARY_DIR INSTALL_RUNTIME_DIR INSTALL_CONFIG_DIR
        INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
    )
    unset(PKG_NAME)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/metadef-config.cmake
        DESTINATION ${INSTALL_CONFIG_DIR} COMPONENT opensdk EXCLUDE_FROM_ALL
    )
endif()
