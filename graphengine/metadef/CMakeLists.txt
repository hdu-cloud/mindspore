cmake_minimum_required(VERSION 3.14)
project (MetaDef[CXX])

set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)
set(INSTALL_RUNTIME_DIR bin)
set(INSTALL_INCLUDE_DIR include)
set(INSTALL_CONFIG_DIR  cmake)
include(CMakePackageConfigHelpers)

set(METADEF_DIR ${CMAKE_CURRENT_LIST_DIR})

option(ENABLE_OPEN_SRC "Enable graphengine compile in opensource." FALSE)

if (ENABLE_OPEN_SRC)
    if (DEFINED ENV{D_PKG_SERVER})
        set(METADEF_PB_PKG $ENV{D_PKG_SERVER})
    endif()
    set(HI_PYTHON python3)
    include(cmake/external_libs/protobuf_shared.cmake)
    include(cmake/external_libs/protobuf_static.cmake)
    include(cmake/external_libs/protoc.cmake)
    include(cmake/external_libs/securec.cmake)
    include(cmake/external_libs/json.cmake)
    include(cmake/FindModule.cmake)
    include(cmake/intf_pub_linux.cmake)
    include(cmake/external_libs/gtest.cmake)
    include(cmake/external_libs/gbenchmark.cmake)
    include(cmake/test_funcs.cmake)
    if(METADEF_LIB_DIR)
        find_module(slog libascendalog.so ${METADEF_LIB_DIR})
        find_module(static_mmpa libmmpa.a ${METADEF_LIB_DIR})
        unset(METADEF_LIB_DIR CACHE)
    endif()
endif()

include(cmake/common_funcs.cmake)
add_subdirectory(inc)
add_subdirectory(proto)
add_subdirectory(third_party)
add_subdirectory(graph)
add_subdirectory(exe_graph)
add_subdirectory(error_manager)
add_subdirectory(register)
if (ENABLE_OPEN_SRC)
    add_subdirectory(tests)
endif()

install(TARGETS exe_graph error_manager graph register metadef_headers
    EXPORT metadef-targets
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL COMPONENT opensdk
    ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR} OPTIONAL COMPONENT opensdk
    RUNTIME DESTINATION ${INSTALL_RUNTIME_DIR} OPTIONAL COMPONENT opensdk
)
install(FILES   ${METADEF_DIR}/third_party/transformer/inc/axis_util.h
                ${METADEF_DIR}/third_party/transformer/inc/expand_dimension.h
                ${METADEF_DIR}/third_party/transformer/inc/transfer_range_according_to_format.h
                ${METADEF_DIR}/third_party/transformer/inc/transfer_shape_according_to_format.h
        DESTINATION ${INSTALL_INCLUDE_DIR}/metadef/transformer COMPONENT opensdk EXCLUDE_FROM_ALL
)
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
