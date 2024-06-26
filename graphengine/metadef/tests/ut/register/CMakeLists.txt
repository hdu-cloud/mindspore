# Copyright 2019-2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

project(ut_register)

set(CMAKE_CXX_STANDARD 11)
# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos/proto)
include_directories(${METADEF_DIR})
include_directories(${METADEF_DIR}/register)
include_directories(${METADEF_DIR}/third_party/transformer/inc)

file(GLOB_RECURSE REGISTER_UT_FILES CONFIGURE_DEPENDS "${METADEF_DIR}/tests/ut/register/*.cc" )
add_executable(ut_register
    ${REGISTER_UT_FILES}
)

target_compile_options(ut_register PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
)

target_compile_definitions(ut_register PRIVATE
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_link_libraries(ut_register
    $<BUILD_INTERFACE:intf_pub>
    metadef_headers
    fwk_mmpa_headers
    graphengine_headers
    register graph exe_graph
    gtest
    gtest_main
    ascend_protobuf
    slog_stub
    json
    c_sec
    error_manager
    -lrt
    -ldl
    -lgcov
)
