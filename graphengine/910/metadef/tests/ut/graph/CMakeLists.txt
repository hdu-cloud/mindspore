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
set(CMAKE_CXX_STANDARD 11)

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${METADEF_DIR}/inc/common/util/trace_manager)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos/proto)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_data_flow_protos)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_data_flow_protos/proto)
include_directories(${METADEF_DIR})
include_directories(${METADEF_DIR}/graph)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CMAKE_BINARY_DIR}/proto/ge/proto)

file(GLOB_RECURSE UT_FILES CONFIGURE_DEPENDS "${METADEF_DIR}/tests/ut/graph/testcase/*.cc")
file(GLOB_RECURSE FAKE_FILES CONFIGURE_DEPENDS "${METADEF_DIR}/tests/depends/cache_desc_stub/runtime_cache_desc.cc")
add_executable(ut_graph ${UT_FILES} ${FAKE_FILES})

target_compile_options(ut_graph PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
)

target_compile_definitions(ut_graph PRIVATE
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    google=ascend_private
    FUNC_VISIBILITY
)

target_link_libraries(ut_graph PRIVATE
    intf_pub -lgcov
    -Wl,--no-as-needed
    platform_stub
    slog_headers
    metadef_headers
    graph graph_base flow_graph error_manager mmpa
    GTest::gtest GTest::gtest_main ascend_protobuf slog_stub c_sec json mmpa_stub -lrt -ldl
)

target_include_directories(ut_graph PRIVATE ${METADEF_DIR}/tests/depends)