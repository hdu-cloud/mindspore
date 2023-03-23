# Copyright 2022 Huawei Technologies Co., Ltd
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
function(target_stub_lib target_name lib_name stub_name)
    if (TARGET ${target_name})
        get_target_property(linkLibs ${target_name} LINK_LIBRARIES)
        list(REMOVE_ITEM linkLibs ${lib_name})
        list(APPEND linkLibs ${stub_name})
        message("replace libs ${linkLibs}")
        set_target_properties(${target_name} PROPERTIES
                LINK_LIBRARIES "${linkLibs}"
                )
    endif()
endfunction()

function(stub_module module stub_name)
    if (TARGET ${module})
        return()
    endif()
    add_library(${module} INTERFACE)
    target_link_libraries(${module} INTERFACE ${stub_name})
endfunction()

function(enable_gcov module)
    if (TARGET ${module})
        target_compile_options(${module} PRIVATE
                --coverage -fprofile-arcs -fPIC -ftest-coverage
                -Werror=format
                )
        target_link_libraries(${module} PUBLIC -lgcov)
    endif()
endfunction()
