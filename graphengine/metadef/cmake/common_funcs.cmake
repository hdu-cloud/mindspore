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

function(to_absolute_path input_sources source_dir out_arg)
    set(output_sources)
    FOREACH(source_file ${${input_sources}})
        if(IS_ABSOLUTE ${source_file})
            list(APPEND output_sources ${source_file})
        else()
            if(${source_file} MATCHES ".cc$")
                list(APPEND output_sources ${${source_dir}}/${source_file})
            else()
                list(APPEND output_sources ${source_file})
            endif()
        endif()
    ENDFOREACH()
    set(${out_arg} ${output_sources} PARENT_SCOPE)
endfunction()


function(target_clone original_library target_library libray_type)
    get_target_property(sourceFiles ${original_library} SOURCES)
    get_target_property(sourceDir ${original_library} SOURCE_DIR)
    get_target_property(linkLibs ${original_library} LINK_LIBRARIES)
    get_target_property(linkOpts ${original_library} LINK_OPTIONS)
    get_target_property(includeDirs ${original_library} INCLUDE_DIRECTORIES)
    get_target_property(compileDefinitions ${original_library} COMPILE_DEFINITIONS)
    get_target_property(compileOptions ${original_library} COMPILE_OPTIONS)

    to_absolute_path(sourceFiles sourceDir absolute_sources_files)

    add_library(${target_library} ${libray_type}
            ${absolute_sources_files}
            )

    target_include_directories(${target_library} PRIVATE
            ${includeDirs}
            )

    target_link_libraries(${target_library} PRIVATE
            ${linkLibs}
            )

    if (linkOpts)
        target_link_options(${target_library} PRIVATE
                ${linkOpts}
                )
    endif()
    if (compileOptions)
        target_compile_options(${target_library} PRIVATE
                ${compileOptions}
                )
    endif()
    if (compileDefinitions)
        target_compile_definitions(${target_library} PRIVATE
                ${compileDefinitions}
                )
    endif()
endfunction()