include(${METADEF_DIR}/cmake/build_type.cmake)

######### for base #############
set(GRAPH_BASE_SOURCE_LIST
    "types.cc"
    "anchor.cc"
    "ge_attr_value.cc"
    "any_value.cc"
    "attr_store.cc"
    "buffer.cc"
    "aligned_ptr.cc"
    "compute_graph.cc"
    "model.cc"
    "model_serialize.cc"
    "node.cc"
    "op_desc.cc"
    "ir_meta.cc"
    "ir_data_type_symbol_store.cc"
    "sym_dtype.cc"
    "ge_attr_define.cc"
    "ge_tensor.cc"
    "common/large_bm.cc"
    "common/plugin/plugin_manager.cc"
    "common/hyper_status.cc"
    "detail/attributes_holder.cc"
    "utils/anchor_utils.cc"
    "utils/graph_utils.cc"
    "utils/dumper/ge_graph_dumper.cc"
    "utils/trace/trace_manager.cc"
    "utils/ge_ir_utils.cc"
    "utils/node_utils.cc"
    "utils/type_utils.cc"
    "utils/tensor_utils.cc"
    "utils/constant_utils.cc"
    "utils/connection_matrix.cc"
    "utils/cycle_detector.cc"
    "utils/op_type_utils.cc"
    "option/ge_context.cc"
    "option/ge_local_context.cc"
    "utils/file_utils.cc"
    "serialization/attr_serializer.cc"
    "serialization/string_serializer.cc"
    "serialization/data_type_serializer.cc"
    "serialization/named_attrs_serializer.cc"
    "serialization/bool_serializer.cc"
    "serialization/buffer_serializer.cc"
    "serialization/float_serializer.cc"
    "serialization/int_serializer.cc"
    "serialization/tensor_serializer.cc"
    "serialization/tensor_desc_serializer.cc"
    "serialization/graph_serializer.cc"
    "serialization/list_value_serializer.cc"
    "serialization/list_list_int_serializer.cc"
    "serialization/list_list_float_serializer.cc"
    "serialization/attr_serializer_registry.cc"
    "serialization/utils/serialization_util.cc"
    "small_vector.cc"
    "compile_cache_policy/compile_cache_policy.cc"
    "cache_policy/cache_policy.cc"
    "cache_policy/compile_cache_desc.cc"
    "cache_policy/policy_register.cc"
    "cache_policy/cache_state.cc"
    "cache_policy/policy_management/match_policy/match_policy_exact_only.cc"
    "cache_policy/policy_management/match_policy/match_policy_for_exactly_the_same.cc"
    "cache_policy/policy_management/aging_policy/aging_policy_lru.cc"
    "cache_policy/policy_management/aging_policy/aging_policy_lru_k.cc"
    "profiler.cc"
    "tensor.cc"
)

set(DATA_FLOW_GRAPH_SOURCE_LIST
    "flow_graph/flow_attr_util.cc"
    "flow_graph/flow_graph.cc"
    "flow_graph/process_point.cc"
    "flow_graph/data_flow_attr_define.cc"
    "flow_graph/data_flow_utils.cc"
)

SET(GRAPH_SOURCE_LIST
    "ascend_string.cc"
    "attr_value.cc"
    "axis_type_info.cc"
    "operator.cc"
    "operator_factory.cc"
    "operator_factory_impl.cc"
    "operator_impl.cc"
    "graph.cc"
    "gnode.cc"
    "format_refiner.cc"
    "inference_context.cc"
    "ref_relation.cc"
    "resource_context_mgr.cc"
    "runtime_inference_context.cc"
    "shape_refiner.cc"
    "ir_definitions_recover.cc"
    "opsproto/opsproto_manager.cc"
    "utils/op_desc_utils.cc"
    "utils/tuning_utils.cc"
    "utils/ffts_graph_utils.cc"
    "utils/transformer_utils.cc"
    "utils/graph_utils_ex.cc"
    "utils/node_utils_ex.cc"
    "utils/op_desc_utils_ex.cc"
    "utils/enum_attr_utils.cc"
    "parallelism/tensor_parallel_attrs.cc"
    "utils/screen_printer.cc"
    "${METADEF_DIR}/third_party/transformer/src/axis_util.cc"
    "${METADEF_DIR}/third_party/transformer/src/transfer_shape_according_to_format.cc"
    "${METADEF_DIR}/third_party/transformer/src/expand_dimension.cc"
    "${METADEF_DIR}/third_party/transformer/src/transfer_range_according_to_format.cc"
    "${METADEF_DIR}/third_party/transformer/src/transfer_shape_utils.cc"
    "${METADEF_DIR}/ops/op_imp.cpp"
)

######### libgraph_base.so #############
add_library(graph_base SHARED
    ${GRAPH_BASE_SOURCE_LIST}
    $<TARGET_OBJECTS:metadef_graph_protos_obj>
)

target_compile_options(graph_base PRIVATE
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>)

target_compile_definitions(graph_base PRIVATE
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:FMK_SUPPORT_DUMP>
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    google=ascend_private
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
)

target_include_directories(graph_base PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
    ${METADEF_DIR}
)

target_link_options(graph_base PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(graph_base
    PRIVATE
        intf_pub
        static_mmpa
        -Wl,--no-as-needed
        c_sec
        slog
        json
        -Wl,--as-needed
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
    PUBLIC
        metadef_headers
)

target_link_libraries(graph_base PRIVATE ascend_protobuf error_manager)
target_compile_options(graph_base PRIVATE -O2 -Werror)

######### libgraph.so #############
add_library(graph SHARED
    ${GRAPH_SOURCE_LIST}
)

target_compile_options(graph PRIVATE
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>)

target_compile_definitions(graph PRIVATE
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:FMK_SUPPORT_DUMP>
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    google=ascend_private
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
)

target_include_directories(graph PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${METADEF_DIR}
)

target_link_options(graph PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(graph
    PRIVATE
        intf_pub
        static_mmpa
        -Wl,--no-as-needed
        graph_base
        c_sec
        slog
        json
        platform
        -Wl,--as-needed
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
    PUBLIC
        metadef_headers
)

target_link_libraries(graph PRIVATE graph_base error_manager)
target_compile_options(graph PRIVATE -O2 -Werror)

######### libflow_graph.so #############
add_library(flow_graph SHARED
    ${DATA_FLOW_GRAPH_SOURCE_LIST}
    $<TARGET_OBJECTS:metadef_flow_graph_protos_obj>
)

target_compile_options(flow_graph PRIVATE
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>)

target_compile_definitions(flow_graph PRIVATE
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:FMK_SUPPORT_DUMP>
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    google=ascend_private
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
)

target_include_directories(flow_graph PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/metadef_data_flow_protos
    ${METADEF_DIR}
)

target_link_options(flow_graph PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(flow_graph
    PRIVATE
        intf_pub
        static_mmpa
        -Wl,--no-as-needed
        graph_base
        graph
        c_sec
        slog
        json
        -Wl,--as-needed
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
    PUBLIC
        metadef_headers
)

target_link_libraries(flow_graph PRIVATE ascend_protobuf error_manager)
target_compile_options(flow_graph PRIVATE -O2)

if (${ENABLE_OPEN_SRC} STREQUAL "True")
else()
    ######### libgraph.a #############
    add_library(graph_share SHARED
            ${GRAPH_BASE_SOURCE_LIST}
            ${GRAPH_SOURCE_LIST}
            $<TARGET_OBJECTS:metadef_graph_protos_obj>
            )

    target_compile_options(graph_share PRIVATE
            $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
            $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
            $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
            $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>)

    target_compile_definitions(graph_share PRIVATE
            $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:FMK_SUPPORT_DUMP>
            $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
            google=ascend_private
            $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
            $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
            )

    target_include_directories(graph_share PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}
            ${CMAKE_BINARY_DIR}
            ${CMAKE_BINARY_DIR}/proto/metadef_protos
            ${METADEF_DIR}
            )

    target_link_options(graph_share PRIVATE
            -Wl,-Bsymbolic
            )

    target_link_libraries(graph_share
            PRIVATE
            intf_pub
            static_mmpa
            -Wl,--no-as-needed
            c_sec
            slog
            json
            platform
            -Wl,--as-needed
            $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
            -ldl
            PUBLIC
            metadef_headers
            )

    target_clone(graph_share graph_static STATIC)

    target_link_libraries(graph_share PRIVATE ascend_protobuf error_manager)
    target_link_libraries(graph_static PRIVATE ascend_protobuf_static)

    target_compile_options(graph_share PRIVATE -O2)
    target_compile_options(graph_static PRIVATE $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal>)

    set_target_properties(graph_static PROPERTIES
            WINDOWS_EXPORT_ALL_SYMBOLS TRUE
            OUTPUT_NAME $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,libgraph_share,graph_share>
            )
endif()

##############################################################
set(STUB_SRC_LIST
    ${CMAKE_CURRENT_BINARY_DIR}/stub_ascend_string.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_attr_value.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_gnode.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_graph.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_inference_context.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_operator.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_operator_factory.cc
    ${CMAKE_CURRENT_BINARY_DIR}/stub_tensor.cc
)

add_custom_command(
    OUTPUT ${STUB_SRC_LIST}
    COMMAND echo "Generating stub files."
            && ${HI_PYTHON} ${CMAKE_CURRENT_LIST_DIR}/stub/gen_stubapi.py ${METADEF_DIR}/inc/external ${CMAKE_CURRENT_BINARY_DIR}
            && echo "Generating stub files end."
)

add_custom_target(graph_stub
    DEPENDS ${STUB_SRC_LIST}
            ${METADEF_DIR}/tests/stub/operator_stub.cc
)

#############################################################

############ stub/libgraph.so ############
add_library(atc_stub_graph SHARED
    ${STUB_SRC_LIST}
    ${METADEF_DIR}/tests/stub/operator_stub.cc
)

add_dependencies(atc_stub_graph graph_stub)

target_include_directories(atc_stub_graph PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${METADEF_DIR}
)

target_compile_options(atc_stub_graph PRIVATE
    -Wfloat-equal
    -fno-common
)

target_link_libraries(atc_stub_graph
    PRIVATE
        intf_pub
        static_mmpa
        -Wl,--no-as-needed
        json
        platform
        -Wl,--as-needed
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
    PUBLIC
        metadef_headers
)

set_target_properties(atc_stub_graph PROPERTIES
    OUTPUT_NAME graph
    LIBRARY_OUTPUT_DIRECTORY atc_stub
)

############ fwk_stub/libgraph.so ############
add_library(fwk_stub_graph SHARED
    ${STUB_SRC_LIST}
    ${METADEF_DIR}/tests/stub/operator_stub.cc
)

add_dependencies(fwk_stub_graph graph_stub)

target_compile_options(fwk_stub_graph PRIVATE
    -Wfloat-equal
    -fno-common
)

target_include_directories(fwk_stub_graph PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${METADEF_DIR}
)

target_link_libraries(fwk_stub_graph
    PRIVATE
        intf_pub
        static_mmpa
        -Wl,--no-as-needed
        json
        platform
        -Wl,--as-needed
        $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
        -ldl
    PUBLIC
        metadef_headers
)

set_target_properties(fwk_stub_graph PROPERTIES
    OUTPUT_NAME graph
    LIBRARY_OUTPUT_DIRECTORY fwk_stub
)

############ install ############
install(TARGETS atc_stub_graph OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/${CMAKE_SYSTEM_PROCESSOR}/stub
)

install(TARGETS fwk_stub_graph OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/${CMAKE_SYSTEM_PROCESSOR}/fwk_stub
)
