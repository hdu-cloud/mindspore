include(${METADEF_DIR}/cmake/build_type.cmake)

set(GRAPH_SOURCE_LIST
    "types.cc"
    "anchor.cc"
    "ge_attr_value.cc"
    "any_value.cc"
    "attr_store.cc"
    "attr_value.cc"
    "buffer.cc"
    "aligned_ptr.cc"
    "compute_graph.cc"
    "ascend_string.cc"
    "axis_type_info.cc"
    "gnode.cc"
    "graph.cc"
    "inference_context.cc"
    "shape_refiner.cc"
    "format_refiner.cc"
    "ref_relation.cc"
    "model.cc"
    "model_serialize.cc"
    "node.cc"
    "op_desc.cc"
    "ir_meta.cc"
    "ir_data_type_symbol_store.cc"
    "operator.cc"
    "operator_factory.cc"
    "operator_factory_impl.cc"
    "ge_attr_define.cc"
    "ge_tensor.cc"
    "common/large_bm.cc"
    "common/plugin_so_manager.cc"
    "common/hyper_status.cc"
    "detail/attributes_holder.cc"
    "utils/anchor_utils.cc"
    "utils/tuning_utils.cc"
    "utils/graph_utils.cc"
    "utils/ffts_graph_utils.cc"
    "utils/dumper/ge_graph_dumper.cc"
    "utils/trace/trace_manager.cc"
    "utils/ge_ir_utils.cc"
    "utils/node_utils.cc"
    "utils/op_desc_utils.cc"
    "utils/type_utils.cc"
    "utils/tensor_utils.cc"
    "utils/constant_utils.cc"
    "utils/connection_matrix.cc"
    "utils/cycle_detector.cc"
    "tensor.cc"
    "debug/graph_debug.cc"
    "opsproto/opsproto_manager.cc"
    "${METADEF_DIR}/ops/op_imp.cpp"
    "option/ge_context.cc"
    "option/ge_local_context.cc"
    "runtime_inference_context.cc"
    "${METADEF_DIR}/third_party/transformer/src/axis_util.cc"
    "${METADEF_DIR}/third_party/transformer/src/transfer_shape_according_to_format.cc"
    "${METADEF_DIR}/third_party/transformer/src/expand_dimension.cc"
    "${METADEF_DIR}/third_party/transformer/src/transfer_range_according_to_format.cc"
    "${METADEF_DIR}/third_party/transformer/src/transfer_shape_utils.cc"
    "utils/transformer_utils.cc"
    "utils/file_utils.cc"
    "resource_context_mgr.cc"
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
    "small_vector.cc"
    "operator_impl.cc"
    "compile_cache_policy/compile_cache_policy.cc"
    "compile_cache_policy/compile_cache_desc.cc"
    "compile_cache_policy/policy_register.cc"
    "compile_cache_policy/compile_cache_state.cc"
    "compile_cache_policy/compile_cache_hasher.cc"
    "compile_cache_policy/policy_management/match_policy/match_policy_exact_only.cc"
    "compile_cache_policy/policy_management/aging_policy/aging_policy_lru.cc"
    "profiler.cc"
)

######### libgraph.so #############
add_library(graph SHARED
    ${GRAPH_SOURCE_LIST}
    $<TARGET_OBJECTS:metadef_graph_protos_obj>
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
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
    ${METADEF_DIR}
    ${METADEF_DIR}/third_party/transformer/inc
)

target_link_options(graph PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(graph
    PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:slog_headers>>
        $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:mmpa_headers>>
        fwk_mmpa_headers
        graphengine_headers
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

target_clone(graph graph_static STATIC)

target_link_libraries(graph PRIVATE ascend_protobuf error_manager)
target_link_libraries(graph_static PRIVATE ascend_protobuf_static)

target_compile_options(graph PRIVATE -O2)
target_compile_options(graph_static PRIVATE $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal>)


set_target_properties(graph_static PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS TRUE
    OUTPUT_NAME $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,libgraph,graph>
    )

##############################################################
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stub_attr_value.cc
           ${CMAKE_CURRENT_BINARY_DIR}/stub_graph.cc
           ${CMAKE_CURRENT_BINARY_DIR}/stub_operator.cc
           ${CMAKE_CURRENT_BINARY_DIR}/stub_operator_factory.cc
           ${CMAKE_CURRENT_BINARY_DIR}/stub_tensor.cc
           ${CMAKE_CURRENT_BINARY_DIR}/stub_inference_context.cc
           ${CMAKE_CURRENT_BINARY_DIR}/stub_ascend_string.cc
           ${CMAKE_CURRENT_BINARY_DIR}/stub_gnode.cc
    COMMAND echo "Generating stub files."
            && ${HI_PYTHON} ${CMAKE_CURRENT_LIST_DIR}/stub/gen_stubapi.py ${METADEF_DIR}/inc/external ${CMAKE_CURRENT_BINARY_DIR}
            && mv attr_value.cc stub_attr_value.cc
            && mv graph.cc stub_graph.cc
            && mv operator.cc stub_operator.cc
            && mv operator_factory.cc stub_operator_factory.cc
            && mv tensor.cc stub_tensor.cc
            && mv inference_context.cc stub_inference_context.cc
            && mv ascend_string.cc stub_ascend_string.cc
            && mv gnode.cc stub_gnode.cc
            &&  echo "Generating stub files end."
)

add_custom_target(graph_stub
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/stub_attr_value.cc
            ${CMAKE_CURRENT_BINARY_DIR}/stub_graph.cc
            ${CMAKE_CURRENT_BINARY_DIR}/stub_operator.cc
            ${CMAKE_CURRENT_BINARY_DIR}/stub_operator_factory.cc
            ${CMAKE_CURRENT_BINARY_DIR}/stub_tensor.cc
            ${CMAKE_CURRENT_BINARY_DIR}/stub_inference_context.cc
            ${CMAKE_CURRENT_BINARY_DIR}/stub_ascend_string.cc
            ${CMAKE_CURRENT_BINARY_DIR}/stub_gnode.cc
)

#############################################################

############ stub/libgraph.so ############
add_library(atc_stub_graph SHARED
    stub_graph.cc
    stub_operator.cc
    stub_operator_factory.cc
    stub_tensor.cc
    stub_attr_value.cc
    stub_ascend_string.cc
    stub_gnode.cc
)
add_dependencies(atc_stub_graph graph_stub)

target_include_directories(atc_stub_graph PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${METADEF_DIR}
)

target_compile_options(atc_stub_graph PRIVATE
    -fno-common
)

target_link_libraries(atc_stub_graph PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    metadef_headers
    fwk_mmpa_headers
    graphengine_headers
)

set_target_properties(atc_stub_graph PROPERTIES
    OUTPUT_NAME graph
    LIBRARY_OUTPUT_DIRECTORY atc_stub
)

############ fwk_stub/libgraph.so ############
add_library(fwk_stub_graph SHARED
    stub_graph.cc
    stub_operator.cc
    stub_operator_factory.cc
    stub_tensor.cc
    stub_attr_value.cc
    stub_inference_context.cc
    stub_ascend_string.cc
    stub_gnode.cc
)

add_dependencies(fwk_stub_graph graph_stub)

target_compile_options(fwk_stub_graph PRIVATE
    -fno-common
)

target_include_directories(fwk_stub_graph PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${METADEF_DIR}
)

target_link_libraries(fwk_stub_graph PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    metadef_headers
    fwk_mmpa_headers
    graphengine_headers
)

set_target_properties(fwk_stub_graph PROPERTIES
    OUTPUT_NAME graph
    LIBRARY_OUTPUT_DIRECTORY fwk_stub
)

############ install ############
install(TARGETS atc_stub_graph OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/stub
)

install(TARGETS fwk_stub_graph OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/fwk_stub
)
