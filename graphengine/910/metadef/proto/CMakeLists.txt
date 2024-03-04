set(METADEF_PROTO_LIST
        "${METADEF_DIR}/proto/task.proto"
        "${METADEF_DIR}/proto/om.proto"
        "${METADEF_DIR}/proto/ge_ir.proto"
        "${METADEF_DIR}/proto/insert_op.proto"
        "${METADEF_DIR}/proto/dump_task.proto"
        "${METADEF_DIR}/proto/fwk_adapter.proto"
        "${METADEF_DIR}/proto/op_mapping.proto"
        "${METADEF_DIR}/proto/onnx/ge_onnx.proto"
        "${METADEF_DIR}/proto/tensorflow/attr_value.proto"
        "${METADEF_DIR}/proto/tensorflow/function.proto"
        "${METADEF_DIR}/proto/tensorflow/graph.proto"
        "${METADEF_DIR}/proto/tensorflow/node_def.proto"
        "${METADEF_DIR}/proto/tensorflow/op_def.proto"
        "${METADEF_DIR}/proto/tensorflow/resource_handle.proto"
        "${METADEF_DIR}/proto/tensorflow/tensor.proto"
        "${METADEF_DIR}/proto/tensorflow/tensor_shape.proto"
        "${METADEF_DIR}/proto/tensorflow/types.proto"
        "${METADEF_DIR}/proto/tensorflow/versions.proto"
        "${METADEF_DIR}/proto/var_manager.proto"
        "${METADEF_DIR}/proto/flow_model.proto"
        "${METADEF_DIR}/proto/aicpu/cpu_attr.proto"
        "${METADEF_DIR}/proto/aicpu/cpu_node_def.proto"
        "${METADEF_DIR}/proto/aicpu/cpu_tensor_shape.proto"
        "${METADEF_DIR}/proto/aicpu/cpu_tensor.proto"
        )

set(METADEF_DATA_FLOW_PROTO_LIST
        "${METADEF_DIR}/proto/ge_ir.proto"
        "${METADEF_DIR}/proto/dflow.proto"
        )

protobuf_generate(metadef_protos METADEF_PROTO_SRCS METADEF_PROTO_HDRS ${METADEF_PROTO_LIST} TARGET)
protobuf_generate(metadef_data_flow_protos METADEF_PROTO_SRCS METADEF_PROTO_HDRS ${METADEF_DATA_FLOW_PROTO_LIST} TARGET)

set(METADEF_GRAPH_PROTO_SRCS
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/om.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/ge_ir.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/insert_op.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/task.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/dump_task.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/fwk_adapter.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/op_mapping.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/onnx/ge_onnx.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/var_manager.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/flow_model.pb.cc"
        )

set(METADEF_TENSORFLOW_PROTO_SRCS
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/graph.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/node_def.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/function.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/versions.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/attr_value.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/op_def.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/tensor.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/tensor_shape.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/types.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/resource_handle.pb.cc"
        )

set(METADEF_AICPU_PROTO_SRCS
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/aicpu/cpu_attr.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/aicpu/cpu_node_def.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/aicpu/cpu_tensor_shape.pb.cc"
        "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/aicpu/cpu_tensor.pb.cc"
        )

set(METADEF_FLOW_GRAPH_PROTO_SRCS
        "${CMAKE_BINARY_DIR}/proto/metadef_data_flow_protos/proto/dflow.pb.cc"
        )

add_library(metadef_graph_protos_obj OBJECT ${METADEF_GRAPH_PROTO_SRCS})
add_dependencies(metadef_graph_protos_obj metadef_protos)
target_compile_definitions(metadef_graph_protos_obj PRIVATE
        google=ascend_private
        )
target_link_libraries(metadef_graph_protos_obj PRIVATE ascend_protobuf intf_pub)
target_compile_options(metadef_graph_protos_obj PRIVATE
        $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal>
        $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
        $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
        )

add_library(metadef_flow_graph_protos_obj OBJECT ${METADEF_FLOW_GRAPH_PROTO_SRCS})
add_dependencies(metadef_flow_graph_protos_obj metadef_data_flow_protos)
target_compile_definitions(metadef_flow_graph_protos_obj PRIVATE
        google=ascend_private
        )
target_link_libraries(metadef_flow_graph_protos_obj PRIVATE ascend_protobuf intf_pub)
target_compile_options(metadef_flow_graph_protos_obj PRIVATE
        $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal>
        $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
        $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
        )

add_library(metadef_tensorflow_protos_obj OBJECT ${METADEF_TENSORFLOW_PROTO_SRCS})
add_dependencies(metadef_tensorflow_protos_obj metadef_protos)
target_compile_definitions(metadef_tensorflow_protos_obj PRIVATE
        google=ascend_private
        )
target_link_libraries(metadef_tensorflow_protos_obj PRIVATE ascend_protobuf intf_pub)
target_compile_options(metadef_tensorflow_protos_obj PRIVATE
        $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal>
        $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
        $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
        )

add_library(metadef_aicpu_protos_obj OBJECT ${METADEF_AICPU_PROTO_SRCS})
add_dependencies(metadef_aicpu_protos_obj metadef_protos)
target_compile_definitions(metadef_aicpu_protos_obj PRIVATE
        google=ascend_private
        )
target_link_libraries(metadef_aicpu_protos_obj PRIVATE ascend_protobuf intf_pub)
target_compile_options(metadef_aicpu_protos_obj PRIVATE
        $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC -Wextra -Wfloat-equal>
        $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
        $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -fno-common -Wextra -Wfloat-equal>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
        $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
        )