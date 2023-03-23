/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_IR_IMPL_KERNEL_IMPL_REGISTRY_H_
#define AIR_CXX_RUNTIME_V2_IR_IMPL_KERNEL_IMPL_REGISTRY_H_
#include <initializer_list>
#include <string>
#include <map>
#include "graph/ge_error_codes.h"
#include "op_impl_kernel_registry.h"
#include "exe_graph/runtime/tiling_parse_context.h"
namespace gert {
class OpImplRegistry : public OpImplKernelRegistry {
 public:
  static OpImplRegistry &GetInstance();
  OpImplFunctions &CreateOrGetOpImpl(const OpType &op_type);
  const OpImplFunctions *GetOpImpl(const OpType &op_type) const override;
  const PrivateAttrList &GetPrivateAttrs(const OpType &op_type) const override;
 private:
  std::map<OpType, OpImplFunctions> types_to_impl_;
};

class OpImplRegister {
 public:
  typedef UINT32 (*TilingParseFunc)(TilingParseContext *context);

  explicit OpImplRegister(const char *op_type);
  OpImplRegister &InferShape(OpImplKernelRegistry::InferShapeKernelFunc infer_shape_func);
  OpImplRegister &InferShapeRange(OpImplKernelRegistry::InferShapeRangeKernelFunc infer_shape_range_func);
  OpImplRegister &InferDataType(OpImplKernelRegistry::InferDataTypeKernelFunc infer_datatype_func);
  OpImplRegister &Tiling(OpImplKernelRegistry::TilingKernelFunc tiling_func, size_t max_tiling_data_size = 2048);
  OpImplRegister &PrivateAttr(const char *private_attr);
  OpImplRegister &PrivateAttr(const char *private_attr, int64_t private_attr_val);
  OpImplRegister &PrivateAttr(const char *private_attr, const std::vector<int64_t> &private_attr_val);
  OpImplRegister &PrivateAttr(const char *private_attr, const char *private_attr_val);
  OpImplRegister &PrivateAttr(const char *private_attr, float private_attr_val);
  OpImplRegister &PrivateAttr(const char *private_attr, bool private_attr_val);
  OpImplRegister &PrivateAttr(const char *private_attr, const std::vector<float> &private_attr_val);
  template<typename T>
  OpImplRegister &TilingParse(KernelRegistry::KernelFunc tiling_parse_func) {
    functions_.tiling_parse = tiling_parse_func;
    functions_.compile_info_creator = CreateCompileInfo<T>;
    functions_.compile_info_deleter = DeleteCompileInfo<T>;
    return *this;
  }
  template<typename T>
  OpImplRegister &TilingParse(TilingParseFunc tiling_parse_func) {
    functions_.tiling_parse = reinterpret_cast<KernelRegistry::KernelFunc>(tiling_parse_func);
    functions_.compile_info_creator = CreateCompileInfo<T>;
    functions_.compile_info_deleter = DeleteCompileInfo<T>;
    return *this;
  }
  OpImplRegister &InputsDataDependency(std::initializer_list<int32_t> inputs);

 private:
  template<typename T, typename std::enable_if<(!std::is_array<T>::value), int>::type = 0>
  static void *CreateCompileInfo() {
    return new T();
  }
  template<typename T>
  static void DeleteCompileInfo(void *obj) {
    delete reinterpret_cast<T *>(obj);
  }
  template<size_t MaxLen>
  static void *CreateDynamicLenTilingData() {
    return TilingData::CreateCap(MaxLen).release();
  }
  OpImplRegister &PrivateAttrImpl(const char *private_attr, ge::AnyValue private_attr_av);

 private:
  const char *op_type_;
  OpImplRegistry::OpImplFunctions &functions_;
};
}  // namespace gert

#define IMPL_OP(op_type) static gert::OpImplRegister op_impl_register_##op_type = gert::OpImplRegister(#op_type)
#define IMPL_OP_DEFAULT() IMPL_OP(DefaultImpl)

#endif  //AIR_CXX_RUNTIME_V2_IR_IMPL_KERNEL_IMPL_REGISTRY_H_
