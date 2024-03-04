/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#include "register/op_impl_registry.h"
#include "register/op_impl_registry_api.h"
#include "common/ge_common/debug/ge_log.h"
#include "register/shape_inference.h"
#include "graph/any_value.h"
#include "register/op_impl_registry_base.h"
#include "op_impl_register_v2_impl.h"

namespace gert {
namespace {
void RegisterOpImplToRegistry(const OpImplRegisterV2Impl *rd) {
  if (rd == nullptr) {
    GELOGW("The register data is invalid, the impl is nullptr");
    return;
  }
  auto &funcs = OpImplRegistry::GetInstance().CreateOrGetOpImpl(rd->op_type.GetString());
  if (rd->functions.infer_shape != nullptr) {
    funcs.infer_shape = rd->functions.infer_shape;
  }
  if (rd->functions.infer_shape_range != nullptr) {
    funcs.infer_shape_range = rd->functions.infer_shape_range;
  }
  if (rd->functions.infer_datatype != nullptr) {
    funcs.infer_datatype = rd->functions.infer_datatype;
  }
  if (rd->functions.tiling != nullptr) {
    funcs.tiling = rd->functions.tiling;
    funcs.max_tiling_data_size = rd->functions.max_tiling_data_size;
  }
  if (rd->functions.inputs_dependency != 0U) {
    funcs.inputs_dependency = rd->functions.inputs_dependency;
  }
  if (rd->functions.host_inputs != 0U) {
    funcs.host_inputs = rd->functions.host_inputs;
  }
  if (rd->functions.tiling_dependency != 0U) {
    funcs.tiling_dependency = rd->functions.tiling_dependency;
  }
  if (rd->functions.op_execute_func != nullptr) {
    funcs.op_execute_func = rd->functions.op_execute_func;
  }
  if (rd->functions.tiling_parse != nullptr) {
    funcs.tiling_parse = rd->functions.tiling_parse;
    funcs.compile_info_creator = rd->functions.compile_info_creator;
    funcs.compile_info_deleter = rd->functions.compile_info_deleter;
  }
  if (rd->is_private_attr_registered) {
    funcs.private_attrs = rd->functions.private_attrs;
    funcs.unique_private_attrs = rd->functions.unique_private_attrs;
  }
}
}
OpImplRegister::OpImplRegister(const char *op_type)
    : op_type_(op_type),
      functions_(OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type)) {
  (void)reserved_;
  functions_.private_attrs.clear();
  functions_.unique_private_attrs.clear();
}

OpImplRegister &OpImplRegister::InferShape(OpImplKernelRegistry::InferShapeKernelFunc infer_shape_func) {
  functions_.infer_shape = infer_shape_func;
  return *this;
}

OpImplRegister &OpImplRegister::InferShapeRange(
    OpImplKernelRegistry::InferShapeRangeKernelFunc infer_shape_range_func) {
  functions_.infer_shape_range = infer_shape_range_func;
  return *this;
}

OpImplRegister &OpImplRegister::InferDataType(OpImplKernelRegistry::InferDataTypeKernelFunc infer_datatype_func) {
  functions_.infer_datatype = infer_datatype_func;
  return *this;
}
OpImplRegister &OpImplRegister::Tiling(OpImplKernelRegistry::TilingKernelFunc tiling_func,
                                       size_t max_tiling_data_size) {
  functions_.tiling = tiling_func;
  functions_.max_tiling_data_size = max_tiling_data_size;
  return *this;
}

OpImplRegister &OpImplRegister::InputsDataDependency(std::initializer_list<int32_t> inputs) {
  functions_.inputs_dependency = 0UL;
  for (const int32_t index : inputs) {
    if (functions_.SetInputDataDependency(static_cast<size_t>(index)) != ge::GRAPH_SUCCESS) {
      GELOGE(ge::FAILED, "Failed to set data dependency for node %s, the input index %d", op_type_, index);
      return *this;
    }
  }
  return *this;
}

OpImplRegister &OpImplRegister::PrivateAttrImpl(const char *private_attr, ge::AnyValue private_attr_av) {
  if (private_attr == nullptr) {
    GELOGE(ge::FAILED, "Failed to set private attr name using nullptr!");
  } else if (strncmp(private_attr, "", 1U) == 0) {
    GELOGE(ge::FAILED, "Failed to set private attr name using empty string!");
  } else {
    if (functions_.unique_private_attrs.insert(private_attr).second) {
      functions_.private_attrs.emplace_back(std::make_pair(private_attr, std::move(private_attr_av)));
    } else {
      GELOGE(ge::FAILED, "The private attr name: %s has already existed.", private_attr);
    }
  }
  return *this;
}
OpImplRegister &OpImplRegister::PrivateAttr(const char *private_attr) {
  static ge::AnyValue emptyPrivateAttrAV;
  return PrivateAttrImpl(private_attr, emptyPrivateAttrAV);
}
OpImplRegister &OpImplRegister::PrivateAttr(const char *private_attr, int64_t private_attr_val) {
  return PrivateAttrImpl(private_attr, ge::AnyValue::CreateFrom<int64_t>(private_attr_val));
}
OpImplRegister &OpImplRegister::PrivateAttr(const char *private_attr, const std::vector<int64_t> &private_attr_val) {
  return PrivateAttrImpl(private_attr, ge::AnyValue::CreateFrom<std::vector<int64_t>>(private_attr_val));
}
OpImplRegister &OpImplRegister::PrivateAttr(const char *private_attr, const char *private_attr_val) {
  return PrivateAttrImpl(private_attr, ge::AnyValue::CreateFrom<std::string>(private_attr_val));
}
OpImplRegister &OpImplRegister::PrivateAttr(const char *private_attr, const ge::float32_t private_attr_val) {
  return PrivateAttrImpl(private_attr, ge::AnyValue::CreateFrom<ge::float32_t>(private_attr_val));
}
OpImplRegister &OpImplRegister::PrivateAttr(const char *private_attr, bool private_attr_val) {
  return PrivateAttrImpl(private_attr, ge::AnyValue::CreateFrom<bool>(private_attr_val));
}
OpImplRegister &OpImplRegister::PrivateAttr(const char *private_attr, const std::vector<float> &private_attr_val) {
  return PrivateAttrImpl(private_attr, ge::AnyValue::CreateFrom<std::vector<float>>(private_attr_val));
}
OpImplRegistry &OpImplRegistry::GetInstance() {
  static OpImplRegistry instance;
  return instance;
}

OpImplRegistry::OpImplFunctions &OpImplRegistry::CreateOrGetOpImpl(const ge::char_t *op_type) {
  (void)reserved_;
  return types_to_impl_[op_type];
}
const OpImplRegistry::OpImplFunctions *OpImplRegistry::GetOpImpl(const ge::char_t *op_type) const {
  const auto iter = types_to_impl_.find(op_type);
  if (iter == types_to_impl_.end()) {
    return nullptr;
  }
  return &iter->second;
}
const OpImplRegistry::PrivateAttrList &OpImplRegistry::GetPrivateAttrs(const ge::char_t *op_type) const {
  const auto op_impl_ptr = GetOpImpl(op_type);
  if (op_impl_ptr == nullptr) {
    static OpImplRegistry::PrivateAttrList emptyPrivateAttr;
    return emptyPrivateAttr;
  }
  return op_impl_ptr->private_attrs;
}
const std::map<OpImplRegistry::OpType, OpImplRegistry::OpImplFunctions> &OpImplRegistry::GetAllTypesToImpl() const {
  return types_to_impl_;
}
std::map<OpImplRegistry::OpType, OpImplRegistry::OpImplFunctions> &OpImplRegistry::GetAllTypesToImpl() {
  return types_to_impl_;
}
OpImplRegisterV2::OpImplRegisterV2(const ge::char_t *op_type) : impl_(new(std::nothrow) OpImplRegisterV2Impl) {
  if (impl_ != nullptr) {
    impl_->op_type = op_type;
    impl_->functions.infer_shape = nullptr;
    impl_->functions.infer_shape_range = nullptr;
    impl_->functions.infer_datatype = nullptr;
    impl_->functions.inputs_dependency = 0U;
    impl_->functions.op_execute_func = nullptr;
    impl_->functions.host_inputs = 0U;
    impl_->functions.tiling_dependency = 0U;
    // two fields controlled by tiling func
    impl_->functions.tiling = nullptr;
    impl_->functions.max_tiling_data_size = std::numeric_limits<size_t>::max();

    // 3 fields controlled by tiling_parse func
    impl_->functions.tiling_parse = nullptr;
    impl_->functions.compile_info_creator = nullptr;
    impl_->functions.compile_info_deleter = nullptr;

    // private attr controlled by is_private_attr_registered
    (void)OpImplRegistry::GetInstance().CreateOrGetOpImpl(op_type);
  }
}
OpImplRegisterV2::~OpImplRegisterV2() = default;
OpImplRegisterV2::OpImplRegisterV2(const OpImplRegisterV2 &register_data) {
  RegisterOpImplToRegistry(register_data.impl_.get());
}
OpImplRegisterV2::OpImplRegisterV2(OpImplRegisterV2 &&register_data) noexcept {
  RegisterOpImplToRegistry(register_data.impl_.get());
}
OpImplRegisterV2 &OpImplRegisterV2::TilingParse(OpImplKernelRegistry::KernelFunc tiling_parse_func,
                                                OpImplKernelRegistry::CompileInfoCreatorFunc creator_func,
                                                OpImplKernelRegistry::CompileInfoDeleterFunc deleter_func) {
  if (impl_ != nullptr) {
    impl_->functions.tiling_parse = tiling_parse_func;
    impl_->functions.compile_info_creator = creator_func;
    impl_->functions.compile_info_deleter = deleter_func;
  }
  return *this;
}

OpImplRegisterV2 &OpImplRegisterV2::InferShape(OpImplKernelRegistry::InferShapeKernelFunc infer_shape_func) {
  if (impl_ != nullptr) {
    impl_->functions.infer_shape = infer_shape_func;
  }
  return *this;
}

OpImplRegisterV2 &OpImplRegisterV2::InferShapeRange(
    OpImplKernelRegistry::InferShapeRangeKernelFunc infer_shape_range_func) {
  if (impl_ != nullptr) {
    impl_->functions.infer_shape_range = infer_shape_range_func;
  }
  return *this;
}
OpImplRegisterV2 &OpImplRegisterV2::InferDataType(OpImplKernelRegistry::InferDataTypeKernelFunc infer_datatype_func) {
  if (impl_ != nullptr) {
    impl_->functions.infer_datatype = infer_datatype_func;
  }
  return *this;
}
OpImplRegisterV2 &OpImplRegisterV2::Tiling(OpImplKernelRegistry::TilingKernelFunc tiling_func,
                                           size_t max_tiling_data_size) {
  if (impl_ != nullptr) {
    impl_->functions.tiling = tiling_func;
    impl_->functions.max_tiling_data_size = max_tiling_data_size;
  }
  return *this;
}
OpImplRegisterV2 &OpImplRegisterV2::InputsDataDependency(std::initializer_list<int32_t> inputs) {
  if (impl_ != nullptr) {
    for (const int32_t index : inputs) {
      if (impl_->functions.IsInputDataDependency(static_cast<size_t>(index))) {
        continue;
      }
      if (impl_->functions.IsTilingInputDataDependency(static_cast<size_t>(index))) {
        GELOGW("Input[%d] of node %s has been register tiling dependency, "
            "it will be override by data dependency",
            index, impl_->op_type.GetString());
      }
      if (impl_->functions.SetInputDataDependency(static_cast<size_t>(index)) != ge::GRAPH_SUCCESS) {
        GELOGE(ge::FAILED, "Failed to set data dependency for node %s, the input index %d", impl_->op_type.GetString(),
               index);
        return *this;
      }
    }
  }
  return *this;
}

OpImplRegisterV2 &OpImplRegisterV2::TilingInputsDataDependency(std::initializer_list<int32_t> inputs) {
  if (impl_ != nullptr) {
    for (const int32_t index : inputs) {
      if (impl_->functions.IsInputDataDependency(static_cast<size_t>(index))) {
        GELOGW("Failed to set tiling dependency for input[%d] of node %s, "
               "because it has been register data dependency",
               index, impl_->op_type.GetString());
      } else if (impl_->functions.IsTilingInputDataDependency(static_cast<size_t>(index))) {
        continue;
      }
      if (impl_->functions.SetTilingInputDataDependency(static_cast<size_t>(index)) != ge::GRAPH_SUCCESS) {
        GELOGE(ge::FAILED, "Failed to set tiling dependency for node %s, the input index %d",
            impl_->op_type.GetString(), index);
        return *this;
      }
    }
  }
  return *this;
}

OpImplRegisterV2 &OpImplRegisterV2::PrivateAttr(const ge::char_t *private_attr, ge::AnyValue private_attr_av) {
  if (private_attr == nullptr) {
    GELOGW("Failed to set private attr name using nullptr!");
    return *this;
  }
  if (strncmp(private_attr, "", 1U) == 0) {
    GELOGW("Failed to set private attr name using empty string!");
    return *this;
  }
  if (impl_ != nullptr) {
    impl_->is_private_attr_registered = true;
    if (impl_->functions.unique_private_attrs.insert(private_attr).second) {
      impl_->functions.private_attrs.emplace_back(std::make_pair(private_attr, std::move(private_attr_av)));
    } else {
      GELOGW("The private attr name: %s has already existed.", private_attr);
    }
  }
  return *this;
}
OpImplRegisterV2 &OpImplRegisterV2::PrivateAttr(const ge::char_t *private_attr) {
  static ge::AnyValue empty;
  return PrivateAttr(private_attr, empty);
}
OpImplRegisterV2 &OpImplRegisterV2::PrivateAttr(const ge::char_t *private_attr, int64_t private_attr_val) {
  return PrivateAttr(private_attr, ge::AnyValue::CreateFrom<int64_t>(private_attr_val));
}
OpImplRegisterV2 &OpImplRegisterV2::PrivateAttr(const ge::char_t *private_attr,
                                                const std::vector<int64_t> &private_attr_val) {
  return PrivateAttr(private_attr, ge::AnyValue::CreateFrom<std::vector<int64_t>>(private_attr_val));
}
OpImplRegisterV2 &OpImplRegisterV2::PrivateAttr(const ge::char_t *private_attr, const ge::char_t *private_attr_val) {
  return PrivateAttr(private_attr, ge::AnyValue::CreateFrom<std::string>(private_attr_val));
}
OpImplRegisterV2 &OpImplRegisterV2::PrivateAttr(const ge::char_t *private_attr, ge::float32_t private_attr_val) {
  return PrivateAttr(private_attr, ge::AnyValue::CreateFrom<ge::float32_t>(private_attr_val));
}
OpImplRegisterV2 &OpImplRegisterV2::PrivateAttr(const ge::char_t *private_attr, bool private_attr_val) {
  return PrivateAttr(private_attr, ge::AnyValue::CreateFrom<bool>(private_attr_val));
}
OpImplRegisterV2 &OpImplRegisterV2::PrivateAttr(const ge::char_t *private_attr,
                                                const std::vector<ge::float32_t> &private_attr_val) {
  return PrivateAttr(private_attr, ge::AnyValue::CreateFrom<std::vector<ge::float32_t>>(private_attr_val));
}
OpImplRegisterV2 &OpImplRegisterV2::OpExecuteFunc(OpImplKernelRegistry::OpExecuteFunc op_execute_func) {
  if (impl_ != nullptr) {
    impl_->functions.op_execute_func = op_execute_func;
  }
  return *this;
}
OpImplRegisterV2 &OpImplRegisterV2::HostInputs(std::initializer_list<int32_t> inputs) {
  if (impl_ != nullptr) {
    impl_->functions.host_inputs = 0UL;
    for (const int32_t index : inputs) {
      if (impl_->functions.SetHostInputs(static_cast<size_t>(index)) != ge::GRAPH_SUCCESS) {
        GELOGE(ge::FAILED, "Failed to set host input for node %s, the input index %d", impl_->op_type.GetString(),
               index);
        return *this;
      }
    }
  }
  return *this;
}
}  // namespace gert

#ifdef __cplusplus
extern "C" {
#endif

size_t GetRegisteredOpNum(void) {
  return gert::OpImplRegistry::GetInstance().GetAllTypesToImpl().size();
}
int32_t GetOpImplFunctions(TypesToImpl *impl, size_t impl_num) {
  const auto types_to_impl = gert::OpImplRegistry::GetInstance().GetAllTypesToImpl();
  if (impl_num != types_to_impl.size()) {
    GELOGE(ge::FAILED, "Get types_to_impl_ failed, impl_num[%zu] and map size[%zu] not match",
           impl_num, types_to_impl.size());
    return static_cast<int32_t>(ge::GRAPH_FAILED);
  }
  size_t cnt = 0U;
  for (auto &it : types_to_impl) {
    impl[cnt].op_type = it.first.GetString();
    impl[cnt].funcs = it.second;
    cnt++;
  }
  return static_cast<int32_t>(ge::GRAPH_SUCCESS);
}
#ifdef __cplusplus
}
#endif

