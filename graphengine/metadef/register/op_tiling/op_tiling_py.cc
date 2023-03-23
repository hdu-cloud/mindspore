/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include <securec.h>
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/debug/ge_log.h"
#include "register/op_tiling_info.h"
#include "register/op_tiling_registry.h"
#include "op_tiling/op_tiling_utils.h"
#include "op_tiling/op_tiling_constants.h"
#include "common/util/tiling_utils.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/kernel_run_context_builder.h"
#include "exe_graph/runtime/tiling_context.h"
#include "common/checker.h"

namespace optiling {
using ParseAttrFunc = std::function<void(ge::OpDescPtr &, const nlohmann::json &, const std::string &)>;
using CopyConstDataFunc = std::function<bool(const nlohmann::json &, const size_t, std::unique_ptr<uint8_t[]> &)>;

class FuncTable {
public:
  FuncTable() = default;
  FuncTable &Init() {
    funcs_.resize(ge::DT_MAX, nullptr);
    return *this;
  }

  FuncTable &Insert(ge::DataType index, CopyConstDataFunc func) {
    funcs_[index] = func;
    return *this;
  }

  CopyConstDataFunc Find(ge::DataType index) const {
    return funcs_[index];
  }

private:
  std::vector<CopyConstDataFunc> funcs_;
};

namespace {
constexpr uint32_t kRightShiftBits = 4;
constexpr uint32_t kAndBits = 15;
constexpr char kHexDigits[] = "0123456789ABCDEF";
constexpr size_t kSize = 2UL;
constexpr char const *kMaxTilingSize = "op_para_size";
constexpr size_t kMaxTilingDataSize = 16UL * 1024UL;
constexpr size_t kWorkspaceHolerSize = 8UL;

struct ContextComponent {
  std::vector<gert::StorageShape> storage_shapes;
  std::vector<std::pair<uint32_t, std::unique_ptr<uint8_t[]>>> index_to_tensors;
  ge::OpDescPtr op_desc {nullptr};
  std::unique_ptr<uint8_t[]> tiling_data;
  std::unique_ptr<uint8_t[]> workspace_size;
  bool atomic_flag = true;
};

bool EnableGert() {
  const char *const enable_gert = std::getenv("ENABLE_RUNTIME_V2");
  if (enable_gert == nullptr) {
    return false;
  }
  return true;
}

bool FindImplFuncs(const char *op_type, const gert::OpImplRegistry::OpImplFunctions *&funcs) {
    funcs = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type);
    if (funcs == nullptr || funcs->tiling == nullptr || funcs->tiling_parse == nullptr) {
      funcs = gert::OpImplRegistry::GetInstance().GetOpImpl("DefaultImpl");
      if (funcs == nullptr || funcs->tiling == nullptr || funcs->tiling_parse == nullptr) {
        GELOGE(ge::GRAPH_FAILED, "funcs/tiling/tiling_parse is null. op type is %s.", op_type);
        REPORT_CALL_ERROR("E19999", "funcs/tiling/tiling_parse is null. op type is %s.", op_type);
        return false;
      }
    }
    return true;
}

template<typename T>
void ParseAndSetAttr(ge::OpDescPtr &op_desc, const nlohmann::json &attr, const std::string &attr_name) {
  T attr_value = attr["value"].get<T>();
  op_desc->AppendIrAttrName(attr_name);
  (void)op_desc->SetAttr(attr_name, ge::AnyValue::CreateFrom<T>(attr_value));
}

template<typename T>
void ParseAndSetListAttr(ge::OpDescPtr &op_desc, const nlohmann::json &attr, const std::string &attr_name) {
  std::vector<T> attr_value = attr["value"].get<std::vector<T>>();
  op_desc->AppendIrAttrName(attr_name);
  (void)op_desc->SetAttr(attr_name, ge::AnyValue::CreateFrom<std::vector<T>>(attr_value));
}

void ParseAndSetListInt64Attr(ge::OpDescPtr &op_desc, const nlohmann::json &attr, const std::string &attr_name) {
  std::vector<int32_t> attr_value = attr["value"].get<std::vector<int32_t>>();
  std::vector<int64_t> attr_int64_value;
  for (auto item : attr_value) {
    attr_int64_value.emplace_back(static_cast<int64_t>(item));
  }
  op_desc->AppendIrAttrName(attr_name);
  (void)op_desc->SetAttr(attr_name, ge::AnyValue::CreateFrom<std::vector<int64_t>>(attr_int64_value));
}

void ParseAndSetListListAttr(ge::OpDescPtr &op_desc, const nlohmann::json &attr, const std::string &attr_name) {
  std::vector<std::vector<int32_t>> attr_value_int32 = attr["value"].get<std::vector<std::vector<int32_t>>>();
  std::vector<std::vector<int64_t>> attr_value_int64;
  std::vector<int64_t> temp_int64_vec;
  for (const auto &vec_int32 : attr_value_int32) {
    for (const auto &item : vec_int32) {
      int64_t tmp = static_cast<int64_t>(item);
      temp_int64_vec.emplace_back(tmp);
    }
    attr_value_int64.emplace_back(temp_int64_vec);
    temp_int64_vec.clear();
  }
  op_desc->AppendIrAttrName(attr_name);
  (void)op_desc->SetAttr(attr_name, ge::AnyValue::CreateFrom<std::vector<std::vector<int64_t>>>(attr_value_int64));
}

void ParseAndSetListListInt64Attr(ge::OpDescPtr &op_desc, const nlohmann::json &attr, const std::string &attr_name) {
  const std::vector<std::vector<int64_t>> attr_value_int64 = attr["value"].get<std::vector<std::vector<int64_t>>>();
  op_desc->AppendIrAttrName(attr_name);
  (void)op_desc->SetAttr(attr_name, ge::AnyValue::CreateFrom<std::vector<std::vector<int64_t>>>(attr_value_int64));
}

template<typename T>
bool GetConstData(const nlohmann::json &json_array, const size_t total_size,
                  std::unique_ptr<uint8_t[]> &tensor_holder) {
  std::vector<T> value = json_array.get<std::vector<T>>();
  auto tensor = reinterpret_cast<gert::Tensor *>(tensor_holder.get());
  if (memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), value.data(), value.size() * sizeof(T)) !=
      EOK) {
    GELOGE(ge::FAILED, "Call memcpy failed, total value size is %zu.", value.size() * sizeof(T));
    return false;
  }
  return true;
}

bool GetConstDataWithFloat16(const nlohmann::json &json_array, const size_t total_size,
                             std::unique_ptr<uint8_t[]> &tensor_holder) {
  std::vector<float> const_value = json_array.get<std::vector<float>>();
  std::vector<uint16_t> const_data_vec;
  for (size_t i = 0UL; i < const_value.size(); ++i) {
    uint16_t const_data_uint16 = FloatToUint16(const_value[i]);
    const_data_vec.emplace_back(const_data_uint16);
  }
  auto tensor = reinterpret_cast<gert::Tensor *>(tensor_holder.get());
  if (memcpy_s(tensor->GetData<uint8_t>(), total_size - sizeof(gert::Tensor), const_data_vec.data(),
               const_data_vec.size() * sizeof(uint16_t)) != EOK) {
    GELOGE(ge::FAILED, "Call memcpy failed, total value size is %zu.", const_data_vec.size() * sizeof(uint16_t));
    return false;
  }
  return true;
}

const std::unordered_map<std::string, ParseAttrFunc> kDtypeToAttrFunc = {
    {"bool", ParseAndSetAttr<bool>},
    {"float", ParseAndSetAttr<float>},
    {"float32", ParseAndSetAttr<float>},
    {"int", ParseAndSetAttr<int64_t>},
    {"int32", ParseAndSetAttr<int64_t>},
    {"int64", ParseAndSetAttr<int64_t>},
    {"str", ParseAndSetAttr<std::string>},
    {"list_bool", ParseAndSetListAttr<bool>},
    {"list_float", ParseAndSetListAttr<float>},
    {"list_float32", ParseAndSetListAttr<float>},
    {"list_int", ParseAndSetListInt64Attr},
    {"list_int32", ParseAndSetListInt64Attr},
    {"list_int64", ParseAndSetListAttr<int64_t>},
    {"list_str", ParseAndSetListAttr<std::string>},
    {"list_list_int", ParseAndSetListListAttr},
    {"list_list_int32", ParseAndSetListListAttr},
    {"list_list_int64", ParseAndSetListListInt64Attr}};

const FuncTable kFuncTable = FuncTable()
                             .Init()
                             .Insert(ge::DT_INT8, GetConstData<int8_t>)
                             .Insert(ge::DT_UINT8, GetConstData<uint8_t>)
                             .Insert(ge::DT_INT16, GetConstData<int16_t>)
                             .Insert(ge::DT_UINT16, GetConstData<uint16_t>)
                             .Insert(ge::DT_INT32, GetConstData<int32_t>)
                             .Insert(ge::DT_UINT32, GetConstData<uint32_t>)
                             .Insert(ge::DT_INT64, GetConstData<int64_t>)
                             .Insert(ge::DT_UINT64, GetConstData<uint64_t>)
                             .Insert(ge::DT_FLOAT, GetConstData<float>)
                             .Insert(ge::DT_DOUBLE, GetConstData<double>)
                             .Insert(ge::DT_FLOAT16, GetConstDataWithFloat16);

void ParseDtype(const nlohmann::json &json, ge::GeTensorDesc &tensor_desc) {
  if (json.contains("dtype")) {
    std::string dtype_str = json["dtype"].get<std::string>();
    (void)std::transform(dtype_str.begin(), dtype_str.end(), dtype_str.begin(), ::toupper);
    dtype_str = "DT_" + dtype_str;
    ge::DataType ge_dtype = ge::TypeUtils::SerialStringToDataType(dtype_str);
    tensor_desc.SetDataType(ge_dtype);
  }
}

void ParseStorageShape(const nlohmann::json &json, gert::StorageShape &storage_shape,
                       std::vector<gert::StorageShape> &storage_shapes) {
  if (json.contains("shape")) {
    gert::Shape shape;
    const auto dims = json["shape"].get<std::vector<int64_t>>();
    for (auto dim : dims) {
      (void)shape.AppendDim(dim);
    }
    storage_shape.MutableStorageShape() = shape;
  }
  if (json.contains("ori_shape")) {
    gert::Shape shape;
    const auto dims = json["ori_shape"].get<std::vector<int64_t>>();
    for (auto dim : dims) {
      (void)shape.AppendDim(dim);
    }
    storage_shape.MutableOriginShape() = shape;
  }
  storage_shapes.emplace_back(storage_shape);
}

void ParseStorageFormat(const nlohmann::json &json, ge::GeTensorDesc &tensor_desc) {
  if (json.contains("format")) {
    std::string format_str = json["format"].get<std::string>();
    (void)std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::toupper);
    ge::Format ge_format = ge::TypeUtils::SerialStringToFormat(format_str);
    tensor_desc.SetFormat(ge_format);
  }
  if (json.contains("ori_format")) {
    std::string format_str = json["ori_format"].get<std::string>();
    (void)std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::toupper);
    ge::Format ge_format = ge::TypeUtils::SerialStringToFormat(format_str);
    tensor_desc.SetOriginFormat(ge_format);
  }
}

ge::graphStatus ParseConstValue(const nlohmann::json &input, const gert::StorageShape &storage_shape,
                                const ge::GeTensorDesc &tensor_desc, const uint32_t index,
                                std::vector<std::pair<uint32_t, std::unique_ptr<uint8_t[]>>> &index_to_tensor) {
  if (input.contains("const_value")) {
    if (!input.contains("name")) {
      GELOGE(ge::GRAPH_FAILED, "Const tensor has no name.");
      return ge::GRAPH_FAILED;
    }

    size_t total_size = 0UL;
    auto tensor_holder = gert::Tensor::CreateFollowing(storage_shape.GetStorageShape().GetShapeSize(),
                                                       tensor_desc.GetDataType(), total_size);
    GE_CHECK_NOTNULL(tensor_holder);
    auto func = kFuncTable.Find(tensor_desc.GetDataType());
    GE_CHECK_NOTNULL(func);
    if (!func(input["const_value"], total_size, tensor_holder)) {
      GELOGE(ge::GRAPH_FAILED, "Make tensor failed.");
      return ge::GRAPH_FAILED;
    }
    auto tensor = reinterpret_cast<gert::Tensor *>(tensor_holder.get());
    tensor->MutableOriginShape() = storage_shape.GetOriginShape();
    tensor->MutableStorageShape() = storage_shape.GetStorageShape();
    tensor->SetDataType(tensor_desc.GetDataType());
    tensor->SetStorageFormat(tensor_desc.GetFormat());
    tensor->SetOriginFormat(tensor_desc.GetOriginFormat());
    index_to_tensor.emplace_back(index, std::move(tensor_holder));
  } else {
    auto tensor_holder = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[sizeof(gert::Tensor)]);
    GE_ASSERT_NOTNULL(tensor_holder);
    new (tensor_holder.get()) gert::Tensor({{}, {}}, {tensor_desc.GetOriginFormat(), tensor_desc.GetFormat(), {}},
                                           gert::kOnHost, tensor_desc.GetDataType(), nullptr);
    reinterpret_cast<gert::Tensor *>(tensor_holder.get())->MutableStorageShape() = storage_shape.GetStorageShape();
    reinterpret_cast<gert::Tensor *>(tensor_holder.get())->MutableOriginShape() = storage_shape.GetOriginShape();
    index_to_tensor.emplace_back(index, std::move(tensor_holder));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseInput(const nlohmann::json &input, ge::OpDescPtr &op_desc, const uint32_t index,
                           std::vector<gert::StorageShape> &storage_shapes,
                           std::vector<std::pair<uint32_t, std::unique_ptr<uint8_t[]>>> &index_to_tensor) {
  ge::GeTensorDesc tensor_desc;
  gert::StorageShape storage_shape;
  ParseDtype(input, tensor_desc);
  ParseStorageShape(input, storage_shape, storage_shapes);
  ParseStorageFormat(input, tensor_desc);
  const auto ret = ParseConstValue(input, storage_shape, tensor_desc, index, index_to_tensor);
  if (ret != ge::GRAPH_SUCCESS) {
    return ret;
  }

  if (input.contains("name")) {
    const std::string name = input["name"];
    tensor_desc.SetName(name);
    op_desc->AppendIrInput(name, ge::kIrInputRequired);
    (void)op_desc->AddInputDesc(name, tensor_desc);
  } else {
    op_desc->AppendIrInput(std::to_string(index), ge::kIrInputRequired);
    (void)op_desc->AddInputDesc(std::to_string(index), tensor_desc);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseInputs(const char *inputs, ge::OpDescPtr &op_desc, std::vector<gert::StorageShape> &storage_shapes,
                            std::vector<std::pair<uint32_t, std::unique_ptr<uint8_t[]>>> &index_to_tensor) {
  nlohmann::json desc_list;
  try {
    desc_list = nlohmann::json::parse(inputs);
  } catch (const nlohmann::json::exception &e) {
    GELOGE(ge::GRAPH_FAILED, "Parse json exception. %s", inputs);
    return ge::GRAPH_FAILED;
  }
  uint32_t index = 0;
  for (const auto &desc : desc_list) {
    if (desc.is_array()) {
      for (const auto &ele : desc) {
        if (ParseInput(ele, op_desc, index, storage_shapes, index_to_tensor) != ge::GRAPH_SUCCESS) {
          return ge::GRAPH_FAILED;
        }
        ++index;
      }
    } else {
      if (ParseInput(desc, op_desc, index, storage_shapes, index_to_tensor) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
      }
      ++index;
    }
  }
  return ge::GRAPH_SUCCESS;
}

void ParseOutput(const nlohmann::json &output, ge::OpDescPtr &op_desc,
                 std::vector<gert::StorageShape> &storage_shapes) {
  ge::GeTensorDesc tensor_desc;
  gert::StorageShape storage_shape;
  ParseDtype(output, tensor_desc);
  ParseStorageShape(output, storage_shape, storage_shapes);
  ParseStorageFormat(output, tensor_desc);

  if (output.contains("name")) {
    const std::string name = output["name"];
    tensor_desc.SetName(name);
    (void)op_desc->AddOutputDesc(name, tensor_desc);
  } else {
    (void)op_desc->AddOutputDesc(tensor_desc);
  }
}

ge::graphStatus ParseOutputs(const char *outputs, ge::OpDescPtr &op_desc,
                             std::vector<gert::StorageShape> &storage_shapes) {
  nlohmann::json desc_list;
  try {
    desc_list = nlohmann::json::parse(outputs);
  } catch (const nlohmann::json::exception &e) {
    GELOGE(ge::GRAPH_FAILED, "Parse json exception. %s", outputs);
    return ge::GRAPH_FAILED;
  }
  for (const auto &desc : desc_list) {
    if (desc.is_array()) {
      for (const auto &ele : desc) {
        ParseOutput(ele, op_desc, storage_shapes);
      }
    } else {
      ParseOutput(desc, op_desc, storage_shapes);
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseAttrs(const char *attrs, ge::OpDescPtr &op_desc) {
  if (attrs == nullptr) {
    GELOGD("Attrs has not been set.");
  } else {
    nlohmann::json attrs_json;
    try {
      attrs_json = nlohmann::json::parse(attrs);
    } catch (const nlohmann::json::exception &e) {
      GELOGE(ge::GRAPH_FAILED, "Parse json exception. %s", attrs);
      return ge::GRAPH_FAILED;
    }
    for (const auto &attr : attrs_json) {
      if (!attr.contains("name") || !attr.contains("dtype") || !attr.contains("value")) {
        GELOGE(ge::GRAPH_FAILED, "cur attr does not contain name or dtype or value.");
        return ge::GRAPH_FAILED;
      }
      const std::string attr_name = attr["name"].get<std::string>();
      const std::string dtype = attr["dtype"].get<std::string>();
      const auto iter = kDtypeToAttrFunc.find(dtype);
      if (iter == kDtypeToAttrFunc.end()) {
        GELOGE(ge::GRAPH_FAILED, "Unknown dtype[%s], which is unsupported.", dtype.c_str());
        return ge::GRAPH_FAILED;
      }
      (iter->second)(op_desc, attr, attr_name);
      GELOGD("Finish to set attr[name: %s] to Operator.", attr_name.c_str());
    }
  }
  return ge::GRAPH_SUCCESS;
}

std::string DumpTilingData(gert::TilingData *tiling_data) {
  std::string output;
  output.reserve(tiling_data->GetDataSize() * kSize);
  char *data = reinterpret_cast<char *>(tiling_data->GetData());
  for (size_t i = 0UL; i < tiling_data->GetDataSize(); ++i) {
    unsigned char ch = static_cast<unsigned char>(data[i]);
    output.push_back(kHexDigits[ch >> kRightShiftBits]);
    output.push_back(kHexDigits[ch & kAndBits]);
  }
  return output;
}

bool DumpRunInfo(gert::KernelContext *kernel_context, char *run_info_json, const size_t run_info_len) {
  GE_ASSERT_NOTNULL(run_info_json);
  nlohmann::json json_obj;
  auto ws = kernel_context->GetOutputPointer<gert::ContinuousVector>(gert::TilingContext::kOutputWorkspace);
  std::vector<size_t> workspaces(reinterpret_cast<const size_t *>(ws->GetData()),
                                 reinterpret_cast<const size_t *>(ws->GetData()) + ws->GetSize());
  json_obj["block_dim"] = *kernel_context->GetOutputPointer<uint64_t>(gert::TilingContext::kOutputBlockDim);
  json_obj["workspaces"] = workspaces;
  json_obj["tiling_data"] =
      DumpTilingData(kernel_context->GetOutputPointer<gert::TilingData>(gert::TilingContext::kOutputTilingData));
  json_obj["clear_atomic"] = *kernel_context->GetOutputPointer<bool>(gert::TilingContext::kOutputAtomicCleanFlag);
  json_obj["tiling_key"] = *kernel_context->GetOutputPointer<uint64_t>(gert::TilingContext::kOutputTilingKey);
  const std::string str = json_obj.dump();
  return memcpy_s(run_info_json, run_info_len, str.c_str(), str.size() + 1) == EOK;
}
}  // namespace

using ParseAndSetAttrValueFunc = std::function<void(ge::Operator &, const nlohmann::json &, const std::string &)>;
using ParseAndSetAttrValuePtr = std::shared_ptr<ParseAndSetAttrValueFunc>;

thread_local int64_t last_op_tiling_perf = -1;

template<typename T>
void ParseAndSetAttrValue(ge::Operator &op, const nlohmann::json &attr, const std::string &attr_name) {
  T attr_value = attr["value"].get<T>();
  (void)op.SetAttr(attr_name.c_str(), attr_value);
}

template<typename T>
void ParseAndSetAttrListValue(ge::Operator &op, const nlohmann::json &attr, const std::string &attr_name) {
  std::vector<T> attr_value = attr["value"].get<std::vector<T>>();
  (void)op.SetAttr(attr_name.c_str(), attr_value);
}

void ParseAndSetAttrListListValue(ge::Operator &op, const nlohmann::json &attr, const std::string &attr_name) {
  std::vector<std::vector<int32_t>> attr_value_int32 = attr["value"].get<std::vector<std::vector<int32_t>>>();
  std::vector<std::vector<int64_t>> attr_value_int64;
  std::vector<int64_t> temp_int64_vec;
  for (const auto &vec_int32 : attr_value_int32) {
    for (const auto &item : vec_int32) {
      int64_t tmp = static_cast<int64_t>(item);
      temp_int64_vec.emplace_back(tmp);
    }
    attr_value_int64.emplace_back(temp_int64_vec);
    temp_int64_vec.clear();
  }

  (void)op.SetAttr(attr_name.c_str(), attr_value_int64);
}

void ParseAndSetAttrListListInt64Value(ge::Operator &op, const nlohmann::json &attr, const std::string &attr_name) {
  const std::vector<std::vector<int64_t>> attr_value_int64 = attr["value"].get<std::vector<std::vector<int64_t>>>();
  (void)op.SetAttr(attr_name.c_str(), attr_value_int64);
}

const std::map<std::string, ParseAndSetAttrValuePtr> parse_attr_dtype_map = {
    {"bool", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrValue<bool>)},
    {"float", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrValue<float>)},
    {"float32", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrValue<float>)},
    {"int", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrValue<int32_t>)},
    {"int32", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrValue<int32_t>)},
    {"int64", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrValue<int64_t>)},
    {"str", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrValue<std::string>)},
    {"list_bool", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListValue<bool>)},
    {"list_float", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListValue<float>)},
    {"list_float32", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListValue<float>)},
    {"list_int", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListValue<int32_t>)},
    {"list_int32", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListValue<int32_t>)},
    {"list_int64", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListValue<int64_t>)},
    {"list_str", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListValue<std::string>)},
    {"list_list_int", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListListValue)},
    {"list_list_int32", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListListValue)},
    {"list_list_int64", std::make_shared<ParseAndSetAttrValueFunc>(&ParseAndSetAttrListListInt64Value)}};

void ParseShapeDesc(const nlohmann::json &shape, std::vector<TeOpTensor> &tensors) {
  TeOpTensor tensor;
  if (shape.contains("shape")) {
    tensor.shape = shape["shape"].get<std::vector< int64_t>>();
  }
  if (shape.contains("ori_shape")) {
    tensor.ori_shape = shape["ori_shape"].get<std::vector<int64_t>>();
  }
  if (shape.contains("format")) {
    tensor.format = shape["format"].get<std::string>();
  }
  if (shape.contains("ori_format")) {
    tensor.ori_format = shape["ori_format"].get<std::string>();
  }
  if (shape.contains("dtype")) {
    tensor.dtype = shape["dtype"].get<std::string>();
  }
  tensors.emplace_back(tensor);
}

void ParseShapeDescList(const nlohmann::json &shape_list, std::vector<TeOpTensorArg> &op_args) {
  for (const auto &elem : shape_list) {
    TeOpTensorArg tensor_arg;
    tensor_arg.arg_type = TensorArgType::TA_NONE;

    if (elem.is_array()) {
      tensor_arg.arg_type = TensorArgType::TA_LIST;
      for (const auto &shape : elem) {
        ParseShapeDesc(shape, tensor_arg.tensor);
      }
    } else {
      tensor_arg.arg_type = TensorArgType::TA_SINGLE;
      ParseShapeDesc(elem, tensor_arg.tensor);
    }
    op_args.emplace_back(tensor_arg);
  }
}

void ParseShapeDescV2(const nlohmann::json &shape, ge::OpDescPtr &op_desc, const bool &is_input) {
  ge::GeTensorDesc tensor;
  std::string name;
  if (shape.contains("shape")) {
    tensor.SetShape(ge::GeShape(shape["shape"].get<std::vector<int64_t>>()));
  }
  if (shape.contains("ori_shape")) {
    tensor.SetOriginShape(ge::GeShape(shape["ori_shape"].get<std::vector<int64_t>>()));
  }
  if (shape.contains("format")) {
    std::string format_str = shape["format"].get<std::string>();
    std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::toupper);
    ge::Format ge_format = ge::TypeUtils::SerialStringToFormat(format_str);
    tensor.SetFormat(ge_format);
  }
  if (shape.contains("ori_format")) {
    std::string format_str = shape["ori_format"].get<std::string>();
    std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::toupper);
    ge::Format ge_format = ge::TypeUtils::SerialStringToFormat(format_str);
    tensor.SetOriginFormat(ge_format);
  }
  if (shape.contains("dtype")) {
    std::string dtype_str = shape["dtype"].get<std::string>();
    std::transform(dtype_str.begin(), dtype_str.end(), dtype_str.begin(), ::toupper);
    dtype_str = "DT_" + dtype_str;
    ge::DataType ge_dtype = ge::TypeUtils::SerialStringToDataType(dtype_str);
    tensor.SetDataType(ge_dtype);
  }
  if (shape.contains("name")) {
    name = shape["name"];
    tensor.SetName(name);
    is_input ? op_desc->AddInputDesc(name, tensor) : op_desc->AddOutputDesc(name, tensor);
  } else {
    is_input ? op_desc->AddInputDesc(tensor) : op_desc->AddOutputDesc(tensor);
  }
}

void ParseAndSetAttr(const nlohmann::json &attr, ge::Operator &op) {
  if (!attr.contains("name") || !attr.contains("dtype") || !attr.contains("value")) {
    REPORT_CALL_ERROR("E19999", "cur attr does not contain name or dtype or value.");
    return;
  }
  std::string attr_name;
  std::string dtype;
  attr_name = attr["name"].get<std::string>();
  dtype = attr["dtype"].get<std::string>();
  auto iter = parse_attr_dtype_map.find(dtype);
  if (iter == parse_attr_dtype_map.end()) {
    REPORT_CALL_ERROR("E19999", "Unknown dtype[%s], which is unsupported.", dtype.c_str());
    return;
  }
  ParseAndSetAttrValuePtr func_ptr = iter->second;
  if (func_ptr == nullptr) {
    GE_LOGE("ParseAndSetAttrValueFunc ptr cannot be null!");
    return;
  }
  (*func_ptr)(op, attr, attr_name);
  GELOGD("Finish to set attr[name: %s] to Operator.", attr_name.c_str());
}

void ParseShapeDescListV2(const nlohmann::json &shape_list, ge::OpDescPtr &op_desc, const bool &is_input) {
  for (const auto &elem : shape_list) {
    if (elem.is_array()) {
      for (const auto &shape : elem) {
        ParseShapeDescV2(shape, op_desc, is_input);
      }
    } else {
      ParseShapeDescV2(elem, op_desc, is_input);
    }
  }
}

void ParseAndSetAttrsList(const nlohmann::json &attrs_list, ge::Operator &op) {
  for (const auto &attr : attrs_list) {
    ParseAndSetAttr(attr, op);
  }
}

template<typename T>
void GetConstDataPointer(const nlohmann::json &json_array, std::vector<uint8_t> &const_value) {
  std::vector<T> value = json_array.get<std::vector<T>>();
  uint8_t *pv_begin = reinterpret_cast<uint8_t *>(value.data());
  uint8_t *pv_end = pv_begin + (value.size() * sizeof(T));
  const_value = std::vector<uint8_t>(pv_begin, pv_end);
}

void CopyConstDataWithFloat16(const nlohmann::json &json_array, std::vector<uint8_t> &value) {
  std::vector<float> const_value = json_array.get<std::vector<float>>();
  float *const_data_ptr = const_value.data();
  if (const_data_ptr == nullptr) {
    GE_LOGE("Get const data pointer failed");
    return;
  }
  std::vector<uint16_t> const_data_vec;
  const size_t size = sizeof(const_value)/sizeof(float);
  for (size_t i = 0; i < size; ++i) {
    float const_data = *(const_data_ptr + i);
    uint16_t const_data_uint16 = optiling::FloatToUint16(const_data);
    const_data_vec.emplace_back(const_data_uint16);
  }
  uint8_t *pv_begin = reinterpret_cast<uint8_t *>(const_data_vec.data());
  uint8_t *pv_end = pv_begin + (const_data_vec.size() * sizeof(uint16_t));
  value = std::vector<uint8_t>(pv_begin, pv_end);
}

bool CopyConstData(const std::string &dtype, const nlohmann::json &json_array, std::vector<uint8_t> &value) {
  if (dtype == "int8") {
    GetConstDataPointer<int8_t>(json_array, value);
  } else if (dtype == "uint8") {
    GetConstDataPointer<uint8_t>(json_array, value);
  } else if (dtype == "int16") {
    GetConstDataPointer<int16_t>(json_array, value);
  } else if (dtype == "uint16") {
    GetConstDataPointer<uint16_t>(json_array, value);
  } else if (dtype == "int32") {
    GetConstDataPointer<int32_t>(json_array, value);
  } else if (dtype == "uint32") {
    GetConstDataPointer<uint32_t>(json_array, value);
  } else if (dtype == "int64") {
    GetConstDataPointer<int64_t>(json_array, value);
  } else if (dtype == "uint64") {
    GetConstDataPointer<uint64_t>(json_array, value);
  } else if (dtype == "float32") {
    GetConstDataPointer<float>(json_array, value);
  } else if (dtype == "double") {
    GetConstDataPointer<double>(json_array, value);
  } else if (dtype == "float16") {
    CopyConstDataWithFloat16(json_array, value);
  } else {
    GE_LOGE("Unknown dtype: %s", dtype.c_str());
    return false;
  }
  return true;
}

void ParseConstShapeDesc(const nlohmann::json &shape_json, std::map<std::string, TeConstTensorData> &const_tensors,
                         std::map<std::string, std::vector<uint8_t>> &const_values) {
  std::vector<int64_t> shape;
  std::string format_str;
  std::string dtype_str;

  if (!shape_json.contains("const_value")) {
    GELOGI("Not const tenosr");
    return;
  }
  if (!shape_json.contains("name")) {
    GE_LOGE("const tensor has no name");
    return;
  }
  std::string name = shape_json["name"];

  if (shape_json.contains("shape")) {
    shape = shape_json["shape"].get<std::vector<int64_t>>();
  }
  if (shape_json.contains("format")) {
    format_str = shape_json["format"].get<std::string>();
  }
  if (shape_json.contains("dtype")) {
    dtype_str = shape_json["dtype"].get<std::string>();
  }

  std::vector<uint8_t> value;
  const bool bres = CopyConstData(dtype_str, shape_json["const_value"], value);
  if (!bres) {
    GE_LOGE("CopyConstData faild.  buffer is null");
    return;
  }
  auto res = const_values.emplace(name, std::move(value));
  if (res.first == const_values.end()) {
    return;  // CodeDEX complains 'CHECK_CONTAINER_EMPTY'
  }

  ge::Shape ge_shape(shape);
  std::transform(dtype_str.begin(), dtype_str.end(), dtype_str.begin(), ::toupper);
  dtype_str = "DT_" + dtype_str;
  ge::DataType ge_dtype = ge::TypeUtils::SerialStringToDataType(dtype_str);
  std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::toupper);
  ge::Format ge_format = ge::TypeUtils::SerialStringToFormat(format_str);
  ge::Tensor const_tensor(ge::TensorDesc(ge_shape, ge_format, ge_dtype), res.first->second);
  const_tensors.emplace(name, std::make_tuple(const_tensor.GetData(), const_tensor.GetSize(), const_tensor));
  return;
}

void ParseConstTensorList(const nlohmann::json &shape_list, std::map<std::string, TeConstTensorData> &const_tensors,
                          std::map<std::string, std::vector<uint8_t>> &const_values) {
  for (const auto &elem : shape_list) {
    if (elem.is_array()) {
      for (const auto &shape : elem) {
        ParseConstShapeDesc(shape, const_tensors, const_values);
      }
    } else {
      ParseConstShapeDesc(elem, const_tensors, const_values);
    }
  }
}

void ParseConstShapeDescV2(const nlohmann::json &shape_json, ge::Operator &op_para,
                           std::map<std::string, std::vector<uint8_t>> &const_values) {
  std::vector<int64_t> shape;
  std::string format_str;
  std::string dtype_str;

  if (!shape_json.contains("const_value")) {
    GELOGI("Not const tenosr");
    return;
  }
  if (!shape_json.contains("name")) {
    REPORT_CALL_ERROR("E19999", "const tensor has no name");
    return;
  }
  std::string name = shape_json["name"];

  if (shape_json.contains("shape")) {
    shape = shape_json["shape"].get<std::vector<int64_t>>();
  }
  if (shape_json.contains("format")) {
    format_str = shape_json["format"].get<std::string>();
  }
  if (shape_json.contains("dtype")) {
    dtype_str = shape_json["dtype"].get<std::string>();
  }

  std::vector<uint8_t> value;
  const bool bres = CopyConstData(dtype_str, shape_json["const_value"], value);
  if (!bres) {
    REPORT_CALL_ERROR("E19999", "CopyConstData faild.  buffer is null");
    return;
  }
  auto res = const_values.emplace(name, std::move(value));
  if (res.first == const_values.end()) {
    return;  // CodeDEX complains 'CHECK_CONTAINER_EMPTY'
  }

  const ge::GeShape ge_shape(shape);
  ge::DataType ge_dtype = ge::DT_UNDEFINED;
  if (!dtype_str.empty()) {
    std::transform(dtype_str.begin(), dtype_str.end(), dtype_str.begin(), ::toupper);
    dtype_str = "DT_" + dtype_str;
    ge_dtype = ge::TypeUtils::SerialStringToDataType(dtype_str);
  }
  ge::Format ge_format = ge::FORMAT_RESERVED;
  if (!format_str.empty()) {
    std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::toupper);
    ge_format = ge::TypeUtils::SerialStringToFormat(format_str);
  }
  ge::GeTensorDesc ge_tensor(ge_shape, ge_format, ge_dtype);
  ge_tensor.SetName(name);
  ge::GeTensor const_tensor(ge_tensor, res.first->second);
  ge::GeTensorPtr const_tensor_ptr = std::make_shared<ge::GeTensor>(const_tensor);
  ge::OpDescPtr const_op_desc = ge::OpDescUtils::CreateConstOp(const_tensor_ptr);
  ge::Operator const_op = ge::OpDescUtils::CreateOperatorFromOpDesc(const_op_desc);
  (void)op_para.SetInput(name.c_str(), const_op);
  return;
}

void ParseConstTensorListV2(const nlohmann::json &shape_list, ge::Operator &operator_para,
                            std::map<std::string, std::vector<uint8_t>> &const_values) {
  for (const auto &elem : shape_list) {
    if (elem.is_array()) {
      for (const auto &shape : elem) {
        ParseConstShapeDescV2(shape, operator_para, const_values);
      }
    } else {
      ParseConstShapeDescV2(elem, operator_para, const_values);
    }
  }
}

std::string DumpByteBuffer(const ByteBuffer &buf) {
  static const char hex_digits[] = "0123456789ABCDEF";
  std::string str = buf.str();
  std::string output;
  const uint32_t num_two = 2;
  const uint32_t num_four = 4;
  const uint32_t num_fifteen = 15;
  output.reserve(str.size() * num_two);
  for (unsigned char c : str) {
    output.push_back(hex_digits[c >> num_four]);
    output.push_back(hex_digits[c & num_fifteen]);
  }
  return output;
}

bool DumpRunInfo(const OpRunInfo &run_info, char *run_info_json, const size_t &run_info_len) {
  if (run_info_json == nullptr) {
    GE_LOGE("run_info buffer is null");
    return false;
  }

  nlohmann::json json_obj;
  json_obj["block_dim"] = run_info.block_dim;
  json_obj["workspaces"] = run_info.workspaces;
  json_obj["tiling_data"] = DumpByteBuffer(run_info.tiling_data);
  json_obj["clear_atomic"] = run_info.clear_atomic;
  json_obj["tiling_key"] = run_info.tiling_key;

  const std::string str = json_obj.dump();
  if (str.size() >= run_info_len) {
    GE_LOGE("runinfo too large. %zu/%zu", str.size(), run_info_len);
    return false;
  }
  return memcpy_s(run_info_json, str.size() + 1, str.c_str(), str.size() + 1) == EOK;
}

bool DumpRunInfoV2(const OpRunInfoV2 &run_info, char *run_info_json, const size_t &run_info_len) {
  if (run_info_json == nullptr) {
    REPORT_CALL_ERROR("E19999", "run_info buffer is null");
    return false;
  }

  nlohmann::json json_obj;
  std::vector<int64_t> workspaces;
  int64_t workspace;
  for (size_t i = 0; i < run_info.GetWorkspaceNum(); ++i) {
    (void) run_info.GetWorkspace(i, workspace);
    workspaces.push_back(workspace);
  }
  json_obj["block_dim"] = run_info.GetBlockDim();
  json_obj["workspaces"] = workspaces;
  json_obj["tiling_data"] = DumpByteBuffer(run_info.GetAllTilingData());
  json_obj["clear_atomic"] = run_info.GetClearAtomic();
  json_obj["tiling_key"] = run_info.GetTilingKey();

  const std::string str = json_obj.dump();
  if (str.size() >= run_info_len) {
    REPORT_CALL_ERROR("E19999", "runinfo too large. %zu/%zu", str.size(), run_info_len);
    return false;
  }
  return memcpy_s(run_info_json, str.size() + 1, str.c_str(), str.size() + 1) == EOK;
}

extern "C" int TbeOpTilingPyInterfaceEx2BackUp(const char *optype, const char *compile_info, const char *inputs,
                                               const char *outputs, char *run_info_json, size_t run_info_len,
                                               const char *compile_info_hash, uint64_t *elapse,
                                               const OpTilingFunc &tiling_func) {
  if ((optype == nullptr) || (compile_info == nullptr) || (inputs == nullptr) || (outputs == nullptr)) {
    REPORT_CALL_ERROR("E19999", "optype/compile_info/inputs/outputs is null, %s, %s, %s, %s", optype, compile_info,
                      inputs, outputs);
    return 0;
  }

  std::chrono::time_point<std::chrono::steady_clock> before_tiling;
  std::chrono::time_point<std::chrono::steady_clock> after_tiling;
  TeOpParas op_params;
  op_params.op_type = optype;
  std::map<std::string, std::vector<uint8_t>> const_values;
  try {
    const nlohmann::json inputs_json = nlohmann::json::parse(inputs);
    const nlohmann::json outputs_json = nlohmann::json::parse(outputs);
    ParseShapeDescList(inputs_json, op_params.inputs);
    ParseShapeDescList(outputs_json, op_params.outputs);
    ParseConstTensorList(inputs_json, op_params.const_inputs, const_values);
  } catch (...) {
    REPORT_CALL_ERROR("E19999", "Failed to parse json_str. %s, %s, %s", compile_info, inputs, outputs);
    return 0;
  }
  GELOGI("Optiling func found, op_type:%s", optype);

  OpCompileInfo op_compile_info{compile_info, ""};
  if (compile_info_hash != nullptr) {
    op_compile_info.key = compile_info_hash;
  }

  OpRunInfo run_info;
  if (elapse != nullptr) {
    before_tiling = std::chrono::steady_clock::now();
  }

  const bool rc = (tiling_func)(op_params, op_compile_info, run_info);

  if (elapse != nullptr) {
    after_tiling = std::chrono::steady_clock::now();
  }
  if (!rc) {
    GELOGW("Optiling failed. op_type:%s", optype);
    return 0;
  }

  if (elapse != nullptr) {
    *elapse = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(\
        after_tiling - before_tiling).count());
    *(elapse + 1) = static_cast<uint64_t>(last_op_tiling_perf);
    last_op_tiling_perf = -1;
  }

  GELOGI("Optiling succeed. op_type:%s", optype);
  (void)DumpRunInfo(run_info, run_info_json, run_info_len);
  return 1;
}

void CheckAndSetAttr(const char *attrs, ge::Operator &operator_param) {
  if (attrs != nullptr) {
    GELOGD("Attrs set from pyAPI is: %s", attrs);
    const nlohmann::json attrs_json = nlohmann::json::parse(attrs);
    ParseAndSetAttrsList(attrs_json, operator_param);
  } else {
    GELOGD("Attrs has not been set.");
  }
  return;
}

void ParseInputsAndOutputs(const char *inputs, const char *outputs, ge::OpDescPtr &op_desc,
    ge::Operator &operator_param, std::map<std::string, std::vector<uint8_t>> &const_values) {
  const nlohmann::json inputs_json = nlohmann::json::parse(inputs);
  const nlohmann::json outputs_json = nlohmann::json::parse(outputs);
  ParseShapeDescListV2(inputs_json, op_desc, true);
  ParseShapeDescListV2(outputs_json, op_desc, false);
  operator_param = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  ParseConstTensorListV2(inputs_json, operator_param, const_values);
}

extern "C" int TbeOpTilingPyInterfaceEx2New(const char *optype, const char *compile_info, const char *inputs,
                                            const char *outputs, char *run_info_json, size_t run_info_len,
                                            const char *compile_info_hash, uint64_t *elapse,
                                            const OpTilingFuncV2 &tiling_func,
                                            const char *attrs) {
  if ((optype == nullptr) || (compile_info == nullptr) || (inputs == nullptr) || (outputs == nullptr)) {
    REPORT_CALL_ERROR("E19999", "optype/compile_info/inputs/outputs is null, %s, %s, %s, %s", optype, compile_info,
                      inputs, outputs);
    return 0;
  }
  GELOGI("Optiling func v2 found, op_type:%s", optype);

  std::chrono::time_point<std::chrono::steady_clock> before_tiling;
  std::chrono::time_point<std::chrono::steady_clock> after_tiling;
  const std::string compile_info_str = compile_info;
  std::string optype_str = optype;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("", optype_str);
  std::map<std::string, std::vector<uint8_t>> const_values;
  ge::Operator operator_param;
  try {
    ParseInputsAndOutputs(inputs, outputs, op_desc, operator_param, const_values);
    CheckAndSetAttr(attrs, operator_param);
  } catch (...) {
    REPORT_CALL_ERROR("E19999", "Failed to parse json_str. %s, %s, %s", compile_info, inputs, outputs);
    return 0;
  }

  OpCompileInfoV2 op_compile_info{" ", compile_info_str};
  const ge::AscendString opCompileInfoHash(compile_info_hash);
  if (compile_info_hash != nullptr) {
    op_compile_info.SetKey(opCompileInfoHash);
  }

  OpRunInfoV2 run_info(static_cast<uint32_t>(0), false, static_cast<uint64_t>(0));
  if (elapse != nullptr) {
    before_tiling = std::chrono::steady_clock::now();
  }

  const bool rc = (tiling_func)(operator_param, op_compile_info, run_info);

  if (elapse != nullptr) {
    after_tiling = std::chrono::steady_clock::now();
  }
  if (!rc) {
    GELOGW("Optiling failed. op_type:%s", optype);
    return 0;
  }

  if (elapse != nullptr) {
    *elapse = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(\
        after_tiling - before_tiling).count());
    *(elapse + 1) = static_cast<uint64_t>(last_op_tiling_perf);
    last_op_tiling_perf = -1;
  }

  GELOGI("Op tiling v2 succeed. op_type:%s", optype);
  (void)DumpRunInfoV2(run_info, run_info_json, run_info_len);
  return 1;
}

extern "C" int TbeOpTilingPyInterfaceEx3(const char *optype, const char *compile_info, const char *inputs,
                                         const char *outputs, char *run_info_json, size_t run_info_len,
                                         const char *compile_info_hash, uint64_t *elapse,
                                         const OpTilingFuncV3 &tiling_func, const OpParseFuncV3 &parse_func,
                                         const char *attrs) {
  if ((optype == nullptr) || (compile_info == nullptr) || (inputs == nullptr) || (outputs == nullptr)) {
    REPORT_CALL_ERROR("E19999", "optype/compile_info/inputs/outputs is null, %s, %s, %s, %s", optype, compile_info,
                      inputs, outputs);
    return 0;
  }
  GELOGI("Optiling func v3 found, op_type:%s", optype);

  std::chrono::time_point<std::chrono::steady_clock> before_tiling;
  std::chrono::time_point<std::chrono::steady_clock> after_tiling;
  std::string optype_str = optype;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("", optype_str);
  std::map<std::string, std::vector<uint8_t>> const_values;
  ge::Operator operator_param;
  try {
    ParseInputsAndOutputs(inputs, outputs, op_desc, operator_param, const_values);
    CheckAndSetAttr(attrs, operator_param);
  } catch (...) {
    REPORT_CALL_ERROR("E19999", "Failed to parse json_str. %s, %s, %s", compile_info, inputs, outputs);
    return 0;
  }
  if (compile_info_hash == nullptr) {
    return 0;
  }

  const ge::AscendString compile_info_json_str = compile_info;
  void* op_compile_json_ptr = (parse_func)(operator_param, compile_info_json_str);

  OpRunInfoV2 run_info(static_cast<uint32_t>(0), false, static_cast<uint64_t>(0));
  if (elapse != nullptr) {
    before_tiling = std::chrono::steady_clock::now();
  }
  const bool rc = (tiling_func)(operator_param, op_compile_json_ptr, run_info);

  if (elapse != nullptr) {
    after_tiling = std::chrono::steady_clock::now();
  }
  if (!rc) {
    GELOGW("Optiling failed. op_type:%s", optype);
    return 0;
  }

  if (elapse != nullptr) {
    *elapse = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>\
        (after_tiling - before_tiling).count());
    *(elapse + 1) = static_cast<uint64_t>(last_op_tiling_perf);
    last_op_tiling_perf = -1;
  }

  GELOGI("Op tiling v3 succeed. op_type:%s", optype);
  (void)DumpRunInfoV2(run_info, run_info_json, run_info_len);
  return 1;
}

extern "C" int TbeOpTilingPyInterfaceEx4(const char *optype, const char *compile_info, const char *inputs,
                                         const char *outputs, char *run_info_json, size_t run_info_len,
                                         const char *compile_info_hash, uint64_t *elapse,
                                         const OpTilingFuncV4 &tiling_func, const OpParseFuncV4 &parse_func,
                                         const char *attrs) {
  if ((optype == nullptr) || (compile_info == nullptr) || (inputs == nullptr) || (outputs == nullptr)) {
    REPORT_CALL_ERROR("E19999", "optype/compile_info/inputs/outputs is null, %s, %s, %s, %s", optype, compile_info,
                      inputs, outputs);
    return 0;
  }
  GELOGI("Optiling func v4 found, op_type:%s", optype);

  std::chrono::time_point<std::chrono::steady_clock> before_tiling;
  std::chrono::time_point<std::chrono::steady_clock> after_tiling;
  std::string op_type_str = optype;
  ge::OpDescPtr op_desc_ptr = std::make_shared<ge::OpDesc>("", op_type_str);
  std::map<std::string, std::vector<uint8_t>> const_values;
  ge::Operator operator_param;
  try {
    ParseInputsAndOutputs(inputs, outputs, op_desc_ptr, operator_param, const_values);
    CheckAndSetAttr(attrs, operator_param);
  } catch (...) {
    REPORT_CALL_ERROR("E19999", "Failed to parse json during tiling v4. %s, %s, %s", compile_info, inputs, outputs);
    return 0;
  }
  if (compile_info_hash == nullptr) {
    return 0;
  }

  const ge::AscendString compile_info_json = compile_info;
  const CompileInfoPtr op_compile_json_ptr = (parse_func)(operator_param, compile_info_json);

  OpRunInfoV2 run_info(static_cast<uint32_t>(0), false, static_cast<uint64_t>(0));
  if (elapse != nullptr) {
    before_tiling = std::chrono::steady_clock::now();
  }
  const bool rc = (tiling_func)(operator_param, op_compile_json_ptr, run_info);

  if (elapse != nullptr) {
    after_tiling = std::chrono::steady_clock::now();
  }
  if (!rc) {
    GELOGW("Optiling failed. op_type:%s", optype);
    return 0;
  }

  if (elapse != nullptr) {
    *elapse = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(\
        after_tiling - before_tiling).count());
    *(elapse + 1) = static_cast<uint64_t>(last_op_tiling_perf);
    last_op_tiling_perf = -1;
  }

  GELOGI("Op tiling v4 succeed. op_type:%s", optype);
  (void)DumpRunInfoV2(run_info, run_info_json, run_info_len);
  return 1;
}

gert::KernelContextHolder BuildTilingParseContextHolder(ge::OpDescPtr &op_desc, const char *compile_info,
                                                        const char *op_type,
                                                        const gert::OpImplRegistry::OpImplFunctions *funcs) {
  std::vector<std::pair<void *, gert::Chain::Deleter>> tiling_parse_outputs(1, std::make_pair(nullptr, nullptr));
  if (op_desc->GetType() != OP_TYPE_AUTO_TILING) {
    tiling_parse_outputs[0].first = funcs->compile_info_creator();
    tiling_parse_outputs[0].second = funcs->compile_info_deleter;
  }

  return gert::KernelRunContextBuilder()
      .Inputs({std::make_pair(const_cast<char *>(compile_info), nullptr),
               std::make_pair(const_cast<char *>(op_type), nullptr)})
      .Outputs(tiling_parse_outputs)
      .Build(op_desc);
}

gert::KernelContextHolder BuildTilingContext(ContextComponent &context_com, gert::KernelContext *tiling_parse_context) {
  std::vector<void *> tiling_context_inputs(context_com.storage_shapes.size() + kSize, nullptr);
  for (size_t i = 0UL; i < context_com.index_to_tensors.size(); ++i) {
    tiling_context_inputs[context_com.index_to_tensors[i].first] =
        reinterpret_cast<gert::Tensor *>(context_com.index_to_tensors[i].second.get());
  }
  for (size_t i = 0UL; i < context_com.storage_shapes.size(); ++i) {
    if (tiling_context_inputs[i] == nullptr) {
      tiling_context_inputs[i] = &context_com.storage_shapes[i];
    }
  }
  tiling_context_inputs[context_com.storage_shapes.size()] = *tiling_parse_context->GetOutputPointer<void **>(0);
  return gert::KernelRunContextBuilder()
      .Inputs(tiling_context_inputs)
      .Outputs(
      {nullptr, nullptr, &context_com.atomic_flag, context_com.tiling_data.get(), context_com.workspace_size.get()})
      .Build(context_com.op_desc);
}

ge::graphStatus DoTilingParse(const gert::OpImplRegistry::OpImplFunctions *funcs,
                              gert::KernelContextHolder &tiling_parse_context_holder) {
  GE_CHECK_NOTNULL(tiling_parse_context_holder.context_);
  return (funcs->tiling_parse)(tiling_parse_context_holder.context_);
}

ge::graphStatus DoTilingWithTiming(const gert::OpImplRegistry::OpImplFunctions *funcs, uint64_t *elapse,
                                   gert::KernelContextHolder &tiling_context_holder) {
  GE_CHECK_NOTNULL(tiling_context_holder.context_);
  // calcu tiling cost time
  std::chrono::time_point<std::chrono::steady_clock> before_tiling;
  std::chrono::time_point<std::chrono::steady_clock> after_tiling;
  if (elapse != nullptr) {
    before_tiling = std::chrono::steady_clock::now();
  }
  const auto ret = (funcs->tiling)(reinterpret_cast<gert::TilingContext *>(tiling_context_holder.context_));
  if (elapse != nullptr) {
    after_tiling = std::chrono::steady_clock::now();
  }
  if (ret != ge::GRAPH_SUCCESS) {
    return ret;
  }

  if (elapse != nullptr) {
    *elapse = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(after_tiling - before_tiling).count());
    *(elapse + 1) = static_cast<uint64_t>(last_op_tiling_perf);
    last_op_tiling_perf = -1;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ParseJson(const char *inputs, const char *outputs, const char *attrs, ContextComponent &context_com) {
  if (ParseInputs(inputs, context_com.op_desc, context_com.storage_shapes, context_com.index_to_tensors) !=
      ge::GRAPH_SUCCESS) {
    GELOGE(ge::GRAPH_FAILED, "Parse inputs failed.");
    REPORT_CALL_ERROR("E19999", "Parse inputs failed.");
    return ge::GRAPH_FAILED;
  }
  if (ParseOutputs(outputs, context_com.op_desc, context_com.storage_shapes) != ge::GRAPH_SUCCESS) {
    GELOGE(ge::GRAPH_FAILED, "Parse outputs failed.");
    REPORT_CALL_ERROR("E19999", "Parse outputs failed.");
    return ge::GRAPH_FAILED;
  }
  if (ParseAttrs(attrs, context_com.op_desc) != ge::GRAPH_SUCCESS) {
    GELOGE(ge::GRAPH_FAILED, "Parse attrs failed.");
    REPORT_CALL_ERROR("E19999", "Parse attrs failed.");
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

int TbeOptilingPyInterfaceNew(const char *op_type, const char *compile_info, const char *inputs, const char *outputs,
                              char *run_info_json, size_t run_info_len, uint64_t *elapse, const char *attrs) {
  if ((compile_info == nullptr) || (inputs == nullptr) || (outputs == nullptr)) {
    GELOGE(ge::GRAPH_FAILED, "compile_info/inputs/outputs is null.");
    REPORT_CALL_ERROR("E19999", "compile_info/inputs/outputs is null.");
    return 0;
  }

  const gert::OpImplRegistry::OpImplFunctions *funcs;
  if (!FindImplFuncs(op_type, funcs)) {
    return 0;
  }
  ContextComponent context_com {};
  context_com.op_desc = std::make_shared<ge::OpDesc>("", op_type);
  if (context_com.op_desc == nullptr) {
    return 0;
  }
  if (ParseJson(inputs, outputs, attrs, context_com) != ge::GRAPH_SUCCESS) {
    return 0;
  }
  // tiling parse
  auto tiling_parse_context_holder = BuildTilingParseContextHolder(context_com.op_desc, compile_info, op_type, funcs);
  if (DoTilingParse(funcs, tiling_parse_context_holder) != ge::GRAPH_SUCCESS) {
    GELOGE(ge::GRAPH_FAILED, "Op %s tiling parse failed", op_type);
    REPORT_CALL_ERROR("E19999", "Op %s tiling parse failed", op_type);
    return 0;
  }

  // tiling
  int64_t max_size = -1;
  if (!ge::AttrUtils::GetInt(context_com.op_desc, kMaxTilingSize, max_size) || max_size < 0) {
    GELOGI("No max tiling size in opdesc.");
    max_size = static_cast<int64_t>(kMaxTilingDataSize);
  }
  auto aligned_max_size = ge::RoundUp(static_cast<uint64_t>(max_size), sizeof(uintptr_t));
  context_com.tiling_data = gert::TilingData::CreateCap(aligned_max_size);
  context_com.workspace_size = gert::ContinuousVector::Create<size_t>(kWorkspaceHolerSize);
  gert::KernelContextHolder tiling_context_holder =
      BuildTilingContext(context_com, tiling_parse_context_holder.context_);
  if (DoTilingWithTiming(funcs, elapse, tiling_context_holder) != ge::GRAPH_SUCCESS) {
    GELOGE(ge::GRAPH_FAILED, "Op %s tiling failed", op_type);
    REPORT_CALL_ERROR("E19999", "Op %s tiling failed", op_type);
    return 0;
  }

  if (!DumpRunInfo(tiling_context_holder.context_, run_info_json, run_info_len)) {
    GELOGE(ge::GRAPH_FAILED, "Dump op %s tiling result failed", op_type);
    REPORT_CALL_ERROR("E19999", "Dump op %s tiling result failed", op_type);
    return 0;
  }
  GELOGI("Op tiling suceed. op_type:%s", op_type);
  return 1;
}

extern "C" int TbeOpTilingPyInterfaceOld(const char *optype, const char *compile_info, const char *compile_info_hash,
                                         const char *inputs, const char *outputs, const char *attrs,
                                         char *run_info_json, size_t run_info_len, uint64_t *elapse) {
  auto &op_func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  auto iter = op_func_map.find(optype);
  if (iter == op_func_map.end()) {
    GELOGI("Op tiling function is not found by op type[%s].", optype);
    iter = op_func_map.find(OP_TYPE_AUTO_TILING);
    if (iter == op_func_map.end()) {
      GELOGI("Optiling func of op type[%s] is not found by Autotiling.", optype);
      REPORT_CALL_ERROR("E19999", "Optiling func is not found. op_type:%s", optype);
      return static_cast<int32_t>(ge::GRAPH_FAILED);
    }
  }
  OpTilingFuncInfo &op_func_info = iter->second;
  int ret = 0;
  if (op_func_info.IsFunctionV4()) {
    const OpTilingFuncV4 &tiling_func = op_func_info.GetOpTilingFuncV4();
    const OpParseFuncV4 &parse_func = op_func_info.GetOpParseFuncV4();
    ret = TbeOpTilingPyInterfaceEx4(optype, compile_info, inputs, outputs, run_info_json, run_info_len,
                                    compile_info_hash, elapse, tiling_func, parse_func, attrs);
  } else if (op_func_info.IsFunctionV3()) {
    const OpTilingFuncV3 &tiling_func = op_func_info.GetOpTilingFuncV3();
    const OpParseFuncV3 &parse_func = op_func_info.GetOpParseFuncV3();
    ret = TbeOpTilingPyInterfaceEx3(optype, compile_info, inputs, outputs, run_info_json, run_info_len,
                                    compile_info_hash, elapse, tiling_func, parse_func, attrs);
  } else if (op_func_info.IsFunctionV2()) {
    const OpTilingFuncV2  &tiling_func = op_func_info.GetOpTilingFuncV2();
    ret = TbeOpTilingPyInterfaceEx2New(optype, compile_info, inputs, outputs, run_info_json, run_info_len,
                                       compile_info_hash, elapse, tiling_func, attrs);
  } else if (op_func_info.IsFunctionV1()) {
    const OpTilingFunc  &tiling_func = op_func_info.GetOpTilingFunc();
    ret = TbeOpTilingPyInterfaceEx2BackUp(optype, compile_info, inputs, outputs, run_info_json,
                                          run_info_len, compile_info_hash, elapse, tiling_func);
  } else {
    GE_LOGE("Optiling func of op type[%s] is all empty.", optype);
  }
  return ret;
}

extern "C" int TbeOpTilingPyInterface(const char *optype, const char *compile_info, const char *compile_info_hash,
                                      const char *inputs, const char *outputs, const char *attrs, char *run_info_json,
                                      size_t run_info_len, uint64_t *elapse) {
  if (optype == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "op type is null.");
    REPORT_CALL_ERROR("E19999", "op type is null.");
    return 0;
  }

  // compatible some non-switching auto tiling, which will be deleted
  if (EnableGert()) {
    GELOGD("New tiling interface based gert");
    return TbeOptilingPyInterfaceNew(optype, compile_info, inputs, outputs, run_info_json, run_info_len, elapse, attrs);
  } else {
    return TbeOpTilingPyInterfaceOld(optype, compile_info, compile_info_hash, inputs, outputs, attrs, run_info_json,
                                     run_info_len, elapse);
  }
}

extern "C" int TbeOpTilingPyInterfaceEx2(const char *optype, const char *compile_info, const char *inputs,
                                         const char *outputs, char *run_info_json, size_t run_info_len,
                                         const char *compile_info_hash, uint64_t *elapse) {
  return TbeOpTilingPyInterface(optype, compile_info, compile_info_hash, inputs, outputs, nullptr,
                                run_info_json, run_info_len, elapse);
}
}  // namespace optiling