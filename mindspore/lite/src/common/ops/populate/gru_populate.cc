/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "src/common/ops/populate/populate_register.h"
#include "nnacl/fp32/gru_fp32.h"
using mindspore::schema::PrimitiveType_GRU;

namespace mindspore {
namespace lite {
OpParameter *PopulateGruParameter(const void *prim) {
  auto *primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_GRU();
  if (value == nullptr) {
    MS_LOG(ERROR) << "param is nullptr.";
    return nullptr;
  }

  auto *param = reinterpret_cast<GruParameter *>(malloc(sizeof(GruParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc GruParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(GruParameter));

  param->op_parameter_.type_ = primitive->value_type();
  param->bidirectional_ = value->bidirectional();
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_GRU, PopulateGruParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
