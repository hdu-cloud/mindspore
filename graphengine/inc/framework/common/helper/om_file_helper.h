/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#ifndef INC_FRAMEWORK_COMMON_HELPER_OM_FILE_HELPER_H_
#define INC_FRAMEWORK_COMMON_HELPER_OM_FILE_HELPER_H_

#include <string>
#include <vector>

#include "external/ge/ge_ir_build.h"
#include "framework/common/types.h"
#include "framework/common/ge_types.h"

namespace ge {
struct ModelPartition {
  ModelPartitionType type;
  const uint8_t *data = nullptr;
  uint32_t size = 0U;
};

struct OmFileContext {
  std::vector<ModelPartition> partition_datas_;
  std::vector<char_t> partition_table_;
  uint32_t model_data_len_ = 0U;
};

struct SaveParam {
  int32_t encode_mode;
  std::string ek_file;
  std::string cert_file;
  std::string hw_key_file;
  std::string pri_key_file;
  std::string model_name;
};

class GE_FUNC_VISIBILITY OmFileLoadHelper {
 public:
  Status Init(const ModelData &model);

  Status Init(uint8_t *const model_data, const uint32_t model_data_size);

  Status Init(uint8_t *const model_data, const uint32_t model_data_size, const uint32_t model_num);

  Status GetModelPartition(const ModelPartitionType type, ModelPartition &partition);

  Status GetModelPartition(const ModelPartitionType type, ModelPartition &partition, const size_t model_index) const;

  OmFileContext context_;

  std::vector<OmFileContext> model_contexts_;

 private:
  Status LoadModelPartitionTable(uint8_t *const model_data, const uint32_t model_data_size, const size_t model_index,
                                 size_t &mem_offset);

  Status LoadModelPartitionTable(uint8_t *const model_data, const uint32_t model_data_size, const uint32_t model_num);

  bool is_inited_{false};
};

class GE_FUNC_VISIBILITY OmFileSaveHelper {
 public:
  ModelFileHeader &GetModelFileHeader() {
    return model_header_;
  }

  uint32_t GetModelDataSize() const;

  ModelPartitionTable *GetPartitionTable();

  Status AddPartition(const ModelPartition &partition);

  Status AddPartition(const ModelPartition &partition, const size_t cur_index);

  Status SaveModel(const SaveParam &save_param, const char_t *const output_file, ModelBufferData &model,
                   const bool is_offline = true);

  Status SaveModelToFile(const char_t *const output_file, ModelBufferData &model, const bool is_offline = true);

  ModelPartitionTable *GetPartitionTable(const size_t cur_ctx_index);

  Status SaveRootModel(const SaveParam &save_param, const char_t *const output_file, ModelBufferData &model,
                       const bool is_offline);

 private:
  ModelFileHeader model_header_;
  std::vector<OmFileContext> model_contexts_;
};
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_HELPER_OM_FILE_HELPER_H_
