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

#ifndef METADEF_GRAPH_PARALLELISM_COMM_TASK_BUILDER_H_
#define METADEF_GRAPH_PARALLELISM_COMM_TASK_BUILDER_H_

#include "graph/parallelism/tensor_parallel_attrs.h"
#include "nlohmann/json.hpp"

namespace ge {
namespace tp {
class CommTaskBuilder {
 public:
  static CommTaskBuilder &GetInstance() {
    static CommTaskBuilder instance;
    return instance;
  }

  void BuildCommTask(const nlohmann::json &j, CommTask &comm_task);
  Status ConvertToJson(const CommTask &comm_task, nlohmann::json &j);

 private:
  CommTaskBuilder();
  ~CommTaskBuilder() = default;

  void InitCommTaskBuilders();
  void InitJsonConverters();
  template<typename T>
  static Status ConvertToJson(const T *reshard_task, nlohmann::json &j);

  std::map<std::string, std::function<void(const nlohmann::json &, CommTask &)>> builders_;
  std::map<std::string, std::function<Status(const CommTask &, nlohmann::json &)>> json_converters_;
};
}  // namespace tp
}  // namespace ge

#endif  // METADEF_GRAPH_PARALLELISM_COMM_TASK_BUILDER_H_
