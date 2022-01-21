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
#ifndef MINDSPORE_INCLUDE_API_MODEL_MODEL_POOL_H
#define MINDSPORE_INCLUDE_API_MODEL_MODEL_POOL_H
#ifdef USING_SERVING
#include <vector>
#include <memory>
#include <utility>
#include <string>
#include <queue>
#include <map>
#include "include/api/status.h"
#include "include/api/context.h"
#include "src/cxx_api/model/model_thread.h"
#include "src/cxx_api/model/predict_task_queue.h"
namespace mindspore {
class ModelPool {
 public:
  static ModelPool *GetInstance();
  ~ModelPool();

  Status Init(const std::string &model_path, const std::string &config_path, const Key &dec_key = {},
              const std::string &dec_mode = kDecModeAesGcm);

  std::vector<MSTensor> GetInputs();

  Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                 const MSKernelCallBack &before = nullptr, const MSKernelCallBack &after = nullptr);

 private:
  ModelPool() = default;
  Status InitContext(const std::shared_ptr<mindspore::Context> &context,
                     std::map<std::string, std::map<std::string, std::string>> *all_config_info);
  void Run(std::shared_ptr<ModelThread> model);
  void SetBindStrategy(std::vector<std::vector<int>> *all_model_bind_list, int thread_num);
  ModelPoolContex CreateModelContext(const std::string &config_path);

  std::vector<std::thread> model_thread_vec_;
  std::vector<MSTensor> model_inputs_;
  size_t num_models_ = 5;
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_MODEL_MODEL_POOL_H
#endif
