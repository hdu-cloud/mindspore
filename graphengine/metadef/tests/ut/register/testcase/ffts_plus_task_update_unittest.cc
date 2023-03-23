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

#include "gtest/gtest.h"

#define private public
#define protected public
#include "register/ffts_plus_update_manager.h"
#include "common/plugin_so_manager.h"

namespace ge {
class FFTSPlusTaskUpdateStub : public FFTSPlusTaskUpdate {
 public:
  Status GetAutoThreadParam(const NodePtr &node, const std::vector<optiling::utils::OpRunInfo> &op_run_info,
                            AutoThreadParam &auto_thread_param) override {
    return SUCCESS;
  }

  Status UpdateSubTaskAndCache(const NodePtr &node, const AutoThreadSubTaskFlush &sub_task_flush,
                               rtFftsPlusTaskInfo_t &ffts_plus_task_info) override {
    return SUCCESS;
  }

  Status UpdateCommonCtx(const ComputeGraphPtr &sgt_graph, rtFftsPlusTaskInfo_t &task_info) override {
    return SUCCESS;
  }
};

class UtestFftsPlusUpdate : public testing::Test {
 protected:
  void SetUp() {
    const std::string kCoreTypeTest = "FFTS_TEST";     // FftsPlusUpdateManager::FftsPlusUpdateRegistrar
    REGISTER_FFTS_PLUS_CTX_UPDATER(kCoreTypeTest, FFTSPlusTaskUpdateStub);
  }

  void TearDown() {
    FftsPlusUpdateManager::Instance().creators_.clear();
    FftsPlusUpdateManager::Instance().plugin_manager_.reset();
    FftsPlusUpdateManager::Instance().is_init_ = false;
  }
};

TEST_F(UtestFftsPlusUpdate, Initialize) {
  EXPECT_FALSE(FftsPlusUpdateManager::Instance().is_init_);
  EXPECT_EQ(FftsPlusUpdateManager::Instance().Initialize(), SUCCESS);
  EXPECT_TRUE(FftsPlusUpdateManager::Instance().is_init_);
  EXPECT_EQ(FftsPlusUpdateManager::Instance().Initialize(), SUCCESS);
  EXPECT_TRUE(FftsPlusUpdateManager::Instance().is_init_);
}

TEST_F(UtestFftsPlusUpdate, GetUpdater) {
  EXPECT_EQ(FftsPlusUpdateManager::Instance().GetUpdater("AIC_AIV"), nullptr);
  EXPECT_NE(FftsPlusUpdateManager::Instance().GetUpdater("FFTS_TEST"), nullptr);
}
}
