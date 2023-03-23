/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_CONSTANTS_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_CONSTANTS_H_

#include <set>
#include <map>
#include <chrono>
#include <string>
#include <vector>

namespace mindspore {
namespace distributed {
// The detailed reason of failing to run 'mindspore.communication.init()' with ClusterContext.
constexpr char kDetailedFailureReason[] =
  "Maybe you are trying to call 'mindspore.communication.init()' without using 'mpirun', which will make MindSpore "
  "load several environment variables and check their validation. Please use 'mpirun' to launch this process to fix "
  "this issue, or refer to this link if you want to run distributed training without using 'mpirun': "
  "https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/distributed_training_gpu.html#openmpi.";

constexpr char kEnvServerNum[] = "MS_SERVER_NUM";
constexpr char kEnvWorkerNum[] = "MS_WORKER_NUM";
constexpr char kEnvSchedulerHost[] = "MS_SCHED_HOST";
constexpr char kEnvSchedulerPort[] = "MS_SCHED_PORT";

constexpr char kEnvRole[] = "MS_ROLE";
constexpr char kEnvRoleOfServer[] = "MS_SERVER";
constexpr char kEnvRoleOfPServer[] = "MS_PSERVER";
constexpr char kEnvRoleOfWorker[] = "MS_WORKER";
constexpr char kEnvRoleOfScheduler[] = "MS_SCHED";
const std::set<std::string> kValidRoleName = {kEnvRoleOfServer, kEnvRoleOfPServer, kEnvRoleOfWorker,
                                              kEnvRoleOfScheduler};

// Used in parameter server embedding cache scenarios to identify the same Parameter between Worker and Server.
constexpr char kParameterKey[] = "parameter_key";
// Embedding cache lookup operation.
constexpr char kLookupEmbeddingCache[] = "LookupEmbeddingCache";
// Embedding cache update operation.
constexpr char kUpdateEmbeddingCache[] = "UpdateEmbeddingCache";
const std::vector<std::string> kEmbeddingCacheOps = {kLookupEmbeddingCache, kUpdateEmbeddingCache};
// Message header of finalize mux recv actor.
constexpr char kFinalizeMuxRecvActor[] = "FINALIZE_MUX_RECV_ACTOR";

// The distributed execution mode enum.
// For each execution mode, different graph optimization, splitting strategy, device location, etc are applied. For
// details please refer to class DistributedExecutionMode and its subclasses.

// kGeneralMode: Simply split a training graph into multiple devices without other extra features.

// kParallelMode: MindSpore's existing auto-parallel feature along with distributed graph splitting feature are
// combined. This is much more complicated than other mode. It is always applied in MoE scenarios.

// kPSMode: Applied when running Parameter Server training.

// kEmbeddingCacheMode: Applied when embedding cache is enabled. Normally used for training models with large embedding
// layer.
enum class DistExecutionMode { kGeneralMode = 0, kParallelMode, kPSMode, kEmbeddingCacheMode, kInvalidMode };

// The operator's label in distributed execution.
constexpr char kOpLabelRankId[] = "rank_id";
constexpr char kOpLabelRole[] = "ms_role";

constexpr char kLocalHost[] = "127.0.0.1";
constexpr int MAX_HOSTNAME_LEN = 1024;
const uint16_t kDefaultSchedPort = 6667;
const uint16_t kMaxPort = 65535;
constexpr uint32_t kDefaultFinishTimeout = 30;

constexpr char kDataSyncSrcOpName[] = "DataSyncSrc";
constexpr char kDataSyncDstOpName[] = "DataSyncDst";
constexpr char kControlSrcOpName[] = "ControlSrc";
constexpr char kControlDstOpName[] = "ControlDst";

// This macro the current timestamp in milliseconds.
#define CURRENT_TIMESTAMP_MILLI \
  (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()))
}  // namespace distributed
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DISTRIBUTED_CONSTANTS_H_
