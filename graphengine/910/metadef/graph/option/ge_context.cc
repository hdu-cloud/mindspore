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

#include "graph/ge_context.h"
#include "graph/ge_global_options.h"
#include "graph/ge_local_context.h"
#include "graph/types.h"
#include "common/ge_common/debug/ge_log.h"
#include "utils/extern_math_util.h"
#include "nlohmann/json.hpp"

namespace ge {
using Json = nlohmann::json;

namespace {
const uint64_t kMinTrainingTraceJobId = 65536U;
const int32_t kDecimal = 10;
const char_t *kHostExecPlacement = "HOST";
const char_t *kEnabled = "1";

template<class T>
ge::Status GetOptionValue(const std::string &option_name, T &var) {
  std::string option;
  (void) ge::GetContext().GetOption(option_name, option);
  int64_t value = 0;
  try {
    value = static_cast<int64_t>(std::stoi(option.c_str()));
  } catch (std::invalid_argument &) {
    GELOGW("[Init] Transform option %s %s to int failed, as catching invalid_argument exception", option_name.c_str(),
           option.c_str());
    return ge::FAILED;
  } catch (std::out_of_range &) {
    GELOGW("[Init] Transform option %s %s to int failed, as catching out_of_range exception", option_name.c_str(),
           option.c_str());
    return ge::FAILED;
  }
  if (!IntegerChecker<T>::Compat(value)) {
    GELOGW("[Init] Transform option %s %s to int failed, value is invalid_argument", option_name.c_str(),
           option.c_str());
    return ge::FAILED;
  }
  var = value;
  return ge::SUCCESS;
}
}  // namespace
GEContext &GetContext() {
  static GEContext ge_context {};
  return ge_context;
}

thread_local uint64_t GEContext::session_id_ = 0UL;
thread_local uint64_t GEContext::context_id_ = 0UL;
thread_local int32_t GEContext::thread_device_id_ = -1;

graphStatus GEContext::GetOption(const std::string &key, std::string &option) {
  return GetThreadLocalContext().GetOption(key, option);
}

const std::string &GEContext::GetReadableName(const std::string &key) {
  auto iter = option_name_map_.find(key);
  if (iter != option_name_map_.end()) {
    GELOGD("Option %s's readable name is show name: %s", key.c_str(), iter->second.c_str());
    return iter->second;
  }
  GELOGD("Option %s's readable name is GE IR option: %s", key.c_str(), key.c_str());
  return key;
}

bool GEContext::IsOverflowDetectionOpen() const {
  std::string enable_overflow_detection;
  if (GetThreadLocalContext().GetOption("ge.exec.overflow", enable_overflow_detection) != GRAPH_SUCCESS) {
    GELOGD("can not get option ge.exec.overflow.");
    return false;
  }
  GELOGD("Option ge.exec.overflow is %s.", enable_overflow_detection.c_str());
  return (enable_overflow_detection == kEnabled);
}

bool GEContext::IsGraphLevelSat() const {
  std::string graph_level_sat;
  if (GetThreadLocalContext().GetOption("ge.graphLevelSat", graph_level_sat) != GRAPH_SUCCESS) {
    GELOGD("can not get option ge.graphLevelSat.");
    return false;
  }
  GELOGD("Option ge.graphLevelSat is %s.", graph_level_sat.c_str());
  return (graph_level_sat == kEnabled);
}

bool GEContext::GetHostExecFlag() const {
  std::string exec_placement;
  if (GetThreadLocalContext().GetOption("ge.exec.placement", exec_placement) != GRAPH_SUCCESS) {
    GELOGD("get option ge.exec.placement failed.");
    return false;
  }
  GELOGD("Option ge.exec.placement is %s.", exec_placement.c_str());
  return exec_placement == kHostExecPlacement;
}

bool GEContext::GetTrainGraphFlag() const {
  std::string run_mode;
  if ((GetThreadLocalContext().GetOption(ge::OPTION_GRAPH_RUN_MODE, run_mode) == ge::GRAPH_SUCCESS) &&
      (!run_mode.empty())) {
    const int32_t base = 10;
    if (static_cast<ge::GraphRunMode>(std::strtol(run_mode.c_str(), nullptr, base)) >= ge::TRAIN) {
      return true;
    }
  }
  return false;
}

std::mutex &GetGlobalOptionsMutex() {
  static std::mutex global_options_mutex;
  return global_options_mutex;
}

std::map<std::string, std::string> &GetMutableGlobalOptions() {
  static std::map<std::string, std::string> context_global_options{};
  return context_global_options;
}

void GEContext::Init() {
  (void) GetOptionValue("ge.exec.sessionId", session_id_);

  (void) GetOptionValue("ge.exec.deviceId", device_id_);

  std::string job_id;
  (void)GetOption("ge.exec.jobId", job_id);
  std::string s_job_id = "";
  for (const auto c : job_id) {
    if ((c >= '0') && (c <= '9')) {
      s_job_id += c;
    }
  }
  if (s_job_id == "") {
    trace_id_ = kMinTrainingTraceJobId;
    return;
  }
  const auto d_job_id = std::strtoll(s_job_id.c_str(), nullptr, kDecimal);
  if (static_cast<uint64_t>(d_job_id) < kMinTrainingTraceJobId) {
    trace_id_ = static_cast<uint64_t>(d_job_id) + kMinTrainingTraceJobId;
  } else {
    trace_id_ = static_cast<uint64_t>(d_job_id);
  }

  (void) GetOptionValue("stream_sync_timeout", stream_sync_timeout_);

  (void) GetOptionValue("event_sync_timeout", event_sync_timeout_);
}

uint64_t GEContext::SessionId() const { return session_id_; }

uint32_t GEContext::DeviceId() const { return device_id_; }

int32_t GEContext::StreamSyncTimeout() const { return stream_sync_timeout_; }

int32_t GEContext::EventSyncTimeout() const { return event_sync_timeout_; }

void GEContext::SetSessionId(const uint64_t session_id) { session_id_ = session_id; }

void GEContext::SetContextId(const uint64_t context_id) { context_id_ = context_id; }

void GEContext::SetCtxDeviceId(const uint32_t device_id) { device_id_ = device_id; }

void GEContext::SetStreamSyncTimeout(const int32_t timeout) { stream_sync_timeout_ = timeout; }

void GEContext::SetEventSyncTimeout(const int32_t timeout) { event_sync_timeout_ = timeout; }

graphStatus GEContext::SetOptionNameMap(const std::string &option_name_map_json) {
  Json option_json;
  try {
    option_json = Json::parse(option_name_map_json);
  } catch (nlohmann::json::parse_error&) {
    GELOGE(ge::GRAPH_FAILED, "Parse JsonStr to Json failed, JsonStr: %s", option_name_map_json.c_str());
    return ge::GRAPH_FAILED;
  }
  for (auto iter : option_json.items()) {
    if (iter.key().empty()) {
      GELOGE(ge::GRAPH_FAILED, "Check option_name_map failed, key is null");
      return ge::GRAPH_FAILED;
    }
    if (static_cast<std::string>(iter.value()).empty()) {
      GELOGE(ge::GRAPH_FAILED, "Check option_name_map failed, value is null");
      return ge::GRAPH_FAILED;
    }
    option_name_map_.insert({iter.key(), static_cast<std::string>(iter.value())});
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace ge
