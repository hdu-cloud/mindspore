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
#include <gtest/gtest.h>
#include <iostream>
#include "test_structs.h"
#include "func_counter.h"
#include "graph/ge_context.h"
#include "graph/ge_global_options.h"
#include "graph/ge_local_context.h"
#include "graph/node.h"
#include "graph_builder_utils.h"

namespace ge {


class GeContextUt : public testing::Test {};

void SetIROptionToShowName(std::string &option_name_map, const std::string &ir_option, const std::string &show_name) {
  std::string json = "\"" + ir_option + "\": \"" + show_name +"\",\n";
  option_name_map += json;
  return;
}

TEST_F(GeContextUt, All) {
  ge::GEContext cont = GetContext();
  cont.Init();
  EXPECT_EQ(cont.GetHostExecFlag(), false);
  EXPECT_EQ(cont.IsOverflowDetectionOpen(), false);
  EXPECT_EQ(GetMutableGlobalOptions().size(), 0);
  EXPECT_EQ(cont.SessionId(), 0);
  EXPECT_EQ(cont.DeviceId(), 0);

  cont.SetSessionId(1);
  cont.SetContextId(2);
  cont.SetCtxDeviceId(4);
  EXPECT_EQ(cont.SessionId(), 1);
  EXPECT_EQ(cont.DeviceId(), 4);

  EXPECT_EQ(cont.StreamSyncTimeout(), -1);
  cont.SetStreamSyncTimeout(10000);
  EXPECT_EQ(cont.StreamSyncTimeout(), 10000);

  EXPECT_EQ(cont.EventSyncTimeout(), -1);
  cont.SetEventSyncTimeout(20000);
  EXPECT_EQ(cont.EventSyncTimeout(), 20000);
}

TEST_F(GeContextUt, Plus) {
  std::map<std::string, std::string> session_option{{"ge.exec.placement", "ge.exec.placement"}};
  GetThreadLocalContext().SetSessionOption(session_option);
  
  std::string exec_placement;
  GetThreadLocalContext().GetOption("ge.exec.placement", exec_placement);
  EXPECT_EQ(exec_placement, "ge.exec.placement");
  ge::GEContext cont = GetContext();
  EXPECT_EQ(cont.GetHostExecFlag(), false);

  std::map<std::string, std::string> session_option0{{"ge.graphLevelSat", "1"}};
  GetThreadLocalContext().SetSessionOption(session_option0);
  EXPECT_EQ(cont.IsGraphLevelSat(), true);

  std::map<std::string, std::string> session_option1{{"ge.exec.overflow", "1"}};
  GetThreadLocalContext().SetSessionOption(session_option1);
  EXPECT_EQ(cont.IsOverflowDetectionOpen(), true);

  std::map<std::string, std::string> session_option2{{"ge.exec.sessionId", "12345678987654321"}};
  GetThreadLocalContext().SetSessionOption(session_option2);
  cont.Init();
  std::map<std::string, std::string> session_option3{{"ge.exec.deviceId", "12345678987654321"}};
  GetThreadLocalContext().SetSessionOption(session_option3);
  cont.Init();
  std::map<std::string, std::string> session_option4{{"ge.exec.jobId", "12345"}};
  GetThreadLocalContext().SetSessionOption(session_option4);
  cont.Init();
  std::map<std::string, std::string> session_option5{{"ge.exec.jobId", "65536"}};
  GetThreadLocalContext().SetSessionOption(session_option5);
  cont.Init();
}

TEST_F(GeContextUt, set_valid_SyncTimeout_from_option) {
  std::map<std::string, std::string> session_option{{"ge.exec.sessionId", "0"},
                                                    {"ge.exec.deviceId", "1"},
                                                    {"ge.exec.jobId", "2"},
                                                    {"stream_sync_timeout", "10000"},
                                                    {"event_sync_timeout", "20000"}};
  GetThreadLocalContext().SetSessionOption(session_option);
  ge::GEContext ctx = GetContext();
  ctx.Init();
  EXPECT_EQ(ctx.StreamSyncTimeout(), 10000);
  EXPECT_EQ(ctx.EventSyncTimeout(), 20000);
}

TEST_F(GeContextUt, set_invalid_option) {
  std::map<std::string, std::string> session_option{{"ge.exec.sessionId", "-1"},
                                                    {"ge.exec.deviceId", "-1"},
                                                    {"ge.exec.jobId", "-1"},
                                                    {"stream_sync_timeout", ""},
                                                    {"event_sync_timeout", ""}};
  GetThreadLocalContext().SetSessionOption(session_option);
  ge::GEContext ctx = GetContext();
  ctx.Init();
  EXPECT_EQ(ctx.SessionId(), 0U);
  EXPECT_EQ(ctx.DeviceId(), 0U);
  EXPECT_EQ(ctx.StreamSyncTimeout(), -1);
  EXPECT_EQ(ctx.EventSyncTimeout(), -1);
}

TEST_F(GeContextUt, set_OutOfRange_SyncTimeout_from_option) {
  std::map<std::string, std::string> session_option{{"ge.exec.sessionId", "1234567898765432112345"},
                                                    {"ge.exec.deviceId", "1234567898765432112345"},
                                                    {"ge.exec.jobId", "1234567898765432112345"},
                                                    {"stream_sync_timeout", "1234567898765432112345"},
                                                    {"event_sync_timeout", "1234567898765432112345"}};
  GetThreadLocalContext().SetSessionOption(session_option);
  ge::GEContext ctx = GetContext();
  ctx.Init();
  EXPECT_EQ(ctx.SessionId(), 0U);
  EXPECT_EQ(ctx.DeviceId(), 0U);
  EXPECT_EQ(ctx.StreamSyncTimeout(), -1);
  EXPECT_EQ(ctx.EventSyncTimeout(), -1);
}

TEST_F(GeContextUt, set_option_name_map_success) {
  std::string option_name_map;
  SetIROptionToShowName(option_name_map, "ge.enableSmallChannel", "enable_small_channel");
  SetIROptionToShowName(option_name_map, "ge.exec.enable_exception_dump", "enable_exception_dump");
  SetIROptionToShowName(option_name_map, "ge.exec.opWaitTimeout", "op_wait_timeout");
  option_name_map = "{\n" + option_name_map.substr(0, option_name_map.size() - 2) + "\n}";
  ge::GEContext ctx = GetContext();
  EXPECT_EQ(ctx.SetOptionNameMap(option_name_map), ge::GRAPH_SUCCESS);
  std::string show_name;
  show_name = ctx.GetReadableName("ge.enableSmallChannel");
  EXPECT_EQ(show_name, "enable_small_channel");
  show_name = ctx.GetReadableName("ge.exec.enable_exception_dump");
  EXPECT_EQ(show_name, "enable_exception_dump");
  show_name = ctx.GetReadableName("ge.exec.opWaitTimeout");
  EXPECT_EQ(show_name, "op_wait_timeout");
}

TEST_F(GeContextUt, set_invalid_option_name_map) {
  std::string option_name_map_1 = "";
  ge::GEContext ctx = GetContext();
  EXPECT_EQ(ctx.SetOptionNameMap(option_name_map_1), ge::GRAPH_FAILED);
  std::string show_name;
  show_name = ctx.GetReadableName("ge.enableSmallChannel");
  EXPECT_EQ(show_name, "ge.enableSmallChannel");
  show_name = ctx.GetReadableName("ge.exec.enable_exception_dump");
  EXPECT_EQ(show_name, "ge.exec.enable_exception_dump");
  show_name = ctx.GetReadableName("ge.exec.opWaitTimeout");
  EXPECT_EQ(show_name, "ge.exec.opWaitTimeout");

  std::string option_name_map_2;
  SetIROptionToShowName(option_name_map_2, "ge.enableSmallChannel", "");
  option_name_map_2 = "{\n" + option_name_map_2.substr(0, option_name_map_2.size() - 2) + "\n}";
  EXPECT_EQ(ctx.SetOptionNameMap(option_name_map_2), ge::GRAPH_FAILED);
  show_name = ctx.GetReadableName("ge.enableSmallChannel");
  EXPECT_EQ(show_name, "ge.enableSmallChannel");

  std::string option_name_map_3;
  SetIROptionToShowName(option_name_map_3, "", "enable_small_channel");
  option_name_map_2 = "{\n" + option_name_map_3.substr(0, option_name_map_3.size() - 2) + "\n}";
  EXPECT_EQ(ctx.SetOptionNameMap(option_name_map_3), ge::GRAPH_FAILED);
  show_name = ctx.GetReadableName("ge.enableSmallChannel");
  EXPECT_EQ(show_name, "ge.enableSmallChannel");
}

}  // namespace ge
