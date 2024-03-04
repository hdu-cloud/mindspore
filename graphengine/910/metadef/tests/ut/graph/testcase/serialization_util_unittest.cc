/**
 * Copyright 2023-2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or public testing::Test {agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "graph/serialization/utils/serialization_util.h"

namespace ge {
class SerializationUtilUTest : public testing::Test {
 public:
  proto::DataType proto_data_type_;
  DataType ge_data_type_;

 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(SerializationUtilUTest, GetComplex32ProtoDataType) {
  SerializationUtil::GeDataTypeToProto(DT_COMPLEX32, proto_data_type_);
  EXPECT_EQ(proto_data_type_, proto::DT_COMPLEX32);
  SerializationUtil::ProtoDataTypeToGe(proto::DT_COMPLEX32, ge_data_type_);
  EXPECT_EQ(ge_data_type_, DT_COMPLEX32);
}
}  // namespace ge