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
#include "exe_graph/runtime/continuous_vector.h"
#include <gtest/gtest.h>
namespace gert {
class ContinuousVectorUT : public testing::Test {};
TEST_F(ContinuousVectorUT, CreateOk) {
  auto vec_holder = ContinuousVector::Create<size_t>(16);
  auto vec = reinterpret_cast<ContinuousVector *>(vec_holder.get());
  auto c_vec = reinterpret_cast<const ContinuousVector *>(vec_holder.get());
  ASSERT_NE(vec, nullptr);
  EXPECT_EQ(vec->GetSize(), 0);
  EXPECT_EQ(vec->GetCapacity(), 16);
  EXPECT_EQ(c_vec->GetSize(), 0);
  EXPECT_EQ(c_vec->GetCapacity(), 16);
}
TEST_F(ContinuousVectorUT, SetSizeOk) {
  auto vec_holder = ContinuousVector::Create<size_t>(16);
  auto vec = reinterpret_cast<ContinuousVector *>(vec_holder.get());
  ASSERT_NE(vec, nullptr);
  EXPECT_EQ(vec->GetSize(), 0);
  EXPECT_EQ(vec->SetSize(8), ge::GRAPH_SUCCESS);
  EXPECT_EQ(vec->GetSize(), 8);
  EXPECT_EQ(vec->SetSize(16), ge::GRAPH_SUCCESS);
  EXPECT_EQ(vec->GetSize(), 16);
  EXPECT_EQ(vec->SetSize(0), ge::GRAPH_SUCCESS);
  EXPECT_EQ(vec->GetSize(), 0);
}
TEST_F(ContinuousVectorUT, SetSizeFailedOutOfBounds) {
  auto vec_holder = ContinuousVector::Create<size_t>(16);
  auto vec = reinterpret_cast<ContinuousVector *>(vec_holder.get());
  ASSERT_NE(vec, nullptr);
  EXPECT_EQ(vec->GetSize(), 0);
  EXPECT_NE(vec->SetSize(17), ge::GRAPH_SUCCESS);
}
TEST_F(ContinuousVectorUT, CreateNone) {
  auto vec_holder = ContinuousVector::Create<size_t>(0);
  auto vec = reinterpret_cast<ContinuousVector *>(vec_holder.get());
  ASSERT_NE(vec, nullptr);
  EXPECT_EQ(vec->GetSize(), 0);
  EXPECT_EQ(vec->GetCapacity(), 0);
}
TEST_F(ContinuousVectorUT, WriteOk) {
  auto vec_holder = ContinuousVector::Create<size_t>(2);
  auto vec = reinterpret_cast<ContinuousVector *>(vec_holder.get());
  ASSERT_NE(vec, nullptr);
  EXPECT_EQ(vec->GetSize(), 0);
  EXPECT_EQ(vec->GetCapacity(), 2);

  EXPECT_EQ(vec->SetSize(2), ge::GRAPH_SUCCESS);
  auto data = reinterpret_cast<size_t *>(vec->MutableData());
  data[0] = 1024;
  data[1] = 2048;
  EXPECT_EQ(vec->GetSize(), 2);
  EXPECT_EQ(reinterpret_cast<const size_t *>(vec->GetData())[0], 1024);
  EXPECT_EQ(reinterpret_cast<const size_t *>(vec->GetData())[1], 2048);
}
TEST_F(ContinuousVectorUT, TypedOk) {
  auto vec_holder = ContinuousVector::Create<size_t>(16);
  auto vec = reinterpret_cast<ContinuousVector *>(vec_holder.get());
  ASSERT_NE(vec, nullptr);
  EXPECT_EQ(vec->SetSize(4), ge::GRAPH_SUCCESS);
  auto data = reinterpret_cast<size_t *>(vec->MutableData());
  data[0] = 1024;
  data[1] = 2048;
  data[2] = 4096;
  data[3] = 8192;

  auto t_vec = reinterpret_cast<const TypedContinuousVector<size_t> *>(vec);
  EXPECT_EQ(t_vec->GetSize(), 4);
  EXPECT_EQ(t_vec->GetCapacity(), 16);
  EXPECT_EQ(t_vec->GetData()[0], 1024);
  EXPECT_EQ(t_vec->GetData()[1], 2048);
  EXPECT_EQ(t_vec->GetData()[2], 4096);
  EXPECT_EQ(t_vec->GetData()[3], 8192);
  auto mt_vec = reinterpret_cast<TypedContinuousVector<size_t> *>(vec);
  EXPECT_EQ(mt_vec->MutableData()[0], 1024);
  EXPECT_EQ(mt_vec->MutableData()[1], 2048);
  EXPECT_EQ(mt_vec->MutableData()[2], 4096);
  EXPECT_EQ(mt_vec->MutableData()[3], 8192);
}
}  // namespace gert