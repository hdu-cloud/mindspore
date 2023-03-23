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
#include "exe_graph/lowering/buffer_pool.h"
#include <gtest/gtest.h>
#include "exe_graph/runtime/continuous_buffer.h"
namespace gert {
using namespace bg;
class BufferPoolUT : public testing::Test {};
TEST_F(BufferPoolUT, IdContinuous) {
  BufferPool tp;
  EXPECT_EQ(tp.AddStr("Hello"), 0);
  EXPECT_EQ(tp.AddStr("World"), 1);

  auto text_holder = tp.Serialize();
  ASSERT_NE(text_holder, nullptr);
  auto text = reinterpret_cast<ContinuousBuffer *>(text_holder.get());
  EXPECT_EQ(text->GetNum(), 2);
  EXPECT_STREQ(text->Get<char>(0), "Hello");
  EXPECT_STREQ(text->Get<char>(1), "World");
}
TEST_F(BufferPoolUT, Deduplication) {
  BufferPool tp;
  EXPECT_EQ(tp.AddStr("Hello"), 0);
  EXPECT_EQ(tp.AddStr("World"), 1);
  EXPECT_EQ(tp.AddStr("Hello"), 0);
  EXPECT_EQ(tp.AddStr("Zero"), 2);

  auto text_holder = tp.Serialize();
  ASSERT_NE(text_holder, nullptr);
  auto text = reinterpret_cast<ContinuousBuffer *>(text_holder.get());
  EXPECT_EQ(text->GetNum(), 3);
  EXPECT_STREQ(text->Get<char>(0), "Hello");
  EXPECT_STREQ(text->Get<char>(1), "World");
  EXPECT_STREQ(text->Get<char>(2), "Zero");
  EXPECT_EQ(text->Get<char>(3), nullptr);
}
TEST_F(BufferPoolUT, NonString) {
  BufferPool tp;
  char buf[] = "Hello\0World\0Zero";
  EXPECT_EQ(tp.AddBuf(reinterpret_cast<const uint8_t *>(buf), 16), 0);
  EXPECT_EQ(tp.AddStr("World"), 1);
  EXPECT_EQ(tp.AddStr("Hello"), 2);
  EXPECT_EQ(tp.AddStr("Zero"), 3);
  EXPECT_EQ(tp.AddBuf(reinterpret_cast<const uint8_t *>(buf), 16), 0);

  auto text_holder = tp.Serialize();
  ASSERT_NE(text_holder, nullptr);
  auto text = reinterpret_cast<ContinuousBuffer *>(text_holder.get());
  EXPECT_EQ(text->GetNum(), 4);
  size_t size;
  EXPECT_EQ(memcmp(text->Get<uint8_t>(0, size), buf, 16), 0);
  EXPECT_EQ(size, 16);
  EXPECT_STREQ(text->Get<char>(1, size), "World");
  EXPECT_EQ(size, 6);
  EXPECT_STREQ(text->Get<char>(2, size), "Hello");
  EXPECT_EQ(size, 6);
  EXPECT_STREQ(text->Get<char>(3, size), "Zero");
  EXPECT_EQ(size, 5);
}
TEST_F(BufferPoolUT, CorrectLength) {
  BufferPool tp;
  char buf[] = "Hello\0World\0Zero";
  EXPECT_EQ(tp.AddBuf(reinterpret_cast<const uint8_t *>(buf), 16), 0);
  EXPECT_EQ(tp.AddStr("World"), 1);
  EXPECT_EQ(tp.AddStr("Hello"), 2);
  EXPECT_EQ(tp.AddStr("Zero"), 3);
  EXPECT_EQ(tp.AddBuf(reinterpret_cast<const uint8_t *>(buf), 16), 0);

  size_t total_size;
  auto text_holder = tp.Serialize(total_size);
  ASSERT_NE(text_holder, nullptr);
  auto text = reinterpret_cast<ContinuousBuffer *>(text_holder.get());

  auto length = text->GetTotalLength();
  EXPECT_EQ(length, total_size);
  auto another = std::unique_ptr<uint8_t[]>(new uint8_t[length]);
  ASSERT_NE(another, nullptr);
  memcpy(another.get(), text_holder.get(), length);
  text = reinterpret_cast<ContinuousBuffer *>(another.get());

  EXPECT_EQ(text->GetNum(), 4);
  size_t size;
  EXPECT_EQ(memcmp(text->Get<uint8_t>(0, size), buf, 16), 0);
  EXPECT_EQ(size, 16);
  EXPECT_STREQ(text->Get<char>(1, size), "World");
  EXPECT_EQ(size, 6);
  EXPECT_STREQ(text->Get<char>(2, size), "Hello");
  EXPECT_EQ(size, 6);
  EXPECT_STREQ(text->Get<char>(3, size), "Zero");
  EXPECT_EQ(size, 5);
  EXPECT_EQ(text->Get<char>(4, size), nullptr);
}
}  // namespace gert