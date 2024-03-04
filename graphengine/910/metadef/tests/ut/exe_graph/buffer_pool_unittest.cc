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
namespace {
constexpr size_t kLargeBufSizeThreshold = 1024U * 1024U; // 1M
}
using namespace bg;
class BufferPoolUT : public testing::Test {};
TEST_F(BufferPoolUT, IdContinuous) {
  BufferPool tp;
  std::string large_str(kLargeBufSizeThreshold, 'a');
  EXPECT_EQ(tp.AddStr("Hello"), 0);
  EXPECT_EQ(tp.AddStr(large_str.c_str()), 1);
  EXPECT_EQ(tp.AddStr("World"), 2);

  auto text_holder = tp.Serialize();
  ASSERT_NE(text_holder, nullptr);
  auto text = reinterpret_cast<ContinuousBuffer *>(text_holder.get());
  EXPECT_EQ(text->GetNum(), 3);
  EXPECT_STREQ(text->Get<char>(0), "Hello");
  EXPECT_STREQ(text->Get<char>(1), large_str.c_str());
  EXPECT_STREQ(text->Get<char>(2), "World");
}
TEST_F(BufferPoolUT, Deduplication) {
  BufferPool tp;
  std::string large_str(kLargeBufSizeThreshold, 'a');
  EXPECT_EQ(tp.AddStr("Hello"), 0);
  EXPECT_EQ(tp.AddStr("World"), 1);
  EXPECT_EQ(tp.AddStr(large_str.c_str()), 2);
  EXPECT_EQ(tp.AddStr("Hello"), 0);
  EXPECT_EQ(tp.AddStr("Zero"), 3);
  EXPECT_EQ(tp.AddStr(large_str.c_str()), 4);

  auto text_holder = tp.Serialize();
  ASSERT_NE(text_holder, nullptr);
  auto text = reinterpret_cast<ContinuousBuffer *>(text_holder.get());
  EXPECT_EQ(text->GetNum(), 5);
  EXPECT_STREQ(text->Get<char>(0), "Hello");
  EXPECT_STREQ(text->Get<char>(1), "World");
  EXPECT_STREQ(text->Get<char>(2), large_str.c_str());
  EXPECT_STREQ(text->Get<char>(3), "Zero");
  EXPECT_STREQ(text->Get<char>(4), large_str.c_str());
  EXPECT_EQ(text->Get<char>(5), nullptr);
}
TEST_F(BufferPoolUT, NonString) {
  BufferPool tp;
  char buf[] = "Hello\0World\0Zero";
  std::string large_str(kLargeBufSizeThreshold, 'a');
  large_str[1] = '\0';
  EXPECT_EQ(tp.AddBuf(reinterpret_cast<const uint8_t *>(buf), 16), 0);
  EXPECT_EQ(tp.AddStr("World"), 1);
  EXPECT_EQ(tp.AddBuf(reinterpret_cast<const uint8_t *>(large_str.c_str()), kLargeBufSizeThreshold), 2);
  EXPECT_EQ(tp.AddStr("Hello"), 3);
  EXPECT_EQ(tp.AddBuf(reinterpret_cast<const uint8_t *>(large_str.c_str()), kLargeBufSizeThreshold), 4);
  EXPECT_EQ(tp.AddStr("Zero"), 5);
  EXPECT_EQ(tp.AddBuf(reinterpret_cast<const uint8_t *>(buf), 16), 0);

  auto text_holder = tp.Serialize();
  ASSERT_NE(text_holder, nullptr);
  auto text = reinterpret_cast<ContinuousBuffer *>(text_holder.get());
  EXPECT_EQ(text->GetNum(), 6);
  size_t size;
  EXPECT_EQ(memcmp(text->Get<uint8_t>(0, size), buf, 16), 0);
  EXPECT_EQ(size, 16);
  EXPECT_STREQ(text->Get<char>(1, size), "World");
  EXPECT_EQ(size, 6);
  EXPECT_EQ(memcmp(text->Get<uint8_t>(2, size), large_str.c_str(), kLargeBufSizeThreshold), 0);
  EXPECT_EQ(size, kLargeBufSizeThreshold);
  EXPECT_STREQ(text->Get<char>(3, size), "Hello");
  EXPECT_EQ(size, 6);
  EXPECT_EQ(memcmp(text->Get<uint8_t>(4, size), large_str.c_str(), kLargeBufSizeThreshold), 0);
  EXPECT_EQ(size, kLargeBufSizeThreshold);
  EXPECT_STREQ(text->Get<char>(5, size), "Zero");
  EXPECT_EQ(size, 5);
}
TEST_F(BufferPoolUT, CorrectLength) {
  BufferPool tp;
  char buf[] = "Hello\0World\0Zero";
  std::string large_str(kLargeBufSizeThreshold, 'a');
  large_str[1] = '\0';
  EXPECT_EQ(tp.GetSize(), 0);
  EXPECT_EQ(tp.AddBuf(reinterpret_cast<const uint8_t *>(buf), 16), 0);
  EXPECT_EQ(tp.AddStr("World"), 1);
  EXPECT_EQ(tp.AddBuf(reinterpret_cast<const uint8_t *>(large_str.c_str()), kLargeBufSizeThreshold), 2);
  EXPECT_EQ(tp.AddStr("Hello"), 3);
  EXPECT_EQ(tp.AddBuf(reinterpret_cast<const uint8_t *>(large_str.c_str()), kLargeBufSizeThreshold), 4);
  EXPECT_EQ(tp.AddStr("Zero"), 5);
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

  EXPECT_EQ(text->GetNum(), 6);
  size_t size;
  EXPECT_EQ(memcmp(text->Get<uint8_t>(0, size), buf, 16), 0);
  EXPECT_EQ(size, 16);
  EXPECT_STREQ(text->Get<char>(1, size), "World");
  EXPECT_EQ(size, 6);
  EXPECT_EQ(memcmp(text->Get<uint8_t>(2, size), large_str.c_str(), kLargeBufSizeThreshold), 0);
  EXPECT_EQ(size, kLargeBufSizeThreshold);
  EXPECT_STREQ(text->Get<char>(3, size), "Hello");
  EXPECT_EQ(size, 6);
  EXPECT_EQ(memcmp(text->Get<uint8_t>(4, size), large_str.c_str(), kLargeBufSizeThreshold), 0);
  EXPECT_EQ(size, kLargeBufSizeThreshold);
  EXPECT_STREQ(text->Get<char>(5, size), "Zero");
  EXPECT_EQ(size, 5);
  EXPECT_EQ(text->Get<char>(6, size), nullptr);
}

TEST_F(BufferPoolUT, BoundaryValue) {
  BufferPool tp;
  std::string large_str1(kLargeBufSizeThreshold, 'a');
  large_str1[1] = '\0';
  std::string large_str2(kLargeBufSizeThreshold - 1, 'b');
  std::string large_str3(kLargeBufSizeThreshold - 2, 'c');
  EXPECT_EQ(tp.AddBuf(reinterpret_cast<const uint8_t *>(large_str1.c_str()), kLargeBufSizeThreshold), 0);
  EXPECT_EQ(tp.AddBuf(reinterpret_cast<const uint8_t *>(large_str1.c_str()), kLargeBufSizeThreshold), 1);
  EXPECT_EQ(tp.AddBuf(reinterpret_cast<const uint8_t *>(large_str1.c_str()), kLargeBufSizeThreshold - 1), 2);
  EXPECT_EQ(tp.AddBuf(reinterpret_cast<const uint8_t *>(large_str1.c_str()), kLargeBufSizeThreshold - 1), 2);
  EXPECT_EQ(tp.AddStr(large_str2.c_str()), 3);
  EXPECT_EQ(tp.AddStr(large_str2.c_str()), 4);
  EXPECT_EQ(tp.AddStr(large_str3.c_str()), 5);
  EXPECT_EQ(tp.AddStr(large_str3.c_str()), 5);

  EXPECT_NE(tp.GetBufById(0), nullptr);
  EXPECT_NE(tp.GetBufById(1), nullptr);
  EXPECT_NE(tp.GetBufById(2), nullptr);
  EXPECT_NE(tp.GetBufById(3), nullptr);
  EXPECT_NE(tp.GetBufById(4), nullptr);
  EXPECT_NE(tp.GetBufById(5), nullptr);
  EXPECT_EQ(tp.GetBufById(6), nullptr);

  auto text_holder = tp.Serialize();
  ASSERT_NE(text_holder, nullptr);
  auto text = reinterpret_cast<ContinuousBuffer *>(text_holder.get());
  EXPECT_EQ(text->GetNum(), 6);
  size_t size;
  EXPECT_EQ(memcmp(text->Get<uint8_t>(0, size), large_str1.c_str(), kLargeBufSizeThreshold), 0);
  EXPECT_EQ(size, kLargeBufSizeThreshold);
  EXPECT_EQ(memcmp(text->Get<uint8_t>(1, size), large_str1.c_str(), kLargeBufSizeThreshold), 0);
  EXPECT_EQ(size, kLargeBufSizeThreshold);
  EXPECT_EQ(memcmp(text->Get<uint8_t>(2, size), large_str1.c_str(), kLargeBufSizeThreshold - 1), 0);
  EXPECT_EQ(size, kLargeBufSizeThreshold - 1);
  EXPECT_STREQ(text->Get<char>(3, size), large_str2.c_str());
  EXPECT_EQ(size, kLargeBufSizeThreshold);
  EXPECT_STREQ(text->Get<char>(4, size), large_str2.c_str());
  EXPECT_EQ(size, kLargeBufSizeThreshold);
  EXPECT_STREQ(text->Get<char>(5, size), large_str3.c_str());
  EXPECT_EQ(size, kLargeBufSizeThreshold - 1);
  EXPECT_EQ(text->Get<char>(6, size), nullptr);
}
}  // namespace gert