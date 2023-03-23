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

#include "exe_graph/lowering/buffer_pool.h"

#include <securec.h>
#include "framework/common/debug/ge_log.h"
#include "graph/utils/math_util.h"
#include "graph/debug/ge_log.h"
#include "graph/def_types.h"
#include "graph/debug/ge_util.h"
#include "common/checker.h"

#include "exe_graph/runtime/continuous_buffer.h"

namespace gert {
namespace bg {
BufferPool::BufId BufferPool::AddBuf(const uint8_t *data, size_t len) {
  return AddBuf(std::string(ge::PtrToPtr<uint8_t, char>(data), len));
}
BufferPool::BufId BufferPool::AddStr(const char *data) {
  return AddBuf(std::string(data, strlen(data) + 1));
}
BufferPool::BufId BufferPool::AddBuf(std::string &&str) {
  return bufs_to_id_.emplace(std::move(str), bufs_to_id_.size()).first->second;
}
std::unique_ptr<uint8_t[]> BufferPool::Serialize() const {
  size_t total_size;
  return Serialize(total_size);
}
std::unique_ptr<uint8_t[]> BufferPool::Serialize(size_t &total_size) const {
  total_size = sizeof(ContinuousBuffer);
  size_t buf_count = bufs_to_id_.size();
  size_t offset_size;
  size_t text_offset;
  // 申请了n个，但是使用时会用n+1个，多的一个由ContinuousText自带
  if (ge::MulOverflow(sizeof(size_t), buf_count, offset_size)) {
    GE_LOGE("Failed to serialize buffer pool, size overflow, buf num %zu", buf_count);
    return nullptr;
  }
  if (ge::AddOverflow(total_size, offset_size, total_size)) {
    GE_LOGE("Failed to serialize buffer pool, size overflow, buf size %zu", offset_size);
    return nullptr;
  }
  text_offset = total_size;

  std::vector<const std::string *> ids_to_buf(buf_count);
  for (const auto &iter : bufs_to_id_) {
    if (iter.second >= buf_count) {
      return nullptr;
    }
    ids_to_buf[iter.second] = &iter.first;

    if (ge::AddOverflow(total_size, iter.first.size(), total_size)) {
      GE_LOGE("Failed to serialize buffer pool, size overflow, buf size %zu, id %zu", iter.first.size(), iter.second);
      return nullptr;
    }
  }

  auto text_holder = ge::ComGraphMakeUnique<uint8_t[]>(total_size);
  GE_ASSERT_NOTNULL(text_holder);

  auto text = ge::PtrToPtr<uint8_t, ContinuousBuffer>(text_holder.get());
  text->num_ = buf_count;
  size_t i = 0;
  for (; i < buf_count; ++i) {
    auto buf = ids_to_buf[i];
    if (buf == nullptr) {
      GELOGE(ge::FAILED, "Failed to serialize text pool, miss buf id %zu", i);
      return nullptr;
    }
    GE_ASSERT_EOK(memcpy_s(text_holder.get() + text_offset, total_size - text_offset, buf->data(), buf->size()));
    text->offsets_[i] = text_offset;
    text_offset += buf->size();
  }
  text->offsets_[i] = text_offset;

  return text_holder;
}
const char *BufferPool::GetBufById(BufId id) const {
  for (const auto &buf_and_id : bufs_to_id_) {
    if (buf_and_id.second == id) {
      return buf_and_id.first.c_str();
    }
  }
  return nullptr;
}
size_t BufferPool::GetSize() const {
  return bufs_to_id_.size();
}
}  // namespace bg
}  // namespace gert
