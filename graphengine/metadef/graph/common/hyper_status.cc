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

#include "common/hyper_status.h"

#include <cstring>
#include <memory>

namespace gert {
ge::char_t *CreateMessage(const ge::char_t *format, va_list arg) {
  if (format == nullptr) {
    return nullptr;
  }

  va_list arg_copy;
  va_copy(arg_copy, arg);
  int len = vsnprintf(nullptr, 0, format, arg_copy);
  va_end(arg_copy);

  if (len < 0) {
    return nullptr;
  }

  auto msg = std::unique_ptr<ge::char_t[]>(new (std::nothrow) ge::char_t[len + 1]);
  if (msg == nullptr) {
    return nullptr;
  }

  auto ret = vsnprintf_s(msg.get(), len + 1, len, format, arg);
  if (ret < 0) {
    return nullptr;
  }

  return msg.release();
}
HyperStatus HyperStatus::Success() {
  return {};
}
HyperStatus::HyperStatus(const HyperStatus &other) : status_{nullptr} {
  *this = other;
}
HyperStatus &HyperStatus::operator=(const HyperStatus &other) {
  delete [] status_;
  if (other.status_ == nullptr) {
    status_ = nullptr;
  } else {
    size_t status_len = strlen(other.status_) + 1;
    status_ = new (std::nothrow) ge::char_t[status_len];
    if (status_ != nullptr) {
      auto ret = strcpy_s(status_, status_len, other.status_);
      if (ret != EOK) {
        status_[0] = '\0';
      }
    }
  }
  return *this;
}
HyperStatus::HyperStatus(HyperStatus &&other) noexcept {
  status_ = other.status_;
  other.status_ = nullptr;
}
HyperStatus &HyperStatus::operator=(HyperStatus &&other) noexcept {
  delete [] status_;
  status_ = other.status_;
  other.status_ = nullptr;
  return *this;
}
HyperStatus HyperStatus::ErrorStatus(const ge::char_t *message, ...) {
  HyperStatus status;
  va_list arg;
  va_start(arg, message);
  status.status_ = CreateMessage(message, arg);
  va_end(arg);
  return status;
}
HyperStatus HyperStatus::ErrorStatus(std::unique_ptr<ge::char_t[]> message) {
  HyperStatus status;
  status.status_ = message.release();
  return status;
}
}