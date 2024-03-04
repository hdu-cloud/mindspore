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

#include "external/graph/ascend_string.h"
#include "debug/ge_log.h"
#include "common/util/mem_utils.h"

namespace ge {
AscendString::AscendString(const char_t *const name) {
  if (name != nullptr) {
    name_ = MakeShared<std::string>(name);
    if (name_ == nullptr) {
      REPORT_CALL_ERROR("E18888", "new string failed.");
      GELOGE(FAILED, "[New][String]AscendString[%s] make shared failed.", name);
    }
  }
}

AscendString::AscendString(const char_t *const name, size_t length) {
  if (name != nullptr) {
    name_ = MakeShared<std::string>(name, length);
    if (name_ == nullptr) {
      REPORT_CALL_ERROR("E18888", "new string with length failed.");
      GELOGE(FAILED, "[New][String]AscendString make shared failed, length=%zu.", length);
    }
  }
}

const char_t *AscendString::GetString() const {
  if (name_ == nullptr) {
    const static char *empty_value = "";
    return empty_value;
  }

  return (*name_).c_str();
}

size_t AscendString::GetLength() const {
  if (name_ == nullptr) {
    return 0UL;
  }

  return (*name_).length();
}

size_t AscendString::Hash() const {
  if (name_ == nullptr) {
    const static size_t kEmptyStringHash = std::hash<std::string>()("");
    return kEmptyStringHash;
  }

  return std::hash<std::string>()(*name_);
}

bool AscendString::operator<(const AscendString &d) const {
  if ((name_ == nullptr) && (d.name_ == nullptr)) {
    return false;
  } else if (name_ == nullptr) {
    return true;
  } else if (d.name_ == nullptr) {
    return false;
  } else {
    return (*name_) < (*(d.name_));
  }
}

bool AscendString::operator>(const AscendString &d) const {
  if ((name_ == nullptr) && (d.name_ == nullptr)) {
    return false;
  } else if (name_ == nullptr) {
    return false;
  } else if (d.name_ == nullptr) {
    return true;
  } else {
    return (*name_) > (*(d.name_));
  }
}

bool AscendString::operator==(const AscendString &d) const {
  if ((name_ == nullptr) && (d.name_ == nullptr)) {
    return true;
  } else if (name_ == nullptr) {
    return false;
  } else if (d.name_ == nullptr) {
    return false;
  } else {
    return (*name_) == (*(d.name_));
  }
}

bool AscendString::operator<=(const AscendString &d) const {
  if (name_ == nullptr) {
    return true;
  } else if (d.name_ == nullptr) {
    return false;
  } else {
    return (*name_) <= (*(d.name_));
  }
}

bool AscendString::operator>=(const AscendString &d) const {
  if (d.name_ == nullptr) {
    return true;
  } else if (name_ == nullptr) {
    return false;
  } else {
    return (*name_) >= (*(d.name_));
  }
}

bool AscendString::operator!=(const AscendString &d) const {
  if ((name_ == nullptr) && (d.name_ == nullptr)) {
    return false;
  } else if (name_ == nullptr) {
    return true;
  } else if (d.name_ == nullptr) {
    return true;
  } else {
    return (*name_) != (*(d.name_));
  }
}

size_t AscendString::Find(const AscendString &ascend_string) const {
  if ((name_ == nullptr) || (ascend_string.name_ == nullptr)) {
    return std::string::npos;
  }
  return name_->find(*(ascend_string.name_));
}
}  // namespace ge
