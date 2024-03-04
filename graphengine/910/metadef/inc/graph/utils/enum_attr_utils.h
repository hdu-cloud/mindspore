/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef __INC_METADEF_ENUM_ATTR_UTILS_H
#define __INC_METADEF_ENUM_ATTR_UTILS_H

#include <vector>
#include "common/ge_common/util.h"
#include "graph/ge_error_codes.h"
#include "graph/ge_tensor.h"

namespace ge {
using namespace std;
constexpr uint16_t kMaxValueOfEachDigit = 127U;
constexpr size_t kAppendNum = 1U;
constexpr char_t prefix = '\0';

class EnumAttrUtils {
 public:
  static void GetEnumAttrName(vector<string> &enum_attr_names, const string &attr_name, string &enum_attr_name,
                              bool &is_new_attr);
  static void GetEnumAttrValue(vector<string> &enum_attr_values, const string &attr_value, int64_t &enum_attr_value);
  static void GetEnumAttrValues(vector<string> &enum_attr_values, const vector<string> &attr_values,
                                vector<int64_t> &enum_values);

  static graphStatus GetAttrName(const vector<string> &enum_attr_names, const vector<bool> name_use_string_values,
                                 const string &enum_attr_name, string &attr_name, bool &is_value_string);
  static graphStatus GetAttrValue(const vector<string> &enum_attr_values, const int64_t enum_attr_value,
                                  string &attr_value);
  static graphStatus GetAttrValues(const vector<string> &enum_attr_values, const vector<int64_t> &enum_values,
                                   vector<string> &attr_values);
 private:
  static void Encode(const uint32_t src, string &dst);
  static void Decode(const string &src, size_t &dst);
};
} // namespace ge
#endif // __INC_METADEF_ENUM_ATTR_UTILS_H
