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

#ifndef METADEF_CXX_TESTS_DEPENDS_CACHE_DESC_STUB_RUNTIME_CACHE_DESC_H
#define METADEF_CXX_TESTS_DEPENDS_CACHE_DESC_STUB_RUNTIME_CACHE_DESC_H
#include <vector>
#include "graph/cache_policy/cache_desc.h"
#include "exe_graph/runtime/shape.h"

namespace ge {
class RuntimeCacheDesc : public CacheDesc {
 public:
  const std::vector<gert::Shape> &GetShapes() const {
    return shapes_;
  }
  void SetShapes(const std::vector<gert::Shape> &shapes) {
    shapes_ = shapes;
  }

  bool IsEqual(const CacheDescPtr &other) const override;
  bool IsMatch(const CacheDescPtr &other) const override;
  CacheHashKey GetCacheDescHash() const override;

 private:
  bool operator==(const RuntimeCacheDesc &rht) const;

 private:
  std::vector<gert::Shape> shapes_;
};
}  // namespace ge
#endif  // METADEF_CXX_TESTS_DEPENDS_CACHE_DESC_STUB_RUNTIME_CACHE_DESC_H
