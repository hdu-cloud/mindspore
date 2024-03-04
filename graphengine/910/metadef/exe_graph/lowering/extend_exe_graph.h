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

#ifndef AIR_CXX_RUNTIME_V2_METADEF_EXE_GRAPH_EXETEND_EXE_GRAPH_ATTRS_H_
#define AIR_CXX_RUNTIME_V2_METADEF_EXE_GRAPH_EXETEND_EXE_GRAPH_ATTRS_H_
#include "graph/compute_graph.h"
namespace gert {
template<typename K, typename V>
inline bool FindValFromMapExtAttr(const ge::ComputeGraphPtr &exe_graph, const char *attr_name, const K &key, V &val) {
  auto ext_attr = exe_graph->GetExtAttr<std::unordered_map<K, V>>(attr_name);
  if (ext_attr == nullptr) {
    return false;
  }
  const auto iter = ext_attr->find(key);
  if (iter != ext_attr->cend()) {
    val = iter->second;
    return true;
  }
  return false;
}

template<typename K, typename V>
inline void AddKVToMapExtAttr(const ge::ComputeGraphPtr &exe_graph, const char *attr_name, const K &key, const V &val) {
  auto ext_attr = exe_graph->GetExtAttr<std::unordered_map<K, V>>(attr_name);
  if (ext_attr == nullptr) {
    std::unordered_map<K, V> temp_ext_attr {};
    temp_ext_attr[key] = val;
    exe_graph->SetExtAttr(attr_name, temp_ext_attr);
  } else {
    (*ext_attr)[key] = val;
  }
}
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_METADEF_EXE_GRAPH_EXETEND_EXE_GRAPH_ATTRS_H_
