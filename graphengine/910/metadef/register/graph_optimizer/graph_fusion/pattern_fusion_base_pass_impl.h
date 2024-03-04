/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef FE_PATTERN_FUSION_BASE_PASS_IMPL_H
#define FE_PATTERN_FUSION_BASE_PASS_IMPL_H

#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "graph/debug/ge_log.h"
#include "common/opskernel/ops_kernel_info_store.h"
#include "register/graph_optimizer/graph_fusion/fusion_pattern.h"

namespace fe {
using OpDesc = FusionPattern::OpDesc;
using Mapping = std::map<const std::shared_ptr<OpDesc>, std::vector<ge::NodePtr>, CmpKey>;
using Mappings = std::vector<Mapping>;
using OpsKernelInfoStorePtr = std::shared_ptr<ge::OpsKernelInfoStore>;
struct CandidateAndMapping {
  std::vector<ge::NodePtr> candidate_nodes;
  std::vector<FusionPattern::OpDescPtr> candidate_op_descs;
  Mapping &mapping;
  CandidateAndMapping(Mapping &mapping_param) : mapping(mapping_param) {}
};

/** Base pattern impl
 * @ingroup FUSION_PASS_GROUP
 * @note New virtual methods should be append at the end of this class
 */
class PatternFusionBasePassImpl {
 public:
  PatternFusionBasePassImpl();

  virtual ~PatternFusionBasePassImpl();

  const std::vector<FusionPattern *> &GetPatterns();

  void GetPatterns(std::vector<FusionPattern *> &patterns);

  void SetPatterns(const std::vector<FusionPattern *> &patterns);

  void SetOpsKernelInfoStore(const OpsKernelInfoStorePtr ops_kernel_info_store_ptr);

  PatternFusionBasePassImpl &operator=(const PatternFusionBasePassImpl &) = delete;
  PatternFusionBasePassImpl(const PatternFusionBasePassImpl &another_pattern_fusion) = delete;

  bool CheckOpSupported(const ge::OpDescPtr &op_desc_ptr) const;

  bool CheckOpSupported(const ge::NodePtr &node) const;

  bool CheckAccuracySupported(const ge::NodePtr &node) const;

  static bool IsNodesExist(const ge::NodePtr current_node, std::vector<ge::NodePtr> &nodes);

  static bool IsMatched(const std::shared_ptr<OpDesc> op_desc, const ge::NodePtr node, const Mapping &mapping);

  void DumpMappings(const FusionPattern &pattern, const Mappings &mappings) const;

  static bool IsOpTypeExist(const std::string &type, const std::vector<std::string> &types);

  bool MatchFromOutput(const ge::NodePtr output_node, const std::shared_ptr<OpDesc> output_op_desc,
                       Mapping &mapping) const;

  bool GetMatchOutputNodes(const ge::ComputeGraph &graph, const FusionPattern &pattern,
                           std::vector<ge::NodePtr> &matched_output_nodes) const;

 private:
  std::vector<FusionPattern *> patterns_;

  OpsKernelInfoStorePtr ops_kernel_info_store_ptr_;

  bool GetSortedInAnchors(const ge::NodePtr &node, const std::string &op_id,
                          std::vector<ge::InDataAnchorPtr> &in_anchors) const;

  void MatchOneOutputNode(const ge::NodePtr &output_node,
                          const std::vector<FusionPattern::OpDescPtr> &outputs_desc,
                          size_t &out_idx, const std::unique_ptr<bool[]> &usage_flags,
                          CandidateAndMapping &cand) const;

  bool MatchFromOutput(CandidateAndMapping &cand) const;

  void MatchFuzzyOutputs(const ge::NodePtr &node, const FusionPattern::OpDescPtr &op_desc,
                         size_t &out_idx, const std::unique_ptr<bool[]> &usage_flags,
                         CandidateAndMapping &cand) const;

  bool MatchOutputs(CandidateAndMapping &cand) const;

  void UpdateCandidates(const CandidateAndMapping &temp_cand, CandidateAndMapping &cand) const;

  void AddCandidateQueue(const FusionPattern::OpDescPtr &op_desc, const ge::NodePtr &node,
                         CandidateAndMapping &cand) const;

  bool MatchAsInput(std::vector<ge::NodePtr> &candidate_nodes,
                    std::vector<FusionPattern::OpDescPtr> &candidate_op_descs, Mapping &mapping) const;

  static bool MatchAllEdges(const size_t &input_size, const std::unique_ptr<bool[]> &usage_flags);

  static void GetInDataAnchors(const ge::NodePtr &node, std::vector<ge::InDataAnchorPtr> &in_anchor_vec);

  static void GetOutDataAnchors(const ge::NodePtr &node, std::vector<ge::OutDataAnchorPtr> &out_anchor_vec);
};

}  // namespace fe

#endif  // FE_PATTERN_FUSION_BASE_PASS_H
