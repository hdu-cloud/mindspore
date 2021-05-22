/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_GRAPH_SCHEDULER_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_GRAPH_SCHEDULER_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <fstream>
#include "runtime/framework/actor/data_source_actor.h"
#include "runtime/framework/actor/loop_count_actor.h"
#include "runtime/framework/actor/kernel_actor.h"
#include "runtime/framework/actor/output_actor.h"
#include "runtime/framework/actor/switch_actor.h"
#include "runtime/framework/actor/copy_actor.h"
#include "runtime/hardware/device_context.h"
#include "backend/session/kernel_graph.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::session::KernelWithIndex;
using KernelMapPosition = std::map<KernelWithIndex, size_t, session::KernelWithIndexCmp>;
using ActorInfo = std::string;

// The second element of pair represents the output index of op actor corresponding to the graph output node.
using GraphOutputPair = std::pair<OpActor<DeviceTensor> *, size_t>;

// DataArrowPair represent data edge between from actor and to actor.
// The first element of pair is the AID of from actor, and
// second element is op arrow between actors.
using DataArrowPair = std::pair<AID, DataArrowPtr>;

enum class GraphExecutionStrategy {
  kPipeline,  // The actor running is triggered only by data.
  kStep       // The actor running need be triggered by control in addition.
};

// The graph compiler info generated by graph compiler is the express of executable graph.
// The device context is unified interface of interaction with device of corresponding graph.
// The tensors mask is used to distinguish input tensor's type.
// The input tensor is used to link graphs in the dynamic build scenario.
// The control node is used to link graphs in the control flow scenario.
// The origin parameters order is used to correspond to the input args.
// The origin outputs order is used to correspond to the output args.
struct GraphCompilerInfo {
  GraphCompilerInfo(const std::vector<KernelGraphPtr> &graphs, const std::vector<DeviceContext *> &device_contexts,
                    const std::vector<std::vector<int64_t> *> &tensors_mask,
                    const std::vector<std::vector<TensorPtr> *> &input_tensors,
                    const std::vector<AnfNodePtr> &control_nodes,
                    const std::vector<AnfNodePtr> &origin_parameters_order,
                    const KernelMapPosition &origin_outputs_order, const std::string &name)
      : graphs_(graphs),
        device_contexts_(device_contexts),
        tensors_mask_(tensors_mask),
        input_tensors_(input_tensors),
        control_nodes_(control_nodes),
        origin_parameters_order_(origin_parameters_order),
        origin_outputs_order_(origin_outputs_order),
        name_(name) {}
  std::vector<KernelGraphPtr> graphs_;
  std::vector<DeviceContext *> device_contexts_;
  std::vector<std::vector<int64_t> *> tensors_mask_;
  std::vector<std::vector<TensorPtr> *> input_tensors_;
  std::vector<AnfNodePtr> control_nodes_;
  std::vector<AnfNodePtr> origin_parameters_order_;
  KernelMapPosition origin_outputs_order_;
  std::string name_;
};

// The actor set generated by graph transformer is the execution unit of actor runtime.
// It includes data source actor, kernel actor, switch actor, copy actor, loop count actor and output actor.
// The data source actor is used to obtain data and process them into device tensors, and send them to kernel actor.
// The kernel actor is used to receive the device tensors to luanch kernel. Specifically notice the no input
// kernel actor, it means that this actor has no input device tensor, need be triggered externally.
// The copy actor is used to convert the device tensor between the different device kernel.
// The loop count actor is used to receive the control of tail kernel actor to represent the end of one step
// and decide whether to loop execution by loop count.
// The output actor is used to receive the output result of actor which represents the graph output.
struct ActorSet {
  explicit ActorSet(const ActorInfo &name) : name_(name) {}
  std::vector<DataSourceActorPtr> data_source_actors_;
  std::vector<KernelActorPtr> kernel_actors_;
  // No input kernel actors need be triggered specifically.
  std::vector<KernelActorPtr> no_input_kernel_actors_;
  std::vector<SwitchActorPtr> switch_actors_;
  std::vector<CopyActorPtr> copy_actors_;
  LoopCountActorPtr loop_count_actor_{nullptr};
  OutputActorPtr output_actor_{nullptr};
  ActorInfo name_;
};
using ActorSetPtr = std::shared_ptr<ActorSet>;

class GraphScheduler {
 public:
  static GraphScheduler &GetInstance() {
    static GraphScheduler instance;
    return instance;
  }

  // 1. Thread pool creating.
  // 2. The memory manager creating and scheduling.
  void Initialize();

  // Transform graph to actor DAG, contains build and link.
  ActorSet *Transform(const GraphCompilerInfo &graph_compiler_info,
                      GraphExecutionStrategy strategy = GraphExecutionStrategy::kPipeline);

  // Schedule actors in the actor runtime. Single machine scheduling is supported currently, and distributed scheduling
  // will be supported in the future.
  void Schedule(const ActorSet *actor_set);

  // The prepare processing before run:
  // 1. Prepare the data of device tensor store(such as weights and value nodes of graph).
  // 2. Prepare the data of host tensor queue(such as non weighted parameters of graph).
  // 3. Prepare the continuous memory for communication kernel.
  void PrepareRun(const ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info,
                  const std::vector<std::vector<TensorPtr>> &input_tensors);

  // The processing entry of actors running.
  bool Run(const ActorSet *actor_set, GraphExecutionStrategy strategy = GraphExecutionStrategy::kPipeline,
           const std::vector<TensorPtr> *input_tensors = nullptr);

  // Fetch the actor set by actor info.
  ActorSet *Fetch(const ActorInfo &actor_info) const;

 private:
  GraphScheduler() = default;
  ~GraphScheduler();
  DISABLE_COPY_AND_ASSIGN(GraphScheduler);

  // Transform the nodes of graph to actors.
  ActorSetPtr Build(const GraphCompilerInfo &graph_compiler_info, GraphExecutionStrategy strategy);
  // Link actors to DAG through the edge connection of graph and graph execution strategy.
  void Link(ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info, GraphExecutionStrategy strategy);

  // The processing of actors build.
  std::vector<DataSourceActorPtr> BuildDataSourceActor(const GraphCompilerInfo &graph_compiler_info,
                                                       const HostTensorQueuePtr &host_queue);
  std::vector<KernelActorPtr> BuildKernelActor(const GraphCompilerInfo &graph_compiler_info);
  LoopCountActorPtr BuildLoopCountActor(const GraphCompilerInfo &graph_compiler_info, GraphExecutionStrategy strategy);
  OutputActorPtr BuildOutputActor(const GraphCompilerInfo &graph_compiler_info, GraphExecutionStrategy strategy);
  std::vector<KernelActorPtr> BuildNoInputKernelActor(const ActorSet *actor_set);

  // Cache the information of graph output node to actor between “build” and “link”, for linking between the tail of
  // previous graph and the head of next graph.
  void CacheGraphOutputToActor(const GraphCompilerInfo &graph_compiler_info);

  // The processing of actors link statically.
  // The gather of linking data allows of kernel, it will call following functions by the different from actor type.
  void LinkDataArrow(KernelActor *to_actor, const ActorSet *actor_set, const KernelGraphPtr &graph,
                     KernelWithIndex from_kernel_with_output_idx, KernelWithIndex to_kernel_with_input_idx,
                     const TensorPtr &tensor);
  // Link data arrows for internal parameter, convert internal parameter to actor by internal parameter cache to link.
  void LinkDataArrowForInternalParameter(const AnfNodePtr &internal_parameter, const KernelGraphPtr &graph,
                                         KernelActor *to_actor, KernelWithIndex to_kernel_with_input_idx);
  // Link data arrows in the copy actor scene, insert the copy actor between from_actor and to_actor.
  void LinkDataArrowForCopyActor(OpActor<DeviceTensor> *from_actor, KernelActor *to_actor,
                                 KernelWithIndex from_kernel_with_output_idx, KernelWithIndex to_kernel_with_input_idx);
  void LinkDataArrowForDeviceDSActor(DeviceQueueDataSourceActor *from_actor, KernelActor *to_actor,
                                     KernelWithIndex from_kernel_with_output_idx,
                                     KernelWithIndex to_to_kernel_with_input_idx);
  void LinkDataArrowForHostDSActor(HostQueueDataSourceActor *from_actor, KernelActor *to_actor,
                                   KernelWithIndex from_kernel_with_output_idx,
                                   KernelWithIndex to_kernel_with_input_idx);
  void LinkDataArrowForKernelActor(KernelActor *from_actor, KernelActor *to_actor,
                                   KernelWithIndex from_kernel_with_output_idx,
                                   KernelWithIndex to_kernel_with_input_idx);
  void LinkControlArrowForKernelActor(std::vector<KernelActorPtr> *from_actors, LoopCountActor *to_actor,
                                      GraphExecutionStrategy strategy);
  void LinkControlArrowForLoopCountActor(LoopCountActor *loop_count_actor, const ActorSet *actor_set,
                                         GraphExecutionStrategy strategy);
  void LinkControlArrowByAutoMonad(KernelActor *to_actor, const AnfNodePtr &from_node);
  void LinkOutputResultArrowForOutputActor(OutputActor *to_actor, const GraphCompilerInfo &graph_compiler_info);
  void LinkDeviceTensorStoreForAutoMonadActor(const std::vector<KernelActor *> &auto_monad_actors);

  // The processing of actors link dynamically.
  // Analyze necessary input data of current actor, generate and cache op arrow
  // between current actor and prev actor, the method executes before calling Schedule.
  void PrepareForDynamiclyLink(ActorSet *actor_set, const CNodePtr &kernel, const AID &aid,
                               const std::vector<TensorPtr> *input_tensors);
  // Link to prev actor dynamically, and send message to prev actor to add the
  // new DataArrow and send output data back, the method must execute after calling Schedule.
  void LinkDataArrowForKernelActorDynamicly(const ActorSet *actor_set);

  // Check whether the actor set is valid.
  bool CheckActorValid(const ActorSet *actor_set,
                       GraphExecutionStrategy strategy = GraphExecutionStrategy::kPipeline) const;

  // Persist device tensors of graph's some nodes(such as weights and value nodes).
  void PersistDeviceTensor(const GraphCompilerInfo &graph_compiler_info);

  // Fetch the hsot tensor queue by actor info.
  HostTensorQueue *FetchHostQueue(const ActorInfo &actor_info) const;

  // The operation of the map of actor_name_to_actor_.
  void InsertActor(OpActor<DeviceTensor> *actor);
  OpActor<DeviceTensor> *FetchActor(const std::string actor_name) const;

  // Display the actor information of corresponding kernel graph.
  void DumpActor(const ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info) const;
  void DumpBaseActor(const OpActor<DeviceTensor> *actor, std::ofstream &ofs) const;
  void DumpDSActor(const DataSourceActor *actor, std::ofstream &ofs) const;
  void DumpLoopCountActor(const LoopCountActor *actor, std::ofstream &ofs) const;
  void DumpKernelActor(const KernelActor *actor, std::ofstream &ofs) const;
  void DumpOutputActor(const OutputActor *actor, std::ofstream &ofs) const;
  void DumpCopyActor(const CopyActor *actor, std::ofstream &ofs) const;
  void DumpDeviceTensorStore(const GraphCompilerInfo &graph_compiler_info, std::ofstream &ofs) const;

  // The global maps, only be cleared in the deconstruction.
  std::unordered_map<ActorInfo, ActorSetPtr> actors_;
  std::unordered_map<std::string, OpActor<DeviceTensor> *> actor_name_to_actor_;
  std::unordered_map<ActorInfo, HostTensorQueuePtr> actor_to_host_queue_;
  // The second element of pair represents the output index of op actor corresponding to the device tensor.
  std::unordered_map<DeviceTensorPtr, GraphOutputPair> device_tensor_to_actor_;

  // The local maps and vectors, will be cleared at the beginning of each graph transform.
  // The second element of pair represents the output index of op actor corresponding to the graph output front node.
  std::map<KernelWithIndex, GraphOutputPair, session::KernelWithIndexCmp> graph_output_to_actor_;
  // Beaceuse the copy actors are built in the link, so need record the all copy actors in the link process to push into
  // the actor set after link.
  std::vector<CopyActorPtr> copy_actors_;

  // The id of memory manager actor.
  AID memory_manager_aid_;

  bool init_{false};
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_GRAPH_SCHEDULER_H_
