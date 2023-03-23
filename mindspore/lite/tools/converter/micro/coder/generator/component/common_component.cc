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

#include "coder/generator/component/common_component.h"
#include <memory>
#include "coder/generator/component/component.h"
#include "coder/utils/type_cast.h"
#include "coder/utils/coder_utils.h"
#include "coder/log.h"
#include "include/errorcode.h"
#include "nnacl/op_base.h"
#include "include/c_api/model_c.h"

namespace mindspore::lite::micro {
const char micro_model_define_source[] = R"RAW(
typedef struct {
  void *runtime_buffer;
  bool train_mode;  // true: train mode, false: eval mode
  MSTensorHandleArray inputs;
  MSTensorHandleArray outputs;
} MicroModel;
)RAW";
const char model_runtime_malloc_source[] = R"RAW(
MSModelHandle MSModelCreate() {
  MicroModel *micro_model = (MicroModel *)malloc(sizeof(MicroModel));
  if (micro_model == NULL) {
    return NULL;
  }
  int buffer_size = GetBufferSize();
  void *runtime_buffer = malloc(buffer_size);
  if (runtime_buffer == NULL) {
    return NULL;
  }
  micro_model->runtime_buffer = runtime_buffer;
  int ret = SetBuffer(runtime_buffer);
  if (ret != kMSStatusSuccess) {
    return NULL;
  }

)RAW";

const char handle_array_destroy_state[] = R"RAW(
void MSTensorHandleArrayDestroy(MSTensorHandleArray inputs);
)RAW";

const char handle_array_destroy[] = R"RAW(
void MSTensorHandleArrayDestroy(MSTensorHandleArray inputs) {
  if (inputs.handle_list == NULL) {
    return;
  }
  for (size_t i = 0; i < inputs.handle_num; i++) {
    MicroTensor *micro_tensor = inputs.handle_list[i];
    if (!micro_tensor) {
      continue;
    }
    if (micro_tensor->data) {
      free(micro_tensor->data);
      micro_tensor->data = NULL;
    }
    if (micro_tensor->shape) {
      free(micro_tensor->shape);
      micro_tensor->shape = NULL;
    }
    free(micro_tensor);
    micro_tensor = NULL;
  }
  free(inputs.handle_list);
  inputs.handle_list = NULL;
}

)RAW";
const char cortex_set_workspace[] = R"RAW(
  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL) {
    return;
  }
  if (workspace_size < MSModelCalcWorkspaceSize(model)) {
    return;
  }
  if (micro_model->inputs.handle_num != GRAPH_INPUTS_SIZE) {
    return;
  }
  if (micro_model->outputs.handle_num != GRAPH_OUTPUTS_SIZE) {
    return;
  }

  micro_model->runtime_buffer = workspace;
  int buffer_size = GetBufferSize();
  char* buf = workspace;
  SetBuffer(buf);
  buffer_size = UP_ROUND(buffer_size, 4);
)RAW";
void CodeMSModelCalcWorkspaceSize(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx,
                                  const Configurator &config) {
  if (config.target() == kCortex_M) {
    ofs << "size_t MSModelCalcWorkspaceSize(MSModelHandle model) {\n";
    ofs << "  size_t shape_size=0;\n"
        << "  if (model == NULL) {\n"
        << "    return 0;\n"
        << "  }\n";
    std::vector<Tensor *> inputs = ctx->graph_inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
      ofs << "  shape_size += " << inputs[i]->shape().size() << " * sizeof(int64_t);\n";
    }
    std::vector<Tensor *> outputs = ctx->graph_outputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
      ofs << "  shape_size += " << outputs[i]->shape().size() << " * sizeof(int64_t);\n";
    }
    ofs << "  return UP_ROUND(GetBufferSize(),4) + UP_ROUND(WEIGHT_BUF_SIZE,4) + shape_size + "
        << "(UP_ROUND(sizeof(MicroTensor),4) + UP_ROUND(sizeof(MicroTensor *),4)) * "
        << (ctx->graph_inputs().size() + ctx->graph_outputs().size()) << ";\n}\n";
  } else {
    ofs << "size_t MSModelCalcWorkspaceSize(MSModelHandle model) {\n  return 0;\n}\n";
  }
  ofs << "\n";
}

void CodeMSModelSetWorkspace(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx, const Configurator &config) {
  ofs << "void MSModelSetWorkspace(MSModelHandle model, void *workspace, size_t workspace_size) {\n";
  if (config.target() == kCortex_M) {
    ofs << cortex_set_workspace;
    ofs << "  " << ctx->weight_name() << " = (uint8_t *)&buf[buffer_size];\n";
    ofs << R"RAW(
  buffer_size += WEIGHT_BUF_SIZE;
  buffer_size = UP_ROUND(buffer_size,4);

  micro_model->inputs.handle_list = (MSTensorHandle *)&buf[buffer_size];
  buffer_size +=  GRAPH_INPUTS_SIZE * sizeof(MicroTensor *);
  buffer_size = UP_ROUND(buffer_size,4);
  MicroTensor **input_tensors = (MicroTensor **)micro_model->inputs.handle_list;

  micro_model->outputs.handle_list = (MSTensorHandle *)&buf[buffer_size];
  buffer_size +=  GRAPH_OUTPUTS_SIZE * sizeof(MicroTensor *);
  buffer_size = UP_ROUND(buffer_size,4);
  MicroTensor **output_tensors = (MicroTensor **)micro_model->outputs.handle_list;
)RAW";
    ofs << "  int i;\n"
        << "  for (i = 0; i < GRAPH_INPUTS_SIZE; i++) {\n";
    std::vector<Tensor *> inputs = ctx->graph_inputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
      ofs << "    input_tensors[i] = (MicroTensor *)&buf[buffer_size];\n"
          << "    buffer_size += sizeof(MicroTensor);\n"
          << "    buffer_size = UP_ROUND(buffer_size,4);\n";
      ofs << "    input_tensors[i]->shape = (int64_t *)&buf[buffer_size];\n"
          << "    buffer_size += " << inputs[i]->shape().size() * sizeof(int64_t) << ";\n"
          << "    buffer_size = UP_ROUND(buffer_size,4);\n";
    }
    ofs << "  }\n";

    ofs << "  for (i = 0; i < GRAPH_OUTPUTS_SIZE; i++) {\n";
    std::vector<Tensor *> outputs = ctx->graph_outputs();
    for (size_t i = 0; i < outputs.size(); ++i) {
      ofs << "    output_tensors[i] = (MicroTensor *)&buf[buffer_size];\n"
          << "    buffer_size += sizeof(MicroTensor);\n"
          << "    buffer_size = UP_ROUND(buffer_size,4);\n";
      ofs << "    output_tensors[i]->shape = (int64_t *)&buf[buffer_size];\n"
          << "    buffer_size += " << outputs[i]->shape().size() * sizeof(int64_t) << ";\n"
          << "    buffer_size = UP_ROUND(buffer_size,4);\n";
    }
    ofs << "  }\n";
    ofs << R"RAW(
  if (buffer_size > workspace_size) {
    micro_model->runtime_buffer = NULL;
    SetBuffer(NULL);
    return;
  }
)RAW";
    auto array_tostring = [&ofs](Tensor *tensor, const std::string &prefix, size_t index) {
      ofs << kAlignedString << prefix << "_tensors[" << index << "]->type = " << EnumNameMSDataType(tensor->data_type())
          << ";\n";
      ofs << kAlignedString << prefix << "_tensors[" << index << "]->format = kMSFormatNHWC;\n";
      ofs << kAlignedString << prefix << "_tensors[" << index << "]->ndim = " << tensor->shape().size() << ";\n";
      size_t shape_size = tensor->shape().size();
      for (size_t i = 0; i < shape_size; i++) {
        ofs << kAlignedString << prefix << "_tensors[" << index << "]->shape[" << i << "]= " << tensor->shape()[i]
            << ";\n";
      }
      ofs << kAlignedString << prefix << "_tensors[" << index << "]->name = \"" << tensor->tensor_name() << "\";\n";
      ofs << kAlignedString << prefix << "_tensors[" << index << "]->data = NULL;\n";
    };
    for (size_t i = 0; i < inputs.size(); ++i) {
      array_tostring(inputs[i], "input", i);
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      array_tostring(outputs[i], "output", i);
    }
  }
  ofs << "}\n\n";
}

void CodeMSTensorHandleArrayDestroyState(std::ofstream &ofs, const Configurator &config) {
  if (config.target() != kCortex_M) {
    ofs << handle_array_destroy_state;
  }
}

void CodeMSModelCreate(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx, const Configurator &config) {
  ofs << micro_model_define_source;
  if (config.target() != kCortex_M) {
    ofs << model_runtime_malloc_source;
    if (config.code_mode() == CodeMode::Inference) {
      ofs << "  micro_model->train_mode = false;\n";
    } else if (config.code_mode() == CodeMode::Train) {
      ofs << "  micro_model->train_mode = true;\n";
    }
    auto array_tostring = [&ofs](Tensor *tensor, const std::string &prefix, size_t index) {
      ofs << kAlignedString << prefix << "_tensors[" << index << "] = malloc(sizeof(MicroTensor));\n";
      ofs << kAlignedString << prefix << "_tensors[" << index << "]->type = " << EnumNameMSDataType(tensor->data_type())
          << ";\n";
      ofs << kAlignedString << prefix << "_tensors[" << index << "]->format = kMSFormatNHWC;\n";
      ofs << kAlignedString << prefix << "_tensors[" << index << "]->ndim = " << tensor->shape().size() << ";\n";
      size_t shape_size = tensor->shape().size();
      ofs << kAlignedString << prefix << "_tensors[" << index << "]->shape = "
          << "malloc(" << shape_size << " * sizeof(int64_t));\n";
      for (size_t i = 0; i < shape_size; i++) {
        ofs << kAlignedString << prefix << "_tensors[" << index << "]->shape[" << i << "]= " << tensor->shape()[i]
            << ";\n";
      }
      ofs << kAlignedString << prefix << "_tensors[" << index << "]->name = \"" << tensor->tensor_name() << "\";\n";
      ofs << kAlignedString << prefix << "_tensors[" << index << "]->data = NULL;\n";
    };
    std::vector<Tensor *> inputs = ctx->graph_inputs();
    std::vector<Tensor *> outputs = ctx->graph_outputs();
    if (config.code_mode() == CodeMode::Inference) {
      outputs = ctx->graph_outputs();
    } else if (config.code_mode() == CodeMode::Train) {
      outputs = ctx->graph_train_outputs();
    }
    size_t inputs_size = inputs.size();
    ofs << "  MSTensorHandleArray model_inputs;\n";
    ofs << "  model_inputs.handle_num = " << inputs_size << ";\n";
    ofs << "  MicroTensor **input_tensors = malloc(" << inputs_size << " * sizeof(MicroTensor *));\n";
    ofs << "  model_inputs.handle_list = (MSTensorHandle *)(input_tensors);\n";
    ofs << "  micro_model->inputs = model_inputs;\n";
    for (size_t i = 0; i < inputs_size; ++i) {
      Tensor *input = inputs[i];
      array_tostring(input, "input", i);
    }
    size_t outputs_size = outputs.size();
    ofs << "  MSTensorHandleArray model_outputs;\n";
    ofs << "  model_outputs.handle_num = " << outputs_size << ";\n";
    ofs << "  MicroTensor **output_tensors = malloc(" << outputs_size << " * sizeof(MicroTensor *));\n";
    ofs << "  model_outputs.handle_list = (MSTensorHandle *)(output_tensors);\n";
    ofs << "  micro_model->outputs = model_outputs;\n";
    for (size_t i = 0; i < outputs_size; ++i) {
      Tensor *output = outputs[i];
      array_tostring(output, "output", i);
    }
    ofs << "  return (MSModelHandle)micro_model;\n";
  } else {
    ofs << "#define GRAPH_INPUTS_SIZE " << ctx->graph_inputs().size() << "\n";
    ofs << "#define GRAPH_OUTPUTS_SIZE " << ctx->graph_outputs().size() << "\n";
    ofs << "#define WEIGHT_BUF_SIZE " << ctx->weight_buffer_size() << "\n";
    ofs << "MSModelHandle MSModelCreate() {\n";
    ofs << "  static MicroModel model;\n";
    ofs << "  model.runtime_buffer = NULL;\n";
    ofs << "  model.inputs.handle_num = GRAPH_INPUTS_SIZE;\n";
    ofs << "  model.inputs.handle_list = NULL;\n";
    ofs << "  model.outputs.handle_num = GRAPH_OUTPUTS_SIZE;\n";
    ofs << "  model.outputs.handle_list = NULL;\n";
    ofs << "  model.train_mode = false;\n";
    ofs << "  return (MSModelHandle)&model;\n";
  }
  ofs << "}\n\n";
}

void CodeMSModelBuild(std::ofstream &ofs, const Configurator *config) {
  ofs
    << "MSStatus MSModelBuild(MSModelHandle model, const void *model_data, size_t data_size, MSModelType model_type,\n"
       "                      const MSContextHandle model_context) {\n"
       "  if (model_type != kMSModelTypeMindIR) {\n"
       "    return kMSStatusLiteNotSupport;\n"
       "  }\n"
       "  if (model == NULL) {\n"
       "    return kMSStatusLiteParamInvalid;\n"
       "  }\n"
       "  if (((MicroModel *)model)->runtime_buffer == NULL) {\n"
       "    return kMSStatusLiteMemoryFailed;\n"
       "  }\n";
  ofs << "  int ret = RET_OK;\n";
  if (config->target() != kCortex_M) {
    ofs << "  ret = Init((void*)model_data, data_size);\n";
  } else {
    ofs << "  ret = Init(NULL, 0);\n";
  }
  if (config->support_parallel()) {
    ofs << "  MicroContext *micro_context = (MicroContext *)model_context;\n"
           "  if (micro_context == NULL) {\n"
           "      return kMSStatusLiteNullptr;"
           "  }\n"
           "  ret = CreateThreadPool(micro_context->thread_num_);\n"
           "  if(ret != RET_OK) {\n"
           "     return ret;\n"
           "  }\n"
           "  ret = SetCoreAffinity(micro_context->affinity_mode);\n";
  }
  ofs << "  return ret;\n";
  ofs << "}\n";
}

void CodeMSModelDestory(std::ofstream &ofs, const Configurator *config) {
  if (config->target() != kCortex_M) {
    ofs << handle_array_destroy;
  }
  ofs << "void MSModelDestroy(MSModelHandle *model) {\n";
  if (config->target() != kCortex_M) {
    ofs << "  if (*model) {\n"
           "    MicroModel *micro_model = (MicroModel *)*model;\n";
    ofs << "    if (micro_model->runtime_buffer) {\n"
           "      free(micro_model->runtime_buffer);\n"
           "      micro_model->runtime_buffer = NULL;\n"
           "    }\n";
    ofs << "    MSTensorHandleArrayDestroy(micro_model->inputs);\n"
           "    MSTensorHandleArrayDestroy(micro_model->outputs);\n"
           "    free(*model);\n"
           "    *model = NULL;\n"
           "  }\n";

    if (config->support_parallel()) {
      ofs << "  ClearThreadPool();\n";
    }
  } else {
    ofs << "  if (*model) {\n"
           "    MicroModel *micro_model = (MicroModel *)*model;\n";
    ofs << "    micro_model->runtime_buffer = NULL;\n"
           "    *model = NULL;\n"
           "  }\n";
  }
  ofs << "}\n";
}

void CodeMSModelPredict(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  auto inputs_num = ctx->graph_inputs().size();
  auto outputs_num = ctx->graph_outputs().size();
  ofs << R"RAW(
MSStatus MSModelPredict(MSModelHandle model, const MSTensorHandleArray inputs, MSTensorHandleArray *outputs,
                        const MSKernelCallBackC before, const MSKernelCallBackC after) {
  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL) {
    return kMSStatusLiteNullptr;
  }
  if (micro_model->runtime_buffer == NULL) {
    return kMSStatusLiteMemoryFailed;
  }
)RAW";
  ofs << "  if (inputs.handle_num != " << inputs_num << ") {\n";
  ofs << "    return kMSStatusLiteParamInvalid;\n";
  ofs << "  }\n";
  ofs << "  if (outputs->handle_num != " << outputs_num << ") {\n";
  ofs << "    return kMSStatusLiteParamInvalid;\n";
  ofs << "  }\n";
  ofs << "  const void *inputs_data_array[" << inputs_num << "];\n";
  ofs << "  for (int i = 0; i < " << inputs_num << "; i++) {\n";
  ofs << "    inputs_data_array[i] = ((MicroTensor *)inputs.handle_list[i])->data;\n";
  ofs << "  }\n";
  ofs << "  SetInputs(inputs_data_array, " << inputs_num << ");\n";
  ofs << "\n";
  ofs << "  Execute(micro_model->train_mode);\n";
  ofs << "\n";
  ofs << "  void *outputs_data_array[" << outputs_num << "];\n";
  ofs << "  for (int i = 0; i < " << outputs_num << "; i++) {\n";
  ofs << "    outputs_data_array[i] = MSTensorGetMutableData(outputs->handle_list[i]);\n";
  ofs << "  }\n";
  ofs << "  CopyOutputsData(outputs_data_array, " << outputs_num << ");\n";
  ofs << "  return kMSStatusSuccess;\n";
  ofs << "}\n\n";
}

void CodeCopyOutputsState(std::ofstream &ofs) { ofs << "int CopyOutputsData(void **outputs, int num);\n\n"; }

void CodeCopyOutputsImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  auto tensor_map = ctx->tensors_map();
  std::vector<Tensor *> outputs = ctx->graph_outputs();
  size_t outputs_size = outputs.size();
  ofs << "int CopyOutputsData(void **outputs, int num) {\n"
         "  if (outputs == NULL) {\n"
         "    return RET_ERROR;\n"
         "  }\n"
      << "  if (num != " << outputs_size << ") {\n"
      << "    return RET_ERROR;\n"
         "  }\n";
  for (size_t i = 0; i < outputs_size; ++i) {
    Tensor *output = outputs[i];
    MS_CHECK_PTR_IF_NULL(output);
    ofs << "  memcpy(outputs[" << i << "], " << tensor_map[output] << ", " << output->Size() << ");\n";
  }
  ofs << "  return RET_OK;\n"
         "}\n\n";
}

void CodeGlobalCodeBlocks(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  for (const auto &block : ctx->global_code_blocks()) {
    ofs << block << "\n";
  }
}

void CodeInputState(std::ofstream &ofs) {
  ofs << "/**\n"
      << "  * set input tensors\n"
      << "  * @param inputs, the input data ptr's array of the model, the tensors' count of input may be greater than "
         "one.\n"
      << "  * @param num, the input data's number of the model.\n"
      << "  **/\n"
      << "int "
      << "SetInputs(const void **inputs, int num);\n\n";
}

void CodeInputImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  // input tensors
  std::vector<Tensor *> inputs = ctx->graph_inputs();
  for (size_t i = 0; i < inputs.size(); ++i) {
    ofs << "const unsigned char *" << ctx->input_name() + std::to_string(i) << " = 0;\n";
  }
  size_t size = inputs.size();
  ofs << "int "
      << "SetInputs(const void **inputs, int num) {\n"
      << "  if (inputs == NULL) {\n"
         "    return RET_ERROR;\n"
         "  }\n"
      << "  if (num !=" << size << ") {\n"
      << "    return RET_ERROR;\n"
         "  }\n";
  for (size_t i = 0; i < size; ++i) {
    ofs << "\t" << ctx->input_name() << i << " = (unsigned char *)inputs[" << i << "];\n";
  }
  ofs << "  return RET_OK;\n}\n";
}

void CodeGraphQuantArgsState(std::ofstream &ofs) {
  ofs << "/**\n"
      << "  * get input and output QuantArgs of the model \n"
      << "  **/\n"
      << "GraphQuantArgs "
      << "GetInOutQuantArgs();\n\n";
}

void CodeGraphQuantArgsImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  std::vector<Tensor *> graph_inputs = ctx->graph_inputs();
  if (graph_inputs.empty()) {
    MS_LOG(ERROR) << "graph input tensors' number is 0";
    return;
  }
  Tensor *in_tensor = graph_inputs.at(kInputIndex);
  MS_CHECK_PTR_IF_NULL(in_tensor);
  std::vector<Tensor *> graph_outputs = ctx->graph_outputs();
  if (graph_outputs.empty()) {
    MS_LOG(ERROR) << "graph output tensors' number is 0";
    return;
  }
  Tensor *out_tensor = graph_outputs.at(kOutputIndex);
  MS_CHECK_PTR_IF_NULL(out_tensor);
  std::vector<LiteQuantParam> in_quant_args = in_tensor->quant_params();
  std::vector<LiteQuantParam> out_quant_args = out_tensor->quant_params();
  if (in_quant_args.empty() || out_quant_args.empty()) {
    MS_LOG(ERROR) << "code model quant args failed";
    return;
  }
  ofs << "GraphQuantArgs "
      << "GetInOutQuantArgs() {\n"
      << "\t\tGraphQuantArgs quan_args = { " << in_quant_args.at(0).scale << ", " << out_quant_args.at(0).scale << ", "
      << in_quant_args.at(0).zeroPoint << ", " << out_quant_args.at(0).zeroPoint << "};\n"
      << "\t\treturn quan_args;\n"
      << "}\n";
}

void CodeManageResourceState(std::ofstream &ofs) {
  ofs << "/**\n"
      << "  * get the memory space size of the inference.\n"
      << "  **/\n"
      << "int "
      << "GetBufferSize();\n";

  ofs << "/**\n"
      << "  * set the memory space for the inference\n"
      << "  **/\n"
      << "int "
      << "SetBuffer(void *buffer);\n\n";

  ofs << "/**\n"
      << "  * free the memory of packed weights, and set the membuf buffer and input address to NULL\n"
      << "  **/\n"
      << "void "
      << "FreeResource();\n";
}

void CodeInitResourceImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx) {
  ofs << "int "
      << "GetBufferSize() {\n"
      << "  return " << ctx->total_buffer_size() << ";\n"
      << "}\n";
  ofs << "int "
      << "SetBuffer( void *buffer) {\n";
  ofs << "  " << ctx->buffer_name() << " = (unsigned char *)buffer;\n"
      << "  return RET_OK;\n"
         "}\n";
}

void CodeFreeResourceImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx,
                               const Configurator &config) {
  ofs << "void "
      << "FreeResource() {\n";
  if (config.target() != kCortex_M) {
    ofs << "  " << ctx->buffer_name() << "= NULL;\n";
    std::vector<Tensor *> inputs = ctx->graph_inputs();
    size_t size = inputs.size();
    for (size_t i = 0; i < size; ++i) {
      ofs << "  " << ctx->input_name() + std::to_string(i) << " = NULL;\n";
    }
    ofs << "  void **allocated[] = {\n";
    size_t num = 0;
    for (const auto &item : ctx->tensors_map()) {
      Tensor *tensor = item.first;
      std::string name = item.second;
      if (tensor->data() != nullptr && !(CheckConstantTensor(tensor))) {
        ofs << "    (void**)&" << name << ",\n";
        num++;
      }
    }
    ofs << "\n  };\n";
    ofs << "  for (int i = 0; i < " << num << "; ++i) {\n"
        << "    *(allocated[i]) = NULL;\n"
        << "  }\n";
    ofs << "  if (" << ctx->weight_name() << " != NULL) {\n";
    ofs << "    free(" << ctx->weight_name() << ");\n";
    ofs << "    " << ctx->weight_name() << " = NULL;\n  }\n";
  }
  ofs << "}\n";
}

void CodeExecuteState(std::ofstream &ofs) {
  ofs << "/**\n"
      << "  * net execute function\n"
      << "  **/\n"
      << "void "
      << "Execute(bool train_mode);\n\n";
}
}  // namespace mindspore::lite::micro
