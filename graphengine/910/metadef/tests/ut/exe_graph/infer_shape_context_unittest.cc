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
#include "exe_graph/runtime/infer_shape_context.h"
#include "graph/ge_error_codes.h"
#include <gtest/gtest.h>
#include "faker/kernel_run_context_faker.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/kernel_registry.h"
namespace gert {
class InferShapeContextUT : public testing::Test {};
TEST_F(InferShapeContextUT, GetInputShapeOk) {
  gert::StorageShape in_shape1 = {{8, 3, 224, 224}, {8, 1, 224, 224, 16}};
  gert::StorageShape in_shape2 = {{2, 2, 3, 8}, {8, 1, 2, 2, 16}};
  gert::StorageShape out_shape = {{8, 3, 224, 224}, {8, 1, 224, 224, 16}};
  auto context_holder = InferShapeContextFaker()
                            .IrInputNum(2)
                            .NodeIoNum(2, 1)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
                            .InputShapes({&in_shape1, &in_shape2})
                            .OutputShapes({&out_shape})
                            .Build();
  auto context = context_holder.GetContext<InferShapeContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetInputShape(0), nullptr);
  EXPECT_EQ(*context->GetInputShape(0), in_shape1.GetOriginShape());

  ASSERT_NE(context->GetInputShape(1), nullptr);
  EXPECT_EQ(*context->GetInputShape(1), in_shape2.GetOriginShape());

  EXPECT_EQ(context->GetInputShape(2), nullptr);
}

TEST_F(InferShapeContextUT, GetDynamicInputShapeOk) {
  gert::StorageShape in_shape1 = {{8, 3, 224, 224}, {8, 1, 224, 224, 16}};
  gert::StorageShape in_shape2 = {{2, 2, 3, 8}, {2, 2, 3, 8}};
  gert::StorageShape in_shape3 = {{3, 2, 3, 8}, {3, 2, 3, 8}};
  gert::StorageShape in_shape4 = {{4, 2, 3, 8}, {4, 2, 3, 8}};
  gert::StorageShape out_shape = {{8, 3, 224, 224}, {8, 1, 224, 224, 16}};
  auto context_holder = InferShapeContextFaker()
                            .IrInstanceNum({1, 2, 0, 1})
                            .NodeIoNum(4, 1)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                            .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                            .InputShapes({&in_shape1, &in_shape2, &in_shape3, &in_shape4})
                            .OutputShapes({&out_shape})
                            .Build();
  auto context = context_holder.GetContext<InferShapeContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetOptionalInputShape(0), nullptr);
  EXPECT_EQ(*context->GetOptionalInputShape(0), in_shape1.GetOriginShape());

  ASSERT_NE(context->GetDynamicInputShape(1, 0), nullptr);
  EXPECT_EQ(*context->GetDynamicInputShape(1, 0), in_shape2.GetOriginShape());

  ASSERT_NE(context->GetDynamicInputShape(1, 1), nullptr);
  EXPECT_EQ(*context->GetDynamicInputShape(1, 1), in_shape3.GetOriginShape());

  EXPECT_EQ(context->GetOptionalInputShape(2), nullptr);

  ASSERT_NE(context->GetOptionalInputShape(3), nullptr);
  EXPECT_EQ(*context->GetOptionalInputShape(3), in_shape4.GetOriginShape());
}

TEST_F(InferShapeContextUT, GetOutShapeOk) {
  gert::StorageShape in_shape1 = {{8, 3, 224, 224}, {8, 1, 224, 224, 16}};
  gert::StorageShape in_shape2 = {{2, 2, 3, 8}, {8, 1, 2, 2, 16}};
  gert::StorageShape out_shape = {{8, 3, 224, 224}, {8, 1, 224, 224, 16}};
  auto context_holder = InferShapeContextFaker()
                            .IrInputNum(2)
                            .NodeIoNum(2, 1)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_HWCN, ge::FORMAT_FRACTAL_Z)
                            .InputShapes({&in_shape1, &in_shape2})
                            .OutputShapes({&out_shape})
                            .Build();
  auto context = context_holder.GetContext<InferShapeContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetOutputShape(0), nullptr);
  EXPECT_EQ(*context->GetOutputShape(0), out_shape.GetOriginShape());

  EXPECT_EQ(context->GetOutputShape(1), nullptr);
}

TEST_F(InferShapeContextUT, GetInferenceContextPtrOK) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test0", "test1");
  const ge::GeTensorDesc tensor1(ge::GeShape({1,2,3,4}));
  const ge::GeTensorDesc tensor2(ge::GeShape({2,2,3,4}));
  const ge::GeTensorDesc tensor3(ge::GeShape({3,2,3,4}));
  ASSERT_EQ(op_desc->AddInputDesc(tensor1), ge::GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->AddInputDesc(tensor2), ge::GRAPH_SUCCESS);
  ASSERT_EQ(op_desc->AddOutputDesc(tensor3), ge::GRAPH_SUCCESS);
  KernelRunContextBuilder builder;
  gert::StorageShape shape1({1,2,3,4}, {1,2,3,4});
  gert::StorageShape shape2({2,2,3,4}, {2,2,3,4});
  gert::StorageShape shape3({3,2,3,4}, {3,2,3,4});
  auto inference_ctx = std::shared_ptr<ge::InferenceContext>(ge::InferenceContext::Create());
  KernelRegistry::KernelFunc kernel_func = [](KernelContext *context)->ge::graphStatus {
    return ge::GRAPH_SUCCESS;
  };
  auto holder = builder.Inputs({{&shape1, nullptr},
                                {&shape2, nullptr},
                                {reinterpret_cast<void *>(kernel_func), nullptr},
                                {inference_ctx.get(), nullptr}})
                    .Outputs({&shape3})
                    .Build(op_desc);
  auto infer_shape_ctx = reinterpret_cast<gert::InferShapeContext *>(holder.context_);
  ASSERT_NE(infer_shape_ctx, nullptr);
  const size_t inputs_num = infer_shape_ctx->GetComputeNodeInputNum();
  const size_t offset = inputs_num + static_cast<size_t>(InputExternLayout::kInferShapeFunc);
  EXPECT_EQ(reinterpret_cast<const void *>(holder.context_->GetInputValue<KernelRegistry::KernelFunc>(offset - 1U)),
            kernel_func);
  EXPECT_EQ(infer_shape_ctx->GetInferenceContextPtr(), inference_ctx.get());
}

TEST_F(InferShapeContextUT, GetOptionalInputTensorFailed_NotSetOptionalInput) {
  gert::StorageShape in_shape1 = {{8, 3, 224, 224}, {8, 1, 224, 224, 16}};
  auto infer_shape_func_addr = reinterpret_cast<void *>(0x11);

  auto context_holder = KernelRunContextFaker()
                            .IrInputNum(2)
                            .IrInstanceNum({1, 0})
                            .KernelIONum(2, 0)
                            .NodeIoNum(1, 0)
                            .Inputs({&in_shape1, infer_shape_func_addr})
                            .Build();
  auto context = context_holder.GetContext<InferShapeContext>();
  ASSERT_NE(context, nullptr);

  ASSERT_NE(context->GetInputShape(0), nullptr);
  EXPECT_EQ(*context->GetInputShape(0), in_shape1.GetOriginShape());

  EXPECT_EQ(context->GetOptionalInputTensor(1), nullptr);
  EXPECT_NE(context->GetInputTensor(1), nullptr);
}

TEST_F(InferShapeContextUT, GetOptionalInputTensorOK_SetOptionalInput) {
gert::StorageShape in_shape1 = {{8, 3, 224, 224}, {8, 1, 224, 224, 16}};
gert::Tensor in_tensor_2 = {{{1, 16, 256}, {1, 16, 256}},                // shape
                            {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, {}},  // format
                            kOnDeviceHbm,                                // placement
                            ge::DT_FLOAT16,                              // data type
                            (void *) 0xabc};
auto infer_shape_func_addr = reinterpret_cast<void *>(0x11);

auto context_holder = KernelRunContextFaker()
    .IrInputNum(2)
    .IrInstanceNum({1, 1})
    .KernelIONum(3, 0)
    .NodeIoNum(2, 0)
    .Inputs({&in_shape1, &in_tensor_2, infer_shape_func_addr})
    .Build();
auto context = context_holder.GetContext<InferShapeContext>();
ASSERT_NE(context, nullptr);

ASSERT_NE(context->GetInputShape(0), nullptr);
EXPECT_EQ(*context->GetInputShape(0), in_shape1.GetOriginShape());

EXPECT_NE(context->GetOptionalInputTensor(1), nullptr);
EXPECT_NE(context->GetInputTensor(2), nullptr);
}
}  // namespace gert