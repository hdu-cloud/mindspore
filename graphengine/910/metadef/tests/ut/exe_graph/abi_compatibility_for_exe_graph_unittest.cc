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
#include <gtest/gtest.h>
#define protected public
#define private public
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "exe_graph/runtime/runtime_attrs_def.h"
#include "exe_graph/runtime/infer_shape_range_context.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "exe_graph/runtime/atomic_clean_tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "exe_graph/runtime/tiling_data.h"

namespace gert {
namespace {
constexpr const size_t kPointerSize = 8U;
constexpr const size_t kReservedFieldSize = 40U;
constexpr const size_t kComputeNodeInfoReservedFieldSize = 24U;
constexpr const size_t kExtendInfoReservedFieldSize = 56U;

constexpr const size_t kShapeSize = 248U;
constexpr const size_t kExpandDimsTypeSize = 48U;
constexpr const size_t kStorageShapeSize = 536U;
constexpr const size_t kStorageFormatSize = 96U;
constexpr const size_t kTensorDataSize = 72U;
constexpr const size_t kTensorSize = 752U;
constexpr const size_t kCompileTimeTensorDescSize = 144U;
constexpr const size_t kAnchorInstanceInfoSize = 48U;
constexpr const size_t kComputeNodeInfoSize = 88U;
constexpr const size_t kRuntimeAttrsSize = 8U;
constexpr const size_t kRuntimeAttrsDefSize = 48U;
constexpr const size_t kContinuousVectorSize = 64U;
constexpr const size_t kContinuousVectorVectorSize = 64U;
constexpr const size_t kChainSize = 16U;
constexpr const size_t kRangeSize = 56U;
constexpr const size_t kKernelExtendInfoSize = 72U;
constexpr const size_t kKernelRunContextSize = 48U;
constexpr const size_t kTilingDataSize = 64U;
}  // namespace
constexpr size_t Shape::kMaxDimNum;
class AbiCompatibilityForExeGraphUT : public testing::Test {};

TEST_F(AbiCompatibilityForExeGraphUT, Shape_CheckMemLayoutNotChanged) {
  Shape s;
  ASSERT_EQ(sizeof(s), kShapeSize);
  ASSERT_EQ(static_cast<void *>(&s), static_cast<void *>(&s.dim_num_));

  EXPECT_EQ(reinterpret_cast<uintptr_t>(&s.dims_) - reinterpret_cast<uintptr_t>(&s.dim_num_), sizeof(size_t));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&s.reserved_) - reinterpret_cast<uintptr_t>(&s.dims_), 25 * sizeof(int64_t));
  EXPECT_EQ(sizeof(s.reserved_), kReservedFieldSize);

  EXPECT_EQ(Shape::kMaxDimNum, 25);
}

TEST_F(AbiCompatibilityForExeGraphUT, StorageShape_CheckMemLayoutNotChanged) {
  StorageShape s;
  ASSERT_EQ(sizeof(s), kStorageShapeSize);
  ASSERT_EQ(static_cast<void *>(&s), static_cast<void *>(&s.origin_shape_));

  EXPECT_EQ(reinterpret_cast<uintptr_t>(&s.storage_shape_) - reinterpret_cast<uintptr_t>(&s.origin_shape_), kShapeSize);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&s.reserved_) - reinterpret_cast<uintptr_t>(&s.storage_shape_), kShapeSize);
  EXPECT_EQ(sizeof(s.reserved_), kReservedFieldSize);
}

TEST_F(AbiCompatibilityForExeGraphUT, ExpandDimsType_CheckMemLayoutNotChanged) {
  ExpandDimsType e;
  ASSERT_EQ(sizeof(e), kExpandDimsTypeSize);

  EXPECT_EQ(reinterpret_cast<uintptr_t>(&e.reserved_) - reinterpret_cast<uintptr_t>(&e), 8);
  EXPECT_EQ(sizeof(e.reserved_), kReservedFieldSize);
}

TEST_F(AbiCompatibilityForExeGraphUT, StorageFormat_CheckMemLayoutNotChanged) {
  StorageFormat s;
  ASSERT_EQ(sizeof(s), kStorageFormatSize);
  ASSERT_EQ(static_cast<void *>(&s), static_cast<void *>(&s.origin_format_));

  EXPECT_EQ(reinterpret_cast<uintptr_t>(&s.storage_format_) - reinterpret_cast<uintptr_t>(&s.origin_format_),
            sizeof(ge::Format));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&s.expand_dims_type_) - reinterpret_cast<uintptr_t>(&s.storage_format_),
            sizeof(ge::Format));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&s.reserved_) - reinterpret_cast<uintptr_t>(&s.expand_dims_type_),
            kExpandDimsTypeSize);
  EXPECT_EQ(sizeof(s.reserved_), kReservedFieldSize);
}

TEST_F(AbiCompatibilityForExeGraphUT, TensorData_CheckMemLayoutNotChanged) {
  TensorData t;
  ASSERT_EQ(sizeof(t), kTensorDataSize);
  ASSERT_EQ(static_cast<void *>(&t), static_cast<void *>(&t.addr_));

  EXPECT_EQ(reinterpret_cast<uintptr_t>(&t.manager_) - reinterpret_cast<uintptr_t>(&t.addr_), kPointerSize);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&t.size_) - reinterpret_cast<uintptr_t>(&t.manager_), kPointerSize);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&t.placement_) - reinterpret_cast<uintptr_t>(&t.size_), sizeof(size_t));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&t.reserved_0_) - reinterpret_cast<uintptr_t>(&t.placement_),
            sizeof(gert::TensorPlacement));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&t.reserved_1_) - reinterpret_cast<uintptr_t>(&t.reserved_0_),
            sizeof(uint32_t));
  EXPECT_EQ(sizeof(t.reserved_1_), kReservedFieldSize);
}

TEST_F(AbiCompatibilityForExeGraphUT, Tensor_CheckMemLayoutNotChanged) {
  Tensor t;
  ASSERT_EQ(sizeof(t), kTensorSize);
  ASSERT_EQ(static_cast<void *>(&t), static_cast<void *>(&t.storage_shape_));

  EXPECT_EQ(reinterpret_cast<uintptr_t>(&t.storage_format_) - reinterpret_cast<uintptr_t>(&t.storage_shape_),
            kStorageShapeSize);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&t.reserved_) - reinterpret_cast<uintptr_t>(&t.storage_format_),
            kStorageFormatSize);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&t.data_type_) - reinterpret_cast<uintptr_t>(&t.reserved_), 4);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&t.tensor_data_) - reinterpret_cast<uintptr_t>(&t.data_type_),
            sizeof(ge::DataType));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&t.reserved_field_) - reinterpret_cast<uintptr_t>(&t.tensor_data_),
            kTensorDataSize);
  EXPECT_EQ(sizeof(t.reserved_field_), kReservedFieldSize);
}

TEST_F(AbiCompatibilityForExeGraphUT, CompileTimeTensorDesc_CheckMemLayoutNotChanged) {
  CompileTimeTensorDesc t;
  ASSERT_EQ(sizeof(t), kCompileTimeTensorDescSize);
  ASSERT_EQ(static_cast<void *>(&t), static_cast<void *>(&t.data_type_));

  EXPECT_EQ(reinterpret_cast<uintptr_t>(&t.storage_format_) - reinterpret_cast<uintptr_t>(&t.data_type_), 8);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&t.reserved_) - reinterpret_cast<uintptr_t>(&t.storage_format_),
            kStorageFormatSize);
  EXPECT_EQ(sizeof(t.reserved_), kReservedFieldSize);
}

TEST_F(AbiCompatibilityForExeGraphUT, AnchorInstanceInfo_CheckMemLayoutNotChanged) {
  AnchorInstanceInfo a;
  ASSERT_EQ(sizeof(a), kAnchorInstanceInfoSize);
  ASSERT_EQ(static_cast<void *>(&a), static_cast<void *>(&a.instance_start_));

  EXPECT_EQ(reinterpret_cast<uintptr_t>(&a.instantiation_num_) - reinterpret_cast<uintptr_t>(&a.instance_start_), 4);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&a.reserved_) - reinterpret_cast<uintptr_t>(&a.instantiation_num_), 4);
  EXPECT_EQ(sizeof(a.reserved_), kReservedFieldSize);
}

TEST_F(AbiCompatibilityForExeGraphUT, ComputeNodeInfo_CheckMemLayoutNotChanged) {
  auto holder = malloc(kComputeNodeInfoSize);
  auto c = reinterpret_cast<ComputeNodeInfo *>(holder);

  ASSERT_EQ(static_cast<void *>(c), static_cast<void *>(&c->node_type_));
  ASSERT_EQ(reinterpret_cast<uintptr_t>(&c->place_holder) - reinterpret_cast<uintptr_t>(c), kComputeNodeInfoSize - 8);

  EXPECT_EQ(reinterpret_cast<uintptr_t>(&c->node_name_) - reinterpret_cast<uintptr_t>(&c->node_type_), sizeof(char *));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&c->ir_inputs_num_) - reinterpret_cast<uintptr_t>(&c->node_name_),
            sizeof(char *));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&c->inputs_num_) - reinterpret_cast<uintptr_t>(&c->ir_inputs_num_),
            sizeof(size_t));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&c->outputs_num_) - reinterpret_cast<uintptr_t>(&c->inputs_num_),
            sizeof(size_t));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&c->reserved_) - reinterpret_cast<uintptr_t>(&c->outputs_num_), sizeof(size_t) * 3);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&c->place_holder) - reinterpret_cast<uintptr_t>(&c->reserved_),
            kComputeNodeInfoReservedFieldSize);

  free(holder);
}

TEST_F(AbiCompatibilityForExeGraphUT, RuntimeAttrs_CheckMemLayoutNotChanged) {
  auto holder = malloc(kRuntimeAttrsSize);
  auto c = reinterpret_cast<RuntimeAttrs *>(holder);

  ASSERT_EQ(static_cast<void *>(c), static_cast<void *>(&c->placeholder_));
  EXPECT_EQ(sizeof(c->placeholder_), sizeof(uint64_t));

  free(holder);
}

TEST_F(AbiCompatibilityForExeGraphUT, RuntimeAttrsDef_CheckMemLayoutNotChanged) {
  RuntimeAttrsDef r;
  ASSERT_EQ(sizeof(r), kRuntimeAttrsDefSize);
  ASSERT_EQ(static_cast<void *>(&r), static_cast<void *>(&r.attr_num));

  EXPECT_EQ(reinterpret_cast<uintptr_t>(&r.reserved_) - reinterpret_cast<uintptr_t>(&r.attr_num), sizeof(size_t));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&r.offset) - reinterpret_cast<uintptr_t>(&r.reserved_), kReservedFieldSize);
  EXPECT_EQ(sizeof(r.offset), 0);
}

TEST_F(AbiCompatibilityForExeGraphUT, ContinuousVector_CheckMemLayoutNotChanged) {
  ContinuousVector c;
  ASSERT_EQ(sizeof(c), kContinuousVectorSize);
  ASSERT_EQ(static_cast<void *>(&c), static_cast<void *>(&c.capacity_));

  EXPECT_EQ(reinterpret_cast<uintptr_t>(&c.size_) - reinterpret_cast<uintptr_t>(&c.capacity_), sizeof(size_t));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&c.reserved_) - reinterpret_cast<uintptr_t>(&c.size_), sizeof(size_t));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&c.elements) - reinterpret_cast<uintptr_t>(&c.reserved_), kReservedFieldSize);
  EXPECT_EQ(sizeof(c.elements), 8);
}

TEST_F(AbiCompatibilityForExeGraphUT, TypedContinuousVector_CheckMemLayoutNotChanged) {
  TypedContinuousVector<int64_t> c;
  ASSERT_EQ(sizeof(c), kContinuousVectorSize);
  ASSERT_EQ(static_cast<void *>(&c), static_cast<void *>(&c.capacity_));
}

TEST_F(AbiCompatibilityForExeGraphUT, ContinuousVectorVector_CheckMemLayoutNotChanged) {
  ContinuousVectorVector c;
  ASSERT_EQ(sizeof(c), kContinuousVectorVectorSize);
  ASSERT_EQ(static_cast<void *>(&c), static_cast<void *>(&c.capacity_));

  EXPECT_EQ(reinterpret_cast<uintptr_t>(&c.size_) - reinterpret_cast<uintptr_t>(&c.capacity_), sizeof(size_t));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&c.reserved_) - reinterpret_cast<uintptr_t>(&c.size_), sizeof(size_t));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&c.offset_) - reinterpret_cast<uintptr_t>(&c.reserved_), kReservedFieldSize);
  EXPECT_EQ(sizeof(c.offset_), 8);
}

TEST_F(AbiCompatibilityForExeGraphUT, Chain_CheckMemLayoutNotChanged) {
  Chain c;
  ASSERT_EQ(sizeof(c), kChainSize);
  ASSERT_EQ(static_cast<void *>(&c), static_cast<void *>(&c.any_value_));

  EXPECT_EQ(sizeof(c.any_value_), kChainSize);
}

TEST_F(AbiCompatibilityForExeGraphUT, Range_CheckMemLayoutNotChanged) {
  Range<Shape> r;
  ASSERT_EQ(sizeof(r), kRangeSize);
  ASSERT_EQ(static_cast<void *>(&r), static_cast<void *>(&r.min_));

  EXPECT_EQ(reinterpret_cast<uintptr_t>(&r.max_) - reinterpret_cast<uintptr_t>(&r.min_), kPointerSize);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&r.reserved_) - reinterpret_cast<uintptr_t>(&r.max_), kPointerSize);
  EXPECT_EQ(sizeof(r.reserved_), kReservedFieldSize);
}

TEST_F(AbiCompatibilityForExeGraphUT, KernelExtendInfo_CheckMemLayoutNotChanged) {
  auto holder = malloc(kKernelExtendInfoSize);
  auto k = reinterpret_cast<KernelExtendInfo *>(holder);

  ASSERT_EQ(reinterpret_cast<uintptr_t>(&k->reserved_) - reinterpret_cast<uintptr_t>(k),
            kKernelExtendInfoSize - kExtendInfoReservedFieldSize);
  ASSERT_EQ(static_cast<void *>(k), static_cast<void *>(&k->kernel_name_));

  EXPECT_EQ(reinterpret_cast<uintptr_t>(&k->kernel_type_) - reinterpret_cast<uintptr_t>(&k->kernel_name_),
            kPointerSize);
  free(holder);
}

TEST_F(AbiCompatibilityForExeGraphUT, KernelRunContext_CheckMemLayoutNotChanged) {
  KernelRunContext k;
  ASSERT_EQ(sizeof(KernelRunContext), kKernelRunContextSize);
  ASSERT_EQ(static_cast<void *>(&k), static_cast<void *>(&k.input_size));

  EXPECT_EQ(reinterpret_cast<uintptr_t>(&k.output_size) - reinterpret_cast<uintptr_t>(&k.input_size), sizeof(size_t));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&k.compute_node_info) - reinterpret_cast<uintptr_t>(&k.output_size),
            sizeof(size_t));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&k.kernel_extend_info) - reinterpret_cast<uintptr_t>(&k.compute_node_info),
            kPointerSize);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&k.output_start) - reinterpret_cast<uintptr_t>(&k.kernel_extend_info),
            kPointerSize);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&k.values) - reinterpret_cast<uintptr_t>(&k.output_start), kPointerSize);
  EXPECT_EQ(sizeof(k.values), kPointerSize);
}


TEST_F(AbiCompatibilityForExeGraphUT, KernelContext_CheckMemLayoutNotChanged) {
  KernelContext k;
  ASSERT_EQ(sizeof(KernelContext), kKernelRunContextSize);
  ASSERT_EQ(static_cast<void *>(&k), static_cast<void *>(&k.context_));
  EXPECT_EQ(sizeof(k.context_), kKernelRunContextSize);
}

TEST_F(AbiCompatibilityForExeGraphUT, ExtendedKernelContext_CheckMemLayoutNotChanged) {
  ExtendedKernelContext k;
  ASSERT_EQ(sizeof(ExtendedKernelContext), kKernelRunContextSize);
  ASSERT_EQ(static_cast<void *>(&k), static_cast<void *>(&k.context_));
  EXPECT_EQ(sizeof(k.context_), kKernelRunContextSize);
}

TEST_F(AbiCompatibilityForExeGraphUT, InferShapeContext_CheckMemLayoutNotChanged) {
  InferShapeContext c;
  ASSERT_EQ(sizeof(InferShapeContext), kKernelRunContextSize);
  ASSERT_EQ(static_cast<void *>(&c), static_cast<void *>(&c.context_));
  EXPECT_EQ(sizeof(c.context_), kKernelRunContextSize);
}

TEST_F(AbiCompatibilityForExeGraphUT, InferShapeRangeContext_CheckMemLayoutNotChanged) {
  InferShapeRangeContext c;
  ASSERT_EQ(sizeof(InferShapeRangeContext), kKernelRunContextSize);
  ASSERT_EQ(static_cast<void *>(&c), static_cast<void *>(&c.context_));
  EXPECT_EQ(sizeof(c.context_), kKernelRunContextSize);
}

TEST_F(AbiCompatibilityForExeGraphUT, InferDataTypeContext_CheckMemLayoutNotChanged) {
  InferDataTypeContext c;
  ASSERT_EQ(sizeof(InferDataTypeContext), kKernelRunContextSize);
  ASSERT_EQ(static_cast<void *>(&c), static_cast<void *>(&c.context_));
  EXPECT_EQ(sizeof(c.context_), kKernelRunContextSize);
}

TEST_F(AbiCompatibilityForExeGraphUT, TilingContext_CheckMemLayoutNotChanged) {
  TilingContext c;
  ASSERT_EQ(sizeof(TilingContext), kKernelRunContextSize);
  ASSERT_EQ(static_cast<void *>(&c), static_cast<void *>(&c.context_));
  EXPECT_EQ(sizeof(c.context_), kKernelRunContextSize);
}

TEST_F(AbiCompatibilityForExeGraphUT, TilingParseContext_CheckMemLayoutNotChanged) {
  TilingParseContext c;
  ASSERT_EQ(sizeof(TilingParseContext), kKernelRunContextSize);
  ASSERT_EQ(static_cast<void *>(&c), static_cast<void *>(&c.context_));
  EXPECT_EQ(sizeof(c.context_), kKernelRunContextSize);
}

TEST_F(AbiCompatibilityForExeGraphUT, AtomicCleanTilingContext_CheckMemLayoutNotChanged) {
  AtomicCleanTilingContext c;
  ASSERT_EQ(sizeof(AtomicCleanTilingContext), kKernelRunContextSize);
  ASSERT_EQ(static_cast<void *>(&c), static_cast<void *>(&c.context_));
  EXPECT_EQ(sizeof(c.context_), kKernelRunContextSize);
}

TEST_F(AbiCompatibilityForExeGraphUT, TilingData_CheckMemLayoutNotChanged) {
  auto holder = malloc(kTilingDataSize);
  auto t = reinterpret_cast<TilingData *>(holder);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(&t->reserved_) - reinterpret_cast<uintptr_t>(t),
            kTilingDataSize - kReservedFieldSize);
  ASSERT_EQ(static_cast<void *>(t), static_cast<void *>(&t->capacity_));

  EXPECT_EQ(reinterpret_cast<uintptr_t>(&t->data_size_) - reinterpret_cast<uintptr_t>(&t->capacity_), sizeof(size_t));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&t->data_) - reinterpret_cast<uintptr_t>(&t->data_size_), sizeof(size_t));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&t->reserved_) - reinterpret_cast<uintptr_t>(&t->data_), kPointerSize);
  EXPECT_EQ(sizeof(t->reserved_), kReservedFieldSize);

  free(holder);
}
}  // namespace gert