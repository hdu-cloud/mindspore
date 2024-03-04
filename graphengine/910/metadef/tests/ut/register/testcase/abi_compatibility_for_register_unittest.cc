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
#include "graph/any_value.h"
#include "register/op_impl_registry.h"
#include "register/op_impl_registry_base.h"
#include "register/kernel_registry_impl.h"

namespace gert {
namespace {
constexpr const size_t kPointerSize = 8U;
constexpr const size_t kVectorSize = 24U;
constexpr const size_t kUnorderedSetSize = 56U;
constexpr const size_t kMapSize = 48U;
constexpr const size_t kVirtualTableSize = 8U;
constexpr const size_t kReservedFieldSize = 16U;
constexpr const size_t kReservedFieldSize2 = 40U;

constexpr const size_t kOpImplFunctionsSize = 200U;
constexpr const size_t kOpImplRegistrySize = 88U + kVirtualTableSize;
constexpr const size_t kOpImplRegisterSize = 56U;
}  // namespace

constexpr size_t OpImplKernelRegistry::OpImplFunctions::kInt64ByteCount;
class AbiCompatibilityForRegisterUT : public testing::Test {};

TEST_F(AbiCompatibilityForRegisterUT, OpImplFunctions_CheckMemLayoutNotChanged) {
  OpImplKernelRegistry::OpImplFunctions f;
  ASSERT_EQ(sizeof(f), kOpImplFunctionsSize);
  ASSERT_EQ(static_cast<void *>(&f), static_cast<void *>(&f.infer_shape));

  EXPECT_EQ(reinterpret_cast<uintptr_t>(&f.infer_shape_range) - reinterpret_cast<uintptr_t>(&f.infer_shape),
            kPointerSize);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&f.infer_datatype) - reinterpret_cast<uintptr_t>(&f.infer_shape_range),
            kPointerSize);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&f.tiling) - reinterpret_cast<uintptr_t>(&f.infer_datatype), kPointerSize);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&f.tiling_parse) - reinterpret_cast<uintptr_t>(&f.tiling), kPointerSize);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&f.compile_info_creator) - reinterpret_cast<uintptr_t>(&f.tiling_parse),
            kPointerSize);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&f.compile_info_deleter) - reinterpret_cast<uintptr_t>(&f.compile_info_creator),
            kPointerSize);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&f.max_tiling_data_size) - reinterpret_cast<uintptr_t>(&f.compile_info_deleter),
            kPointerSize);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&f.inputs_dependency) - reinterpret_cast<uintptr_t>(&f.max_tiling_data_size),
            sizeof(size_t));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&f.private_attrs) - reinterpret_cast<uintptr_t>(&f.inputs_dependency),
            sizeof(uint64_t));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&f.unique_private_attrs) - reinterpret_cast<uintptr_t>(&f.private_attrs),
            kVectorSize);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&f.host_inputs) - reinterpret_cast<uintptr_t>(&f.unique_private_attrs),
            kUnorderedSetSize);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&f.op_execute_func) - reinterpret_cast<uintptr_t>(&f.host_inputs),
            sizeof(uint64_t));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&f.tiling_dependency) - reinterpret_cast<uintptr_t>(&f.op_execute_func),
            kPointerSize);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&f.reserved_0_) - reinterpret_cast<uintptr_t>(&f.tiling_dependency),
            sizeof(uint64_t));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(&f.reserved_1_) - reinterpret_cast<uintptr_t>(&f.reserved_0_), 7);

  EXPECT_EQ(sizeof(f.reserved_1_), kReservedFieldSize);
}

TEST_F(AbiCompatibilityForRegisterUT, OpImplRegistry_CheckMemLayoutNotChanged) {
  OpImplRegistry r;
  ASSERT_EQ(sizeof(r), kOpImplRegistrySize);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(&r.types_to_impl_) - reinterpret_cast<uintptr_t>(&r), kVirtualTableSize);

  EXPECT_EQ(reinterpret_cast<uintptr_t>(&r.reserved_) - reinterpret_cast<uintptr_t>(&r.types_to_impl_),
            kMapSize);
  EXPECT_EQ(sizeof(r.reserved_), kReservedFieldSize2);
}
}  // namespace gert
