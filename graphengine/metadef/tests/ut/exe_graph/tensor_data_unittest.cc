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
#include "exe_graph/runtime/tensor_data.h"
#include <gtest/gtest.h>
namespace gert {
namespace {
template<size_t N>
class ManagerStub {
 public:
  static ge::graphStatus Success(TensorAddress addr, TensorOperateType operate_type, void **out) {
    operate_count[operate_type]++;
    if (operate_type == kGetTensorAddress) {
      *out = reinterpret_cast<void *>(N);
    }
    return ge::GRAPH_SUCCESS;
  }
  static ge::graphStatus Failed(TensorAddress addr, TensorOperateType operate_type, void **out) {
    return ge::GRAPH_FAILED;
  }
  static ge::graphStatus FreeFailed(TensorAddress addr, TensorOperateType operate_type, void **out) {
    operate_count[operate_type]++;
    if (operate_type == kFreeTensor) {
      return ge::GRAPH_FAILED;
    }
    return Success(addr, operate_type, out);
  }
  static void Clear() {
    memset(operate_count, 0, sizeof(operate_count));
  }
  static size_t operate_count[kTensorOperateType];
};

template<size_t N>
size_t ManagerStub<N>::operate_count[kTensorOperateType] = {0};
}  // namespace

class TensorDataUT : public testing::Test {};

TEST_F(TensorDataUT, TensorDataWithMangerSuccess) {
  ManagerStub<8>::Clear();

  auto addr = reinterpret_cast<void *>(0x16);
  {
    TensorData data(addr, ManagerStub<8>::Success);
    EXPECT_EQ(reinterpret_cast<uint64_t>(data.GetAddr()), 8);
    EXPECT_EQ(data.Free(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ManagerStub<8>::operate_count[kFreeTensor], 1);

    EXPECT_EQ(data.GetAddr(), nullptr);
    data.SetAddr(addr, nullptr);
    EXPECT_EQ(reinterpret_cast<uint64_t>(data.GetAddr()), 0x16);
    data.SetAddr(addr, ManagerStub<8>::Failed);
    EXPECT_EQ(data.GetAddr(), nullptr);
  }
  EXPECT_EQ(ManagerStub<8>::operate_count[kFreeTensor], 1);
}

TEST_F(TensorDataUT, TransferOwner) {
  ManagerStub<8>::Clear();

  auto addr = reinterpret_cast<void *>(0x16);
  {
    TensorData td0(addr, nullptr);
    TensorData td1(addr, ManagerStub<8>::Success);
    td0 = std::move(td1);
  }
  EXPECT_EQ(ManagerStub<8>::operate_count[kPlusShareCount], 0);
  EXPECT_EQ(ManagerStub<8>::operate_count[kFreeTensor], 1);
}

TEST_F(TensorDataUT, TransferOnwerFreeOld) {
  ManagerStub<8>::Clear();
  ManagerStub<18>::Clear();

  auto addr = reinterpret_cast<void *>(0x16);
  {
    TensorData td0(addr, ManagerStub<18>::Success);
    TensorData td1(addr, ManagerStub<8>::Success);
    EXPECT_EQ(ManagerStub<18>::operate_count[kFreeTensor], 0);
    td0 = std::move(td1);
    EXPECT_EQ(ManagerStub<18>::operate_count[kFreeTensor], 1);

  }
  EXPECT_EQ(ManagerStub<8>::operate_count[kPlusShareCount], 0);
  EXPECT_EQ(ManagerStub<8>::operate_count[kFreeTensor], 1);
}

TEST_F(TensorDataUT, ConstructRightValue) {
  ManagerStub<8>::Clear();

  auto addr = reinterpret_cast<void *>(0x16);
  {
    TensorData td0(addr, ManagerStub<8>::Success);
    TensorData td1(std::move(td0));
  }
  EXPECT_EQ(ManagerStub<8>::operate_count[kPlusShareCount], 0);
  EXPECT_EQ(ManagerStub<8>::operate_count[kFreeTensor], 1);
}

TEST_F(TensorDataUT, FreeByHand) {
  ManagerStub<8>::Clear();

  auto addr = reinterpret_cast<void *>(0x16);
  {
    TensorData td0(addr, ManagerStub<8>::Success);
    EXPECT_EQ(td0.Free(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ManagerStub<8>::operate_count[kFreeTensor], 1);
  }
  EXPECT_EQ(ManagerStub<8>::operate_count[kPlusShareCount], 0);
  EXPECT_EQ(ManagerStub<8>::operate_count[kFreeTensor], 1);
}

TEST_F(TensorDataUT, TensorDataWithMangerFreeSuccess) {
  ManagerStub<8>::Clear();
  {
    auto addr = reinterpret_cast<void *>(0x16);
    TensorData data(addr, ManagerStub<8>::Success);
    EXPECT_EQ(reinterpret_cast<uint64_t>(data.GetAddr()), 8);
    EXPECT_EQ(data.Free(), ge::GRAPH_SUCCESS);
  }
  EXPECT_EQ(ManagerStub<8>::operate_count[kFreeTensor], 1);
}

TEST_F(TensorDataUT, ShareTensorDataOk) {
  TensorData td;
  td.SetAddr(reinterpret_cast<TensorAddress>(10), nullptr);

  TensorData td1;
  td1.SetAddr(reinterpret_cast<TensorAddress>(11), nullptr);
  ASSERT_EQ(td1.GetAddr(), reinterpret_cast<TensorAddress>(11));

  td1.ShareFrom(td);
  EXPECT_EQ(td1.GetAddr(), reinterpret_cast<TensorAddress>(10));
}

TEST_F(TensorDataUT, ReleaseBeforeShareTensorData) {
  ManagerStub<8>::Clear();

  TensorData td;
  td.SetAddr(reinterpret_cast<TensorAddress>(10), nullptr);

  TensorData td1;
  td1.SetAddr(reinterpret_cast<TensorAddress>(11), ManagerStub<8>::Success);
  ASSERT_EQ(td1.GetAddr(), reinterpret_cast<TensorAddress>(8));

  ASSERT_EQ(ManagerStub<8>::operate_count[kFreeTensor], 0);
  td1.ShareFrom(td);
  EXPECT_EQ(td1.GetAddr(), reinterpret_cast<TensorAddress>(10));
  EXPECT_EQ(ManagerStub<8>::operate_count[kFreeTensor], 1);
}

TEST_F(TensorDataUT, ShareManagedTensorData) {
  ManagerStub<8>::Clear();

  {
    TensorData td;
    td.SetAddr(reinterpret_cast<TensorAddress>(10), ManagerStub<8>::Success);

    ASSERT_EQ(ManagerStub<8>::operate_count[kPlusShareCount], 0);
    ASSERT_EQ(ManagerStub<8>::operate_count[kFreeTensor], 0);
    {
      TensorData td1;
      td1.SetAddr(reinterpret_cast<TensorAddress>(11), nullptr);

      td1.ShareFrom(td);
      ASSERT_EQ(ManagerStub<8>::operate_count[kPlusShareCount], 1);
      EXPECT_EQ(td1.GetAddr(), reinterpret_cast<TensorAddress>(8));
    }
    ASSERT_EQ(ManagerStub<8>::operate_count[kPlusShareCount], 1);
    ASSERT_EQ(ManagerStub<8>::operate_count[kFreeTensor], 1);
  }
  ASSERT_EQ(ManagerStub<8>::operate_count[kFreeTensor], 2);
}

TEST_F(TensorDataUT, ReleaseBeforeShareManagedTensorData) {
  ManagerStub<8>::Clear();
  ManagerStub<18>::Clear();

  TensorData td;
  EXPECT_EQ(td.SetAddr(reinterpret_cast<TensorAddress>(10), ManagerStub<8>::Success), ge::GRAPH_SUCCESS);
  TensorData td1;
  EXPECT_EQ(td1.SetAddr(reinterpret_cast<TensorAddress>(11), ManagerStub<18>::Success), ge::GRAPH_SUCCESS);

  ASSERT_EQ(ManagerStub<8>::operate_count[kPlusShareCount], 0);
  ASSERT_EQ(ManagerStub<18>::operate_count[kFreeTensor], 0);
  td1.ShareFrom(td);
  ASSERT_EQ(ManagerStub<8>::operate_count[kPlusShareCount], 1);
  EXPECT_EQ(ManagerStub<18>::operate_count[kFreeTensor], 1);
  EXPECT_EQ(td1.GetAddr(), reinterpret_cast<TensorAddress>(8));

  ASSERT_EQ(ManagerStub<8>::operate_count[kFreeTensor], 0);
  EXPECT_EQ(td1.Free(), ge::GRAPH_SUCCESS);
  EXPECT_EQ(ManagerStub<8>::operate_count[kFreeTensor], 1);
}

TEST_F(TensorDataUT, ReleaseBeforeSet) {
  ManagerStub<8>::Clear();

  TensorData td;
  EXPECT_EQ(td.SetAddr(reinterpret_cast<TensorAddress>(10), ManagerStub<8>::Success), ge::GRAPH_SUCCESS);

  ASSERT_EQ(ManagerStub<8>::operate_count[kFreeTensor], 0);
  EXPECT_EQ(td.SetAddr(reinterpret_cast<TensorAddress>(100), nullptr), ge::GRAPH_SUCCESS);
  EXPECT_EQ(ManagerStub<8>::operate_count[kFreeTensor], 1);
}

TEST_F(TensorDataUT, ReleaseFailedWhenSet) {
  ManagerStub<8>::Clear();

  TensorData td;
  EXPECT_EQ(td.SetAddr(reinterpret_cast<TensorAddress>(10), ManagerStub<8>::FreeFailed), ge::GRAPH_SUCCESS);
  ASSERT_EQ(td.GetAddr(), reinterpret_cast<TensorAddress>(8));

  ASSERT_EQ(ManagerStub<8>::operate_count[kFreeTensor], 0);
  EXPECT_NE(td.SetAddr(reinterpret_cast<TensorAddress>(100), nullptr), ge::GRAPH_SUCCESS);
  EXPECT_EQ(ManagerStub<8>::operate_count[kFreeTensor], 1);
  EXPECT_EQ(td.GetAddr(), reinterpret_cast<TensorAddress>(8));
}

TEST_F(TensorDataUT, ReleaseFailedWhenShare) {
  ManagerStub<8>::Clear();

  TensorData td;
  td.SetAddr(reinterpret_cast<TensorAddress>(10), nullptr);

  TensorData td1;
  td1.SetAddr(reinterpret_cast<TensorAddress>(11), ManagerStub<8>::FreeFailed);
  ASSERT_EQ(td1.GetAddr(), reinterpret_cast<TensorAddress>(8));

  ASSERT_EQ(ManagerStub<8>::operate_count[kFreeTensor], 0);
  EXPECT_NE(td1.ShareFrom(td), ge::GRAPH_SUCCESS);
  EXPECT_EQ(ManagerStub<8>::operate_count[kFreeTensor], 1);
  EXPECT_EQ(td1.GetAddr(), reinterpret_cast<TensorAddress>(8));
}

TEST_F(TensorDataUT, ShareFromSelf) {
  ManagerStub<8>::Clear();

  auto addr1 = reinterpret_cast<void *>(0x16);
  auto addr2 = reinterpret_cast<void *>(0x26);
  {
    TensorData td0(addr1, nullptr);
    TensorData td1(addr2, ManagerStub<8>::Success);
    EXPECT_FALSE(td0.IsSharedWith(td1));
    td0.ShareFrom(td1);
    EXPECT_TRUE(td0.IsSharedWith(td0));
    EXPECT_EQ(td0.GetAddr(), td1.GetAddr());
    EXPECT_NE(td0.GetAddr(), nullptr);

    td0.ShareFrom(td0);
    EXPECT_EQ(td0.GetAddr(), td1.GetAddr());
    EXPECT_NE(td0.GetAddr(), nullptr);
  }
  EXPECT_EQ(ManagerStub<8>::operate_count[kPlusShareCount], 1);
  EXPECT_EQ(ManagerStub<8>::operate_count[kFreeTensor], 2);
}

TEST_F(TensorDataUT, InitValue) {
  TensorData td(reinterpret_cast<TensorAddress>(10), nullptr);
  EXPECT_EQ(td.GetPlacement(), kTensorPlacementEnd);
  EXPECT_EQ(td.GetSize(), 0U);

  td.SetPlacement(kOnHost);
  EXPECT_EQ(td.GetPlacement(), kOnHost);

  td.SetSize(10);
  EXPECT_EQ(td.GetSize(), 10);

  // test move construct
  TensorData td1(std::move(td));
  EXPECT_EQ(td1.GetPlacement(), kOnHost);
  EXPECT_EQ(td1.GetSize(), 10);

  // test operator=
  TensorData td2 = std::move(td1);
  EXPECT_EQ(td2.GetPlacement(), kOnHost);
  EXPECT_EQ(td2.GetSize(), 10);

  EXPECT_EQ(td.GetPlacement(), kTensorPlacementEnd);
  EXPECT_EQ(td.GetSize(), 0);
}

TEST_F(TensorDataUT, GetPlacementStr_Success) {
  EXPECT_STREQ(GetPlacementStr(kOnHost), "HostDDR");
  EXPECT_STREQ(GetPlacementStr(kOnDeviceHbm), "DeviceHbm");
  EXPECT_STREQ(GetPlacementStr(kFollowing), "HostDDR");
  EXPECT_STREQ(GetPlacementStr(kTensorPlacementEnd), "Unknown");
}
}  // namespace gert