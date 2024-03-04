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
#include "exe_graph/runtime/tiling_data.h"
#include <benchmark/benchmark.h>

namespace gert {
namespace {
struct TestData {
  int64_t a;
  int32_t b;
  int16_t c;
  int16_t d;
};
}
static void TilingData_AppendBasicType(benchmark::State &state) {
  auto data = TilingData::CreateCap(2 * 1024);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());

  for (auto _ : state) {
    tiling_data->Append(10);
    tiling_data->SetDataSize(0);
  }
}
BENCHMARK(TilingData_AppendBasicType);

static void TilingData_AppendStruct(benchmark::State &state) {
  auto data = TilingData::CreateCap(2048);
  auto tiling_data = reinterpret_cast<TilingData *>(data.get());
  TestData td {
      .a = 1024,
      .b = 512,
      .c = 256,
      .d = 128
  };

  for (auto _ : state) {
    tiling_data->Append(td);
    tiling_data->SetDataSize(0);
  }
}
BENCHMARK(TilingData_AppendStruct);

}

BENCHMARK_MAIN();