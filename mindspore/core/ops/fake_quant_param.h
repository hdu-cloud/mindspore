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

#ifndef MINDSPORE_CORE_OPS_FAKE_QUANT_PARAM_H_
#define MINDSPORE_CORE_OPS_FAKE_QUANT_PARAM_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>

#include "ops/base_operator.h"
#include "ir/dtype/type_id.h"
#include "mindapi/ir/common.h"

namespace mindspore {
namespace ops {
constexpr auto kNameFakeQuantParam = "FakeQuantParam";

constexpr auto kAttrKeyQuantAlgoName = "quant_algo_name";
constexpr auto kAttrKeyQuantDType = "quant_dtype";
constexpr auto kAttrKeyLinearQuantAlgoName = "linear_quant_algo";
constexpr auto kAttrKeyQuantParamIsPerChannel = "is_per_channel";

constexpr auto kAttrKeyLinearQuantParamScale = "linear_quant_scale";
constexpr auto kAttrKeyLinearQuantParamZeroPoint = "linear_quant_zero_point";

/// \brief the FakeQuantParam operator prototype. FakeQuantParam is an operator for storing quant parameter.
class MIND_API FakeQuantParam : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(FakeQuantParam);
  /// \brief Constructor.
  FakeQuantParam() : BaseOperator(kNameFakeQuantParam) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] quant_dtype Define quant data type of quant algorithm.
  /// \param[in] quant_algo_name Define the name of quant algorithm. Use `kAttrKeyLinearQuantAlgoName` for linear quant
  /// algorithm.
  /// \param[in] is_perchannel Define whether quant parameter is per-channel or per-layer.
  void Init(const QuantDataType &quant_dtype, const std::string &quant_algo_name, const bool is_perchannel);

  /// \brief Method to set quant_dtype attribute.
  ///
  /// \param[in] quant_dtype Define quant data type of quant algorithm.
  void set_quant_dtype(const QuantDataType &quant_dtype);

  /// \brief Method to get quant_dtype attribute.
  ///
  /// \return a TypeId represents quant data type of quant algorithm.
  QuantDataType get_quant_dtype() const;

  /// \brief Method to set quant_algo_name attribute.
  ///
  /// \param[in] quant_algo_name Define the name of quant algorithm. Use `kAttrKeyLinearQuantAlgoName` for linear quant
  /// algorithm.
  void set_quant_algo_name(const std::string &quant_algo_name);

  /// \brief Method to get quant_algo_name attribute.
  ///
  /// \return a string represents name of quant algorithm.
  std::string get_quant_algo_name() const;

  /// \brief Method to set is_perchannel attribute.
  ///
  /// \param[in] is_perchannel Define whether quant parameter is per-channel or per-layer.
  void set_is_perchannel(const bool is_perchannel);

  /// \brief Method to get is_perchannel attribute.
  ///
  /// \return a bool represents whether quant parameter is per-channel or per-layer.
  bool get_is_perchannel() const;

  /// \brief Method to add or set quant parameter named `key`.
  ///
  /// \param[in] key Define the name of quant parameter.
  /// \param[in] param Define the value of quant parameter.
  /// \param[in] channel_index Define the index indicates which channel the quant parameter belong to. Default is 0
  /// indicating first channel.
  void set_quant_param(const std::string &key, api::ValuePtr param, size_t channel_index = 0);

  /// \brief Method to get quant parameter named `key`.
  ///
  /// \param[in] key Define the name of quant parameter.
  /// \param[in] channel_index Define the index indicates which channel the quant parameter belong to. Default is 0
  /// indicating first channel.
  ///
  /// \return a ValuePtr represents quant parameter.
  api::ValuePtr get_quant_param(const std::string &key, size_t channel_index = 0) const;

  /// \brief Method to get quant parameter named `scale` for linear algorithm.
  ///
  /// \param[in] scale Define the value of quant parameter.
  /// \param[in] channel_index Define the index indicates which channel the quant parameter belong to. Default is 0
  /// indicating first channel.
  void set_scale(double scale, size_t channel_index = 0);

  /// \brief Method to get quant parameter named `scale` for linear algorithm.
  ///
  /// \param[in] channel_index Define the index indicates which channel the quant parameter belong to. Default is 0
  /// indicating first channel.
  ///
  /// \return a double as scale.
  double get_scale(size_t channel_index = 0) const;

  /// \brief Method to get quant parameter named `zero_point` for linear algorithm.
  ///
  /// \param[in] zero_point Define the value of quant parameter.
  /// \param[in] channel_index Define the index indicates which channel the quant parameter belong to. Default is 0
  /// indicating first channel.
  void set_zero_point(int64_t zero_point, size_t channel_index = 0);

  /// \brief Method to get quant parameter named `zero_point` for linear algorithm.
  ///
  /// \param[in] channel_index Define the index indicates which channel the quant parameter belong to. Default is 0
  /// indicating first channel.
  ///
  /// \return a int64_t as zero_point.
  int64_t get_zero_point(size_t channel_index = 0) const;
};

using FakeQuantParamPtr = std::shared_ptr<FakeQuantParam>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_FAKE_QUANT_PARAM_H_
