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
#include "frontend/operator/bprop/bprop_irbuilder.h"
#include "frontend/operator/bprop/grad/common_utils.h"
#include "include/common/utils/utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::expander::bprop {
NodePtrList Dropout2DBpropExpander(const BpropIRBuilder *ib) {
  auto keep_prob = GetValue<float>(ib->GetAttr("keep_prob"));
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto mask = ib->TupleGetItem(out, 1);
  auto dy = ib->TupleGetItem(dout, 0);
  mask = ib->Cast(mask, kFloat32);
  if (keep_prob != 0) {
    dy = ib->Mul(dy, ib->Tensor((1.0 / keep_prob), ib->GetDtype(dy)));
  }
  dy = ib->Mul(mask, dy);
  dy = ib->Cast(dy, ib->GetDtype(x));
  return {dy};
}

NodePtrList GeLUBpropExpander(const BpropIRBuilder *ib) {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("GeLUGrad", {dout, x, out});
  return {dx};
}

NodePtrList FastGeLUBpropExpander(const BpropIRBuilder *ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("FastGeLUGrad", {dout, x});
  return {dx};
}

NodePtrList Conv2DTransposeBpropExpander(const BpropIRBuilder *ib) {
  auto x = ib->GetInput(kIndex0);
  auto w = ib->GetInput(kIndex1);
  auto f_sizes = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto w_shape = ib->GetShape(w);
  auto dx = ib->Emit(kConv2DOpName, {dout, w},
                     {{"pad_mode", ib->GetAttr("pad_mode")},
                      {"pad", ib->GetAttr("pad")},
                      {"dilation", ib->GetAttr("dilation")},
                      {"stride", ib->GetAttr("stride")},
                      {"group", ib->GetAttr("group")},
                      {"groups", ib->GetAttr("group")},
                      {"format", ib->GetAttr("format")},
                      {"data_format", ib->GetAttr("format")},
                      {"out_channel", ib->GetAttr("out_channel")},
                      {"kernel_size", ib->GetAttr("kernel_size")},
                      {"mode", MakeValue(1)}});
  auto dw = ib->Emit(kConv2DBackpropFilterOpName, {x, dout, ib->Value(w_shape)},
                     {{"mode", ib->GetAttr("mode")},
                      {"dilation", ib->GetAttr("dilation")},
                      {"stride", ib->GetAttr("stride")},
                      {"group", ib->GetAttr("group")},
                      {"groups", ib->GetAttr("group")},
                      {"format", ib->GetAttr("format")},
                      {"data_format", ib->GetAttr("format")},
                      {"out_channel", ib->GetAttr("out_channel")},
                      {"kernel_size", ib->GetAttr("kernel_size")},
                      {"pad_mode", ib->GetAttr("pad_mode")},
                      {"pad", ib->GetAttr("pad")},
                      {"pad_list", ib->GetAttr("pad_list")}});
  return {dx, dw, ib->ZerosLike(f_sizes)};
}

NodePtrList CommonMaxMinGradBprop(const BpropIRBuilder *ib) {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  auto btmp = ib->TupleGetItem(out, 0);
  auto btmp_type = ib->GetDtype(btmp);
  auto out0 = ib->Cast(ib->NotEqual(btmp, ib->Tensor(0, btmp_type)), btmp_type);
  auto btmp1 = ib->TupleGetItem(out, 1);
  auto btmp1_type = ib->GetDtype(btmp1);
  auto out1 = ib->Cast(ib->NotEqual(btmp1, ib->Tensor(0, btmp1_type)), btmp1_type);
  auto dz = ib->Add(ib->Mul(out0, ib->TupleGetItem(dout, 0)), ib->Mul(out1, ib->TupleGetItem(dout, 1)));
  return {ib->ZerosLike(x), ib->ZerosLike(y), dz};
}

REG_BPROP_BUILDERS_BEGIN(GradNnOps)
REG_BPROP_BUILDER("Conv2D").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto w = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto x_shape = ib->GetShape(x);
  auto w_shape = ib->GetShape(w);
  auto format = GetValue<std::string>(ib->GetAttr("format"));
  auto dilation = GetValue<ShapeVector>(ib->GetAttr("dilation"));
  auto stride = GetValue<ShapeVector>(ib->GetAttr("stride"));
  auto dx = ib->Emit(kConv2DBackpropInputOpName, {dout, w, ib->Value<ShapeVector>(x_shape)},
                     {{"mode", ib->GetAttr("mode")},
                      {"dilation", MakeValue(format == "NHWC" ? ConvToNHWC(dilation) : dilation)},
                      {"stride", MakeValue(format == "NHWC" ? ConvToNHWC(stride) : stride)},
                      {"group", ib->GetAttr("group")},
                      {"groups", ib->GetAttr("group")},
                      {"format", ib->GetAttr("format")},
                      {"data_format", ib->GetAttr("format")},
                      {"out_channel", ib->GetAttr("out_channel")},
                      {"kernel_size", ib->GetAttr("kernel_size")},
                      {"pad_mode", ib->GetAttr("pad_mode")},
                      {"pad", ib->GetAttr("pad")},
                      {"pad_list", ib->GetAttr("pad_list")}});
  auto dw = ib->Emit("Conv2DBackpropFilter", {dout, x, ib->Value<ShapeVector>(w_shape)},
                     {{"mode", ib->GetAttr("mode")},
                      {"dilation", MakeValue(dilation)},
                      {"stride", MakeValue(stride)},
                      {"group", ib->GetAttr("group")},
                      {"groups", ib->GetAttr("group")},
                      {"format", ib->GetAttr("format")},
                      {"data_format", ib->GetAttr("format")},
                      {"out_channel", ib->GetAttr("out_channel")},
                      {"kernel_size", ib->GetAttr("kernel_size")},
                      {"pad_mode", ib->GetAttr("pad_mode")},
                      {"pad", ib->GetAttr("pad")},
                      {"pad_list", ib->GetAttr("pad_list")}});
  return {dx, dw};
});
REG_BPROP_BUILDER("MaxPool").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto format = GetValue<std::string>(ib->GetAttr("format"));
  auto kernel_size = GetValue<ShapeVector>(ib->GetAttr("kernel_size"));
  auto strides = GetValue<ShapeVector>(ib->GetAttr("strides"));
  if (format == "NHWC") {
    kernel_size = PoolToNHWC(kernel_size);
    strides = PoolToNHWC(strides);
  }
  auto dx = ib->Emit(kMaxPoolGradOpName, {x, out, dout},
                     {{"kernel_size", MakeValue(kernel_size)},
                      {"strides", MakeValue(strides)},
                      {"pad_mode", ib->GetAttr("pad_mode")},
                      {"data_format", ib->GetAttr("format")},
                      {"format", ib->GetAttr("format")}});
  return {dx};
});

REG_BPROP_BUILDER("BiasAdd").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto w = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto format = GetValue<std::string>(ib->GetAttr("data_format"));
  if (format == "NCDHW") {
    format = "NCHW";
  }
  return {dout,
          ib->Emit(kBiasAddGradOpName, {dout}, {{"format", MakeValue(format)}, {"data_format", MakeValue(format)}})};
});

REG_BPROP_BUILDER("ReLU").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit(kReluGradOpName, {dout, out});
  return {dx};
});

REG_BPROP_BUILDER("TopK").SetBody([](const BpropIRBuilder *builder) -> NodePtrList {
  auto input_x = builder->GetInput(kIndex0);
  auto out = builder->GetInput(kIndex2);
  auto dout = builder->GetInput(kIndex3);

  auto indices = builder->TupleGetItem(out, kIndex1);
  auto dout0 = builder->TupleGetItem(dout, kIndex0);

  auto in_shape = builder->GetShape(input_x);
  auto in_lastdim = in_shape.back();

  auto ind_shape = builder->GetShape(indices);
  auto ind_lastdim = ind_shape.back();

  auto ind_2d = builder->Reshape(indices, {-1, ind_lastdim});
  auto outerdim = builder->GetShape(ind_2d)[0];  // k

  // [0, outerdim, 2*outerdim, ..., (k-1)*outerdim]
  auto indices_dtype = builder->GetDtype(indices);
  std::vector<int64_t> range_flatten_index_vec(LongToSize(outerdim));
  for (int64_t i = 0; i < outerdim; i++) {
    range_flatten_index_vec[i] = i * in_lastdim;
  }
  auto range_flatten_index = builder->Tensor(range_flatten_index_vec, indices_dtype);
  auto ind = builder->Reshape(ind_2d + builder->Reshape(range_flatten_index, {-1, 1}), {-1, 1});
  auto in_shape_1d = ShapeVector(1, std::accumulate(in_shape.begin(), in_shape.end(), 1, std::multiplies<int64_t>()));
  auto out_grad = builder->Emit("ScatterNd", {ind, builder->Reshape(dout0, {-1}), builder->Value(in_shape_1d)});
  out_grad = builder->Reshape(out_grad, in_shape);

  auto grad_k = builder->ZerosLike(builder->GetInput(kIndex1));
  return {out_grad, grad_k};
});

REG_BPROP_BUILDER("PReLU").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto w = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto res = ib->Emit("PReLUGrad", {dout, x, w});
  auto dx = ib->TupleGetItem(res, kIndex0);
  auto dw = ib->TupleGetItem(res, kIndex1);
  return {dx, dw};
});

REG_BPROP_BUILDER("SigmoidCrossEntropyWithLogits").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("SigmoidCrossEntropyWithLogitsGrad", {x, y, dout});
  return {dx, ib->ZerosLike(y)};
});

REG_BPROP_BUILDER("Pad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto paddings = ib->GetAttr<std::vector<std::vector<int64_t>>>("paddings");
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  std::vector<int64_t> begin;
  for (const auto &item : paddings) {
    begin.push_back(item.at(0));
  }
  auto shp = ib->GetShape(x);
  auto dx = ib->Emit("Slice", {dout, ib->EmitValue(MakeValue(begin)), ib->EmitValue(MakeValue(shp))});
  return {dx};
});

REG_BPROP_BUILDER("ROIAlign").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto inputs = ib->GetInput(kIndex0);
  auto rois = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto inputs_shape = ib->GetShape(inputs);
  auto dx = ib->Emit("ROIAlignGrad", {dout, rois, ib->EmitValue(MakeValue(inputs_shape))},
                     {{"pooled_height", ib->GetAttr("pooled_height")},
                      {"pooled_width", ib->GetAttr("pooled_width")},
                      {"xdiff_shape", MakeValue(inputs_shape)},
                      {"spatial_scale", ib->GetAttr("spatial_scale")},
                      {"sample_num", ib->GetAttr("sample_num")}});
  return {dx, ib->ZerosLike(rois)};
});

REG_BPROP_BUILDER("LRN").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("LRNGrad", {dout, x, out},
                     {{"depth_radius", ib->GetAttr("depth_radius")},
                      {"bias", ib->GetAttr("bias")},
                      {"alpha", ib->GetAttr("alpha")},
                      {"beta", ib->GetAttr("beta")}});
  return {dx};
});

REG_BPROP_BUILDER("Dropout").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto mask = ib->TupleGetItem(out, 1);
  auto dy = ib->TupleGetItem(dout, 0);
  auto dx = ib->Emit(kDropoutGradOpName, {dy, mask}, {{"keep_prob", ib->GetAttr("keep_prob")}});
  return {dx};
});

REG_BPROP_BUILDER("BinaryCrossEntropy").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto weight = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = ib->Emit("BinaryCrossEntropyGrad", {x, y, dout, weight}, {{"reduction", ib->GetAttr("reduction")}});
  return {dx, ib->ZerosLike(y), ib->ZerosLike(weight)};
});

REG_BPROP_BUILDER("DropoutGrad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto mask = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dy = dout;
  auto dx = ib->Emit(kDropoutGradOpName, {dy, mask}, {{"keep_prob", ib->GetAttr("keep_prob")}});
  return {dx, ib->ZerosLike(mask)};
});

REG_BPROP_BUILDER("DeformableOffsets").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto offsets = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto out_grad = ib->Emit("DeformableOffsetsGrad", {dout, x, offsets},
                           {{"strides", ib->GetAttr("strides")},
                            {"pads", ib->GetAttr("pads")},
                            {"ksize", ib->GetAttr("ksize")},
                            {"dilations", ib->GetAttr("dilations")},
                            {"format", ib->GetAttr("format")},
                            {"data_format", ib->GetAttr("format")},
                            {"deformable_groups", ib->GetAttr("deformable_groups")},
                            {"modulated", ib->GetAttr("modulated")}});
  return {ib->TupleGetItem(out_grad, 0), ib->TupleGetItem(out_grad, 1)};
});

REG_BPROP_BUILDER("LSTM").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto input_size = ib->GetAttr("input_size");
  auto hidden_size = ib->GetAttr("hidden_size");
  auto num_layers = ib->GetAttr("num_layers");
  auto has_bias = ib->GetAttr("has_bias");
  auto bidirectional = ib->GetAttr("bidirectional");
  auto dropout = ib->GetAttr("dropout");
  auto x = ib->GetInput(kIndex0);
  auto hx = ib->GetInput(kIndex1);
  auto cx = ib->GetInput(kIndex2);
  auto w = ib->GetInput(kIndex3);
  auto out = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex5);
  auto target = ib->GetTargetFromContext();
  if (target == "CPU") {
    if (out != nullptr || dout != nullptr) {
      MS_LOG(INFO) << "Skip bprop expander for 'LSTM' on CPU because LSTMGrad does not have C++ infer";
      return {};
    }
    auto y = ib->TupleGetItem(out, kIndex0);
    auto hy = ib->TupleGetItem(out, kIndex1);
    auto cy = ib->TupleGetItem(out, kIndex2);
    auto reserve = ib->TupleGetItem(out, kIndex3);
    auto dy = ib->TupleGetItem(dout, kIndex0);
    auto dhy = ib->TupleGetItem(dout, kIndex1);
    auto dcy = ib->TupleGetItem(dout, kIndex2);
    auto res = ib->Emit("LSTMGrad", {x, hx, cx, w, y, hy, cy, dy, dhy, dcy, reserve},
                        {{"input_size", input_size},
                         {"hidden_size", hidden_size},
                         {"num_layers", num_layers},
                         {"has_bias", has_bias},
                         {"bidirectional", bidirectional},
                         {"dropout", dropout}});
    auto dx = ib->TupleGetItem(res, kIndex0);
    auto dhx = ib->TupleGetItem(res, kIndex1);
    auto dcx = ib->TupleGetItem(res, kIndex2);
    auto dw = ib->TupleGetItem(res, kIndex3);
    return {dx, dhx, dcx, dw};
  }
  auto y = ib->TupleGetItem(out, kIndex0);
  auto reserve = ib->TupleGetItem(out, kIndex3);
  auto state = ib->TupleGetItem(out, kIndex4);
  auto dy = ib->TupleGetItem(dout, kIndex0);
  auto dhy = ib->TupleGetItem(dout, kIndex1);
  auto dcy = ib->TupleGetItem(dout, kIndex2);
  auto res1 = ib->Emit("LSTMGradData", {y, dy, dhy, dcy, w, hx, cx, reserve, state},
                       {{"input_size", input_size},
                        {"hidden_size", hidden_size},
                        {"num_layers", num_layers},
                        {"has_bias", has_bias},
                        {"bidirectional", bidirectional},
                        {"dropout", dropout}});
  auto dx = ib->TupleGetItem(res1, kIndex0);
  auto dhx = ib->TupleGetItem(res1, kIndex1);
  auto dcx = ib->TupleGetItem(res1, kIndex2);
  auto dw = ib->Emit("LSTMGradWeight", {ib->Emit("Depend", {x, dx}), hx, y, reserve, state},
                     {{"input_size", input_size},
                      {"hidden_size", hidden_size},
                      {"num_layers", num_layers},
                      {"has_bias", has_bias},
                      {"bidirectional", bidirectional},
                      {"dropout", dropout}});
  return {dx, dhx, dcx, dw};
});

REG_BPROP_BUILDER("CudnnGRU").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto input_size = ib->GetAttr("input_size");
  auto hidden_size = ib->GetAttr("hidden_size");
  auto num_layers = ib->GetAttr("num_layers");
  auto has_bias = ib->GetAttr("has_bias");
  auto bidirectional = ib->GetAttr("bidirectional");
  auto dropout = ib->GetAttr("dropout");
  auto x = ib->GetInput(kIndex0);
  auto hx = ib->GetInput(kIndex1);
  auto w = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  auto y = ib->TupleGetItem(out, kIndex0);
  auto reserve = ib->TupleGetItem(out, kIndex2);
  auto state = ib->TupleGetItem(out, kIndex3);
  auto dy = ib->TupleGetItem(dout, kIndex0);
  auto dhy = ib->TupleGetItem(dout, kIndex1);
  auto res1 = ib->Emit("GruGradData", {y, dy, dhy, w, hx, reserve, state},
                       {{"input_size", input_size},
                        {"hidden_size", hidden_size},
                        {"num_layers", num_layers},
                        {"has_bias", has_bias},
                        {"bidirectional", bidirectional},
                        {"dropout", dropout}});
  auto dx = ib->TupleGetItem(res1, kIndex0);
  auto dhx = ib->TupleGetItem(res1, kIndex1);
  auto dw = ib->Emit("GruGradWeight", {ib->Emit("Depend", {x, dx}), hx, y, reserve, state},
                     {{"input_size", input_size},
                      {"hidden_size", hidden_size},
                      {"num_layers", num_layers},
                      {"has_bias", has_bias},
                      {"bidirectional", bidirectional},
                      {"dropout", dropout}});
  return {dx, dhx, dw};
});

REG_BPROP_BUILDER("MirrorPad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto paddings = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("MirrorPadGrad", {dout, paddings}, {{kAttrMode, ib->GetAttr(kAttrMode)}});
  return {dx, ib->ZerosLike(paddings)};
});

REG_BPROP_BUILDER("LayerNorm").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto gamma = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  auto result = ib->Emit(
    "LayerNormGrad", {x, ib->TupleGetItem(dout, 0), ib->TupleGetItem(out, 2), ib->TupleGetItem(out, 1), gamma},
    {{"begin_norm_axis", ib->GetAttr("begin_norm_axis")}, {"begin_params_axis", ib->GetAttr("begin_params_axis")}});
  auto d_x = ib->TupleGetItem(result, 0);
  auto d_gamma = ib->TupleGetItem(result, 1);
  auto d_beta = ib->TupleGetItem(result, 2);
  return {d_x, d_gamma, d_beta};
});

REG_BPROP_BUILDER("LayerNormGrad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dy = ib->GetInput(kIndex1);
  auto variance = ib->GetInput(kIndex2);
  auto mean = ib->GetInput(kIndex3);
  auto gamma = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex6);
  auto result = ib->Emit(
    "LayerNormGradGrad",
    {x, dy, variance, mean, gamma, ib->TupleGetItem(dout, 0), ib->TupleGetItem(dout, 1), ib->TupleGetItem(dout, 2)},
    {{"begin_norm_axis", ib->GetAttr("begin_norm_axis")}, {"begin_params_axis", ib->GetAttr("begin_params_axis")}});
  auto d_x = ib->TupleGetItem(result, 0);
  auto d_dy = ib->TupleGetItem(result, 1);
  auto d_gamma = ib->TupleGetItem(result, 2);
  return {d_x, d_dy, ib->ZerosLike(variance), ib->ZerosLike(mean), d_gamma};
});

REG_BPROP_BUILDER("L2Normalize").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx =
    ib->Emit("L2NormalizeGrad", {x, out, dout}, {{"axis", ib->GetAttr("axis")}, {"epsilon", ib->GetAttr("epsilon")}});
  return {dx};
});

REG_BPROP_BUILDER("SoftmaxCrossEntropyWithLogits").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto labels = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto grad = ib->TupleGetItem(out, 1);
  grad = ib->Mul(grad, (ib->ExpandDims(ib->TupleGetItem(dout, 0), -1)));
  return {grad, ib->ZerosLike(labels)};
});

REG_BPROP_BUILDER("NLLLoss").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto target = ib->GetInput(kIndex1);
  auto weight = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  auto total_weight = ib->TupleGetItem(out, 1);
  auto dout_x = ib->TupleGetItem(dout, 0);
  auto dx =
    ib->Emit("NLLLossGrad", {x, dout_x, target, weight, total_weight}, {{"reduction", ib->GetAttr("reduction")}});
  return {dx, ib->ZerosLike(target), ib->ZerosLike(weight)};
});

REG_BPROP_BUILDER("ResizeBilinear").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit(
    "ResizeBilinearGrad", {dout, x},
    {{"align_corners", ib->GetAttr("align_corners")}, {"half_pixel_centers", ib->GetAttr("half_pixel_centers")}});
  return {dx};
});

REG_BPROP_BUILDER("OneHot").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto indices = ib->GetInput(kIndex0);
  auto depth = ib->GetInput(kIndex1);
  auto on_value = ib->GetInput(kIndex2);
  auto off_value = ib->GetInput(kIndex3);
  return {ib->ZerosLike(indices), ib->ZerosLike(ib->Tensor(0, ib->GetDtype(depth))), ib->ZerosLike(on_value),
          ib->ZerosLike(off_value)};
});

REG_BPROP_BUILDER("SmoothL1Loss").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto prediction = ib->GetInput(kIndex0);
  auto target = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("SmoothL1LossGrad", {prediction, target, dout},
                     {{"beta", ib->GetAttr("beta")}, {"reduction", ib->GetAttr("reduction")}});
  auto dy = ib->Emit("SmoothL1LossGrad", {target, prediction, dout},
                     {{"beta", ib->GetAttr("beta")}, {"reduction", ib->GetAttr("reduction")}});
  return {dx, dy};
});

REG_BPROP_BUILDER("L2Loss").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Mul(x, dout);
  return {dx};
});

REG_BPROP_BUILDER("RNNTLoss").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto labels = ib->GetInput(kIndex1);
  auto act_lens = ib->GetInput(kIndex2);
  auto label_lens = ib->GetInput(kIndex3);
  auto out = ib->GetInput(kIndex4);
  auto grad = ib->TupleGetItem(out, 1);
  return {grad, ib->ZerosLike(labels), ib->ZerosLike(act_lens), ib->ZerosLike(label_lens)};
});

REG_BPROP_BUILDER("Conv3D").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto w = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto x_shape = ib->GetShape(x);
  auto w_shape = ib->GetShape(w);
  auto dx = ib->Emit("Conv3DBackpropInput", {w, dout, ib->Value<ShapeVector>(x_shape)},
                     {{"pad_mode", ib->GetAttr("pad_mode")},
                      {"pad", ib->GetAttr("pad")},
                      {"strides", ib->GetAttr("strides")},
                      {"dilations", ib->GetAttr("dilations")},
                      {"stride", ib->GetAttr("strides")},
                      {"dilation", ib->GetAttr("dilations")},
                      {"group", ib->GetAttr("groups")},
                      {"groups", ib->GetAttr("groups")},
                      {"format", ib->GetAttr("format")},
                      {"out_channel", ib->GetAttr("out_channel")},
                      {"kernel_size", ib->GetAttr("kernel_size")},
                      {"input_size", MakeValue(x_shape)},
                      {"mode", ib->GetAttr("mode")}});
  auto dw = ib->Emit("Conv3DBackpropFilter", {x, dout, ib->Value<ShapeVector>(w_shape)},
                     {{"pad_mode", ib->GetAttr("pad_mode")},
                      {"pad", ib->GetAttr("pad")},
                      {"strides", ib->GetAttr("strides")},
                      {"dilations", ib->GetAttr("dilations")},
                      {"stride", ib->GetAttr("strides")},
                      {"dilation", ib->GetAttr("dilations")},
                      {"group", ib->GetAttr("groups")},
                      {"groups", ib->GetAttr("groups")},
                      {"format", ib->GetAttr("format")},
                      {"out_channel", ib->GetAttr("out_channel")},
                      {"kernel_size", ib->GetAttr("kernel_size")},
                      {"filter_size", MakeValue(w_shape)},
                      {"mode", ib->GetAttr("mode")}});
  return {dx, dw};
});

REG_BPROP_BUILDER("Conv3DTranspose").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto strides = GetValue<std::vector<int64_t>>(ib->GetAttr("strides"));
  auto dilations = GetValue<std::vector<int64_t>>(ib->GetAttr("dilations"));
  std::vector<int64_t> stride = {strides.at(kIndex2), strides.at(kIndex3), strides.at(kIndex4)};
  std::vector<int64_t> dilation = {dilations.at(kIndex2), dilations.at(kIndex3), dilations.at(kIndex4)};
  auto x = ib->GetInput(kIndex0);
  auto w = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto w_shape = ib->GetShape(w);
  auto dx = ib->Emit("Conv3D", {dout, w},
                     {{"out_channel", ib->GetAttr("in_channel")},
                      {"kernel_size", ib->GetAttr("kernel_size")},
                      {"mode", ib->GetAttr("mode")},
                      {"pad_mode", MakeValue("pad")},
                      {"pad", ib->GetAttr("pad_list")},
                      {"strides", ib->GetAttr("strides")},
                      {"dilations", ib->GetAttr("dilations")},
                      {"stride", MakeValue(stride)},
                      {"dilation", MakeValue(dilation)},
                      {"group", ib->GetAttr("groups")},
                      {"groups", ib->GetAttr("groups")},
                      {"offset_x", MakeValue<int64_t>(0)},
                      {"format", ib->GetAttr("format")},
                      {"data_format", ib->GetAttr("format")}});
  auto dw = ib->Emit("Conv3DBackpropFilter", {dout, x, ib->Value<ShapeVector>(w_shape)},
                     {{"out_channel", ib->GetAttr("in_channel")},
                      {"kernel_size", ib->GetAttr("kernel_size")},
                      {"filter_size", MakeValue(w_shape)},
                      {"mode", ib->GetAttr("mode")},
                      {"pad_mode", MakeValue("pad")},
                      {"pad", ib->GetAttr("pad_list")},
                      {"strides", ib->GetAttr("strides")},
                      {"dilations", ib->GetAttr("dilations")},
                      {"stride", ib->GetAttr("strides")},
                      {"dilation", ib->GetAttr("dilations")},
                      {"group", ib->GetAttr("groups")},
                      {"groups", ib->GetAttr("groups")},
                      {"format", ib->GetAttr("format")},
                      {"data_format", ib->GetAttr("format")}});
  return {dx, dw};
});

REG_BPROP_BUILDER("MaxPoolWithArgmax").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("MaxPoolGradWithArgmax", {x, ib->TupleGetItem(dout, 0), ib->TupleGetItem(out, 1)},
                     {{"kernel_size", ib->GetAttr("kernel_size")},
                      {"strides", ib->GetAttr("strides")},
                      {"pad_mode", ib->GetAttr("pad_mode")}});
  return {dx};
});

REG_BPROP_BUILDER("MaxPoolGradGrad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x1 = ib->GetInput(kIndex0);
  auto x2 = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  auto dx1 = ib->ZerosLike(x1);
  auto dx2 = ib->ZerosLike(x2);
  auto dgrad = ib->Emit("MaxPoolGrad", {x1, x2, dout},
                        {{"kernel_size", ib->GetAttr("kernel_size")},
                         {"strides", ib->GetAttr("strides")},
                         {"pad_mode", ib->GetAttr("pad_mode")},
                         {"data_format", MakeValue("NCHW")},
                         {"format", MakeValue("NCHW")}});
  return {dx1, dx2, dgrad};
});

REG_BPROP_BUILDER("MaxPoolGrad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto device_target = ib->GetTargetFromContext();
  auto is_ascend = device_target == "Ascend";
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> strides;
  if (device_target == "CPU") {
    MS_LOG(EXCEPTION) << "MaxPoolGradGrad does not support on CPU!";
  }
  if (device_target == "GPU") {
    if ((GetValue<std::string>(ib->GetAttr("format"))) != "NCHW") {
      MS_LOG(EXCEPTION) << "MaxPoolGradGrad does not support NHWC!";
    }
    kernel_size = GetValue<std::vector<int64_t>>(ib->GetAttr("kernel_size"));
    if (kernel_size.size() == 4) {
      kernel_size = {1, kernel_size[2], kernel_size[3], 1};
    }
    strides = GetValue<std::vector<int64_t>>(ib->GetAttr("strides"));
    if (strides.size() == 4) {
      strides = {1, strides[2], strides[3], 1};
    }
  }
  auto x1 = ib->GetInput(kIndex0);
  auto x2 = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  auto dx1 = ib->ZerosLike(x1);
  auto dx2 = ib->ZerosLike(x2);
  NodePtr dgrad = nullptr;
  if (is_ascend) {
    dgrad = ib->Emit("MaxPoolGradGrad", {x1, x2, dout},
                     {{"kernel_size", ib->GetAttr("kernel_size")},
                      {"strides", ib->GetAttr("strides")},
                      {"data_format", MakeValue("NCHW")},
                      {"format", MakeValue("NCHW")},
                      {"pad_mode", ib->GetAttr("pad_mode")}});
  } else {
    auto x2_shape = ib->GetShape(x2);
    auto b = x2_shape.at(0);
    auto c = x2_shape.at(1);
    auto h = x2_shape.at(2);
    auto w = x2_shape.at(3);
    auto tmp = ib->Emit("MaxPoolWithArgmax", {x1},
                        {{"kernel_size", MakeValue(kernel_size)},
                         {"strides", MakeValue(strides)},
                         {"pad_mode", ib->GetAttr("pad_mode")},
                         {"data_format", MakeValue("NCHW")},
                         {"format", MakeValue("NCHW")}});
    auto ind = ib->TupleGetItem(tmp, 1);
    auto batch = ib->Tensor(Range(b), TypeIdToType(TypeId::kNumberTypeInt32));
    batch = ib->Tile(ib->Reshape(batch, {-1, 1}), {1, (c * h) * w});
    auto gather_ind =
      ib->Emit("Stack", {ib->MakeTuple({batch, ib->Reshape(ind, {b, -1})})}, {{"axis", MakeValue<int64_t>(-1)}});
    dgrad = ib->Reshape(ib->Emit("GatherNd", {ib->Reshape(dout, {b, -1}), gather_ind}), {b, c, h, w});
  }
  return {dx1, dx2, dgrad};
});

REG_BPROP_BUILDER("UpsampleNearest3D").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto input_x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("UpsampleNearest3DGrad", {dout},
                     {{"input_size", MakeValue(ib->GetShape(input_x))},
                      {"output_size", ib->GetAttr("output_size")},
                      {"scales", ib->GetAttr("scales")}});
  return {dx};
});

REG_BPROP_BUILDER("UpsampleTrilinear3D").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("UpsampleTrilinear3DGrad", {dout},
                     {{"input_size", MakeValue(ib->GetShape(x))},
                      {"output_size", ib->GetAttr("output_size")},
                      {"scales", ib->GetAttr("scales")},
                      {"align_corners", ib->GetAttr("align_corners")}});
  return {dx};
});

REG_BPROP_BUILDER("Dropout2D").SetBody(Dropout2DBpropExpander);
REG_BPROP_BUILDER("Dropout3D").SetBody(Dropout2DBpropExpander);

REG_BPROP_BUILDER("CTCLoss").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto labels_indices = ib->GetInput(kIndex1);
  auto labels_values = ib->GetInput(kIndex2);
  auto sequence_length = ib->GetInput(kIndex3);
  auto out = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex5);
  auto grad_loss = ib->TupleGetItem(out, 1);
  auto grad = ib->Mul(grad_loss, (ib->ExpandDims(ib->TupleGetItem(dout, 0), -1)));
  return {grad, ib->ZerosLike(labels_indices), ib->ZerosLike(labels_values), ib->ZerosLike(sequence_length)};
});

REG_BPROP_BUILDER("MaxPool3D").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("MaxPool3DGrad", {x, out, dout},
                     {{"kernel_size", ib->GetAttr("kernel_size")},
                      {"strides", ib->GetAttr("strides")},
                      {"pad_mode", ib->GetAttr("pad_mode")},
                      {"pad_list", ib->GetAttr("pad_list")},
                      {"format", ib->GetAttr("format")}});
  return {dx};
});

REG_BPROP_BUILDER("MaxPool3DGrad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  auto dgrad = ib->Emit("MaxPool3DGradGrad", {x, y, dout},
                        {{"kernel_size", ib->GetAttr("kernel_size")},
                         {"strides", ib->GetAttr("strides")},
                         {"pad_mode", ib->GetAttr("pad_mode")},
                         {"format", ib->GetAttr("format")}});
  return {ib->ZerosLike(x), ib->ZerosLike(y), dgrad};
});

REG_BPROP_BUILDER("MaxPool3DGradGrad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  ShapeVector pad_list(kDim6);
  auto dgrad = ib->Emit("MaxPool3DGrad", {x, y, dout},
                        {{"kernel_size", ib->GetAttr("kernel_size")},
                         {"strides", ib->GetAttr("strides")},
                         {"pad_mode", ib->GetAttr("pad_mode")},
                         {"format", ib->GetAttr("format")},
                         {"pad_list", MakeValue(pad_list)}});
  return {ib->ZerosLike(x), ib->ZerosLike(y), dgrad};
});

REG_BPROP_BUILDER("AvgPool").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto format = GetValue<std::string>(ib->GetAttr("format"));
  auto kernel_size = GetValue<ShapeVector>(ib->GetAttr("kernel_size"));
  auto strides = GetValue<ShapeVector>(ib->GetAttr("strides"));
  if (format == "NHWC") {
    kernel_size = PoolToNHWC(kernel_size);
    strides = PoolToNHWC(strides);
  }
  auto dx = ib->Emit("AvgPoolGrad", {x, out, dout},
                     {{"kernel_size", MakeValue(kernel_size)},
                      {"strides", MakeValue(strides)},
                      {"pad_mode", ib->GetAttr("pad_mode")},
                      {"data_format", ib->GetAttr("format")},
                      {"format", ib->GetAttr("format")}});
  return {dx};
});

REG_BPROP_BUILDER("AvgPool3D").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto x_shape = ib->GetShape(x);
  auto dx = ib->Emit("AvgPool3DGrad", {ib->Value<ShapeVector>(x_shape), dout},
                     {{"kernel_size", ib->GetAttr("kernel_size")},
                      {"origin_input_shape", MakeValue(x_shape)},
                      {"strides", ib->GetAttr("strides")},
                      {"pad_list", ib->GetAttr("pad_list")},
                      {"ceil_mode", ib->GetAttr("ceil_mode")},
                      {"count_include_pad", ib->GetAttr("count_include_pad")},
                      {"divisor_override", ib->GetAttr("divisor_override")},
                      {"format", ib->GetAttr("format")},
                      {"pad_mode", ib->GetAttr("pad_mode")}});
  return {dx};
});

REG_BPROP_BUILDER("Mish").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx1 = ib->Emit("Tanh", {ib->Emit("Softplus", {x})});
  auto dx2 = ib->Emit("SoftplusGrad", {ib->Emit("TanhGrad", {dx1, ib->Mul(x, dout)}), x});
  auto dx = ib->Add((ib->Mul(dx1, dout)), dx2);
  return {dx};
});

REG_BPROP_BUILDER("SeLU").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto scale = 1.0507009873554805;
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto tmp_grad = ib->Emit("EluGrad", {dout, out});
  auto dx = ib->Mul(tmp_grad, ib->Tensor(scale, ib->GetDtype(tmp_grad)));
  return {dx};
});

REG_BPROP_BUILDER("ReLU6").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("ReLU6Grad", {dout, x});
  return {dx};
});

REG_BPROP_BUILDER("ReLUV2").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto mask = ib->TupleGetItem(out, 1);
  auto dx = ib->Emit("ReluGradV2", {ib->TupleGetItem(dout, 0), mask});
  return {dx};
});

REG_BPROP_BUILDER("BiasAddGrad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto data_format = GetValue<std::string>(ib->GetAttr("format"));
  auto dy = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dy_shape = ib->GetShape(dy);
  auto dout_shape = ib->GetShape(dout);
  ShapeVector expanded_shape;
  ShapeVector tile_mults;
  ShapeVector one_vec{1};
  if (data_format == "NCHW") {
    // expanded_shape = np.concatenate([np.ones_like(shape[:1]), bias_shape, np.ones_like(shape[2:])], axis=0)
    expanded_shape = one_vec + dout_shape;
    expanded_shape = dy_shape.size() > 2 ? expanded_shape + ShapeVector(1, dy_shape.size() - 2) : expanded_shape;
    // tile_mults = np.concatenate([shape[:1], [1], shape[2:]], axis=0)
    ShapeVector tmp{dy_shape[0], 1};
    tile_mults = tmp;
    tile_mults = dy_shape.size() > 2 ? tile_mults + ShapeVector(dy_shape.begin() + 2, dy_shape.end()) : tile_mults;
  } else {
    // expanded_shape = np.concatenate([np.ones_like(shape[:-1]), bias_shape], axis=0)
    expanded_shape = ShapeVector(1, dy_shape.size() - 1) + dout_shape;
    // tile_mults = np.concatenate([shape[:-1], [1]], axis=0)
    tile_mults = ShapeVector(dy_shape.begin(), dy_shape.end() - 1) + one_vec;
  }
  auto expanded_grad = ib->Reshape(dout, expanded_shape);
  auto tiled_grad = ib->Tile(expanded_grad, tile_mults);
  return {tiled_grad};
});

REG_BPROP_BUILDER("ExtractImagePatches").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto ksizes_row = GetValue<std::vector<int64_t>>(ib->GetAttr("ksizes"))[2];
  auto ksizes_col = GetValue<std::vector<int64_t>>(ib->GetAttr("ksizes"))[3];
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto x_shape = ib->GetShape(x);
  auto x_batch = x_shape[0];
  auto x_depth = x_shape[1];
  auto x_row = x_shape[2];
  auto x_col = x_shape[3];
  auto x_indices_num = (x_row * x_col) + 1;
  auto x_idx = ib->Tensor(Range(1, x_indices_num), kFloat32);
  x_idx = ib->Reshape(x_idx, {1, 1, x_row, x_col});
  auto x_idx_patch = ib->Cast(ib->Emit("ExtractImagePatches", {x_idx},
                                       {{"ksizes", ib->GetAttr("ksizes")},
                                        {"strides", ib->GetAttr("strides")},
                                        {"rates", ib->GetAttr("rates")},
                                        {"padding", ib->GetAttr("padding")}}),
                              kInt32);
  x_idx_patch = ib->Transpose(x_idx_patch, {0, 2, 3, 1});
  auto out_shape = ib->GetShape(out);
  auto out_row = out_shape[2];
  auto out_col = out_shape[3];
  auto out_indices_num = ((out_row * out_col) * ksizes_row) * ksizes_col;
  auto out_idx = ib->Tensor(Range(out_indices_num), kInt32);
  out_idx = ib->Reshape(out_idx, {1, out_row, out_col, ksizes_row * ksizes_col});
  auto idx_tensor = ib->Emit("Concat", {ib->MakeTuple({ib->ExpandDims(x_idx_patch, -1), ib->ExpandDims(out_idx, -1)})},
                             {{"axis", MakeValue<int64_t>(-1)}});
  idx_tensor = ib->Reshape(idx_tensor, {-1, 2});
  std::vector<int64_t> sp_shape = {x_indices_num, out_indices_num};
  std::vector<int64_t> ones(out_indices_num, 1);
  auto sp_tensor =
    ib->Emit("ScatterNd", {idx_tensor, ib->Tensor(ones, ib->GetDtype(dout)), ib->Value<ShapeVector>(sp_shape)});
  sp_tensor = ib->Emit(
    "Slice", {sp_tensor, ib->Value<ShapeVector>({1, 0}), ib->Value<ShapeVector>({x_indices_num - 1, out_indices_num})});
  auto grad = ib->Transpose(dout, {0, 2, 3, 1});
  grad = ib->Reshape(grad, {x_batch, out_row, out_col, ksizes_row, ksizes_col, x_depth});
  grad = ib->Transpose(grad, {1, 2, 3, 4, 0, 5});
  grad = ib->Reshape(grad, {-1, x_batch * x_depth});
  auto jac = ib->MatMul(sp_tensor, grad, false, false);
  auto dx = ib->Reshape(jac, {x_row, x_col, x_batch, x_depth});
  dx = ib->Transpose(dx, {2, 3, 0, 1});
  return {dx};
});

REG_BPROP_BUILDER("HSwish").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("HSwishGrad", {dout, x});
  return {dx};
});

REG_BPROP_BUILDER("HSigmoid").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("HSigmoidGrad", {dout, x});
  return {dx};
});

REG_BPROP_BUILDER("Elu").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("EluGrad", {dout, out});
  return {dx};
});

REG_BPROP_BUILDER("Sigmoid").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("SigmoidGrad", {out, dout});
  return {dx};
});

REG_BPROP_BUILDER("SigmoidGrad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto y = ib->GetInput(kIndex0);
  auto grad = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dy = ib->Mul((ib->Mul(dout, grad)),
                    (ib->Sub(ib->Tensor(1, ib->GetDtype(grad)), (ib->Mul(ib->Tensor(2, ib->GetDtype(y)), y)))));
  auto dgrad = ib->Emit("SigmoidGrad", {y, dout});
  return {dy, dgrad};
});

REG_BPROP_BUILDER("LogSoftmax").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("LogSoftmaxGrad", {out, dout}, {{"axis", ib->GetAttr("axis")}});
  return {dx};
});

REG_BPROP_BUILDER("Softplus").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("SoftplusGrad", {dout, x});
  return {dx};
});

REG_BPROP_BUILDER("Softsign").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx =
    ib->Mul(dout, ib->Div(ib->Tensor(1, ib->GetDtype(x)),
                          ib->Emit("Square", {ib->Add(ib->Tensor(1, ib->GetDtype(x)), (ib->Emit("Abs", {x})))})));
  return {dx};
});

REG_BPROP_BUILDER("Tanh").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto x_dtype_id = ib->GetDtypeId(x);
  NodePtr dx;
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    dout = ib->Emit("Conj", {dout});
    dx = ib->Emit("TanhGrad", {out, dout});
    dx = ib->Emit("Conj", {dx});
  } else {
    dx = ib->Emit("TanhGrad", {out, dout});
  }
  return {dx};
});

REG_BPROP_BUILDER("TanhGrad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto y = ib->GetInput(kIndex0);
  auto grad = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dy = ib->Mul((ib->Mul((ib->Mul(dout, ib->Tensor(-2.0, ib->GetDtype(dout)))), grad)), y);
  auto dgrad = ib->Emit("TanhGrad", {y, dout});
  return {dy, dgrad};
});

REG_BPROP_BUILDER("GeLU").SetBody(GeLUBpropExpander);
REG_BPROP_BUILDER("Gelu").SetBody(GeLUBpropExpander);

REG_BPROP_BUILDER("FastGeLU").SetBody(FastGeLUBpropExpander);
REG_BPROP_BUILDER("FastGelu").SetBody(FastGeLUBpropExpander);

REG_BPROP_BUILDER("InstanceNorm").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto gamma = ib->GetInput(kIndex1);
  auto mean = ib->GetInput(kIndex3);
  auto variance = ib->GetInput(kIndex4);
  auto out = ib->GetInput(kIndex5);
  auto dout = ib->GetInput(kIndex6);
  auto saved_mean = ib->TupleGetItem(out, 1);
  auto saved_variance = ib->TupleGetItem(out, 2);
  out = ib->Emit("InstanceNormGrad", {ib->TupleGetItem(dout, 0), x, gamma, saved_mean, saved_variance},
                 {{"epsilon", ib->GetAttr("epsilon")}, {"momentum", ib->GetAttr("momentum")}});
  auto dx = ib->TupleGetItem(out, 0);
  auto dgamma = ib->TupleGetItem(out, 1);
  auto dbeta = ib->TupleGetItem(out, 2);
  return {dx, dgamma, dbeta, ib->ZerosLike(mean), ib->ZerosLike(variance)};
});

REG_BPROP_BUILDER("BatchNorm").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto is_training = GetValue<bool>(ib->GetAttr("is_training"));
  auto x = ib->GetInput(kIndex0);
  auto scale = ib->GetInput(kIndex1);
  auto mean = ib->GetInput(kIndex3);
  auto variance = ib->GetInput(kIndex4);
  auto out = ib->GetInput(kIndex5);
  auto dout = ib->GetInput(kIndex6);
  NodePtr saved_mean, saved_variance, reserve;
  if (is_training) {
    saved_mean = ib->TupleGetItem(out, 3);
    saved_variance = ib->TupleGetItem(out, 4);
    reserve = ib->TupleGetItem(out, 2);
  } else {
    saved_mean = mean;
    saved_variance = variance;
    reserve = ib->TupleGetItem(out, 2);
  }
  out = ib->Emit(
    "BatchNormGrad", {ib->TupleGetItem(dout, 0), x, scale, saved_mean, saved_variance, reserve},
    {{"is_training", MakeValue(is_training)}, {"epsilon", ib->GetAttr("epsilon")}, {"format", ib->GetAttr("format")}});
  auto dx = ib->TupleGetItem(out, 0);
  auto dscale = ib->TupleGetItem(out, 1);
  auto dbias = ib->TupleGetItem(out, 2);
  return {dx, dscale, dbias, ib->ZerosLike(mean), ib->ZerosLike(variance)};
});

REG_BPROP_BUILDER("BatchNormGrad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dy = ib->GetInput(kIndex0);
  auto x = ib->GetInput(kIndex1);
  auto scale = ib->GetInput(kIndex2);
  auto mean = ib->GetInput(kIndex3);
  auto variance = ib->GetInput(kIndex4);
  auto reserve = ib->GetInput(kIndex5);
  auto dout = ib->GetInput(kIndex7);
  auto tmp = ib->Emit(
    "BatchNormGradGrad",
    {x, dy, scale, mean, variance, ib->TupleGetItem(dout, 0), ib->TupleGetItem(dout, 1), ib->TupleGetItem(dout, 2)},
    {{"is_training", ib->GetAttr("is_training")},
     {"epsilon", ib->GetAttr("epsilon")},
     {"format", ib->GetAttr("format")}});
  auto dx = ib->TupleGetItem(tmp, 0);
  auto ddy = ib->TupleGetItem(tmp, 1);
  auto dscale = ib->TupleGetItem(tmp, 2);
  return {ddy, dx, dscale, ib->ZerosLike(mean), ib->ZerosLike(variance), ib->ZerosLike(reserve)};
});

REG_BPROP_BUILDER("Softmax").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto axis = GetValue<std::vector<int64_t>>(ib->GetAttr("axis"));
  auto one_axis = axis[0];
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto shp = ib->GetShape(x);
  auto reverse_axis = GetTransposeAxis(shp, one_axis);
  out = ib->Transpose(out, reverse_axis);
  dout = ib->Transpose(dout, reverse_axis);
  ShapeVector reduce_axis = {-1};
  auto dx = ib->Mul(out, ib->Sub(dout, ib->ReduceSum(ib->Mul(out, dout), reduce_axis, true)));
  dx = ib->Transpose(dx, reverse_axis);
  return {dx};
});

REG_BPROP_BUILDER("SparseSoftmaxCrossEntropyWithLogits").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto is_grad = ib->GetAttr<bool>(kAttrIsGrad);
  auto labels = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  if (is_grad) {
    return {ib->TensorGetItem(out, 0), ib->ZerosLike(labels)};
  }
  // is_grad is false
  auto logits = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex3);
  auto grad = ib->Emit(kSparseSoftmaxCrossEntropyWithLogitsOpName, {logits, labels}, {{kAttrIsGrad, MakeValue(true)}});
  grad = ib->Mul(grad, dout);
  return {grad, ib->ZerosLike(labels)};
});

REG_BPROP_BUILDER("DynamicRNN").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto w = ib->GetInput(kIndex1);
  auto b = ib->GetInput(kIndex2);
  auto init_h = ib->GetInput(kIndex4);
  auto init_c = ib->GetInput(kIndex5);
  auto out = ib->GetInput(kIndex6);
  auto dout = ib->GetInput(kIndex7);
  auto dy = ib->TupleGetItem(dout, kIndex0);
  auto dh = ib->TupleGetItem(dout, kIndex1);
  auto dc = ib->TupleGetItem(dout, kIndex2);
  dh = ib->TensorGetItem(dh, -1);
  dc = ib->TensorGetItem(dc, -1);
  auto y = ib->TupleGetItem(out, kIndex0);
  auto h = ib->TupleGetItem(out, kIndex1);
  auto c = ib->TupleGetItem(out, kIndex2);
  auto i = ib->TupleGetItem(out, kIndex3);
  auto j = ib->TupleGetItem(out, kIndex4);
  auto f = ib->TupleGetItem(out, kIndex5);
  auto o = ib->TupleGetItem(out, kIndex6);
  auto tanhct = ib->TupleGetItem(out, kIndex7);
  auto tmp = ib->Emit(
    "DynamicRNNGrad",
    {x, w, b, y, ib->TensorGetItem(init_h, 0), ib->TensorGetItem(init_c, 0), h, c, dy, dh, dc, i, j, f, o, tanhct},
    {{"cell_type", ib->GetAttr("cell_type")},
     {"direction", ib->GetAttr("direction")},
     {"cell_depth", ib->GetAttr("cell_depth")},
     {"use_peephole", ib->GetAttr("use_peephole")},
     {"keep_prob", ib->GetAttr("keep_prob")},
     {"cell_clip", ib->GetAttr("cell_clip")},
     {"num_proj", ib->GetAttr("num_proj")},
     {"time_major", ib->GetAttr("time_major")},
     {"forget_bias", ib->GetAttr("forget_bias")}});
  auto dw = ib->TupleGetItem(tmp, kIndex0);
  auto db = ib->TupleGetItem(tmp, kIndex1);
  auto dx = ib->TupleGetItem(tmp, kIndex2);
  auto dh_prev = ib->TupleGetItem(tmp, kIndex3);
  auto dc_prev = ib->TupleGetItem(tmp, kIndex4);
  dh_prev = ib->ExpandDims(dh_prev, 0);
  dc_prev = ib->ExpandDims(dc_prev, 0);
  constexpr int64_t zero = 0;
  return {dx, dw, db, ib->ZerosLike(ib->Tensor(zero)), dh_prev, dc_prev};
});

REG_BPROP_BUILDER("DynamicGRUV2").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto winput = ib->GetInput(kIndex1);
  auto whidden = ib->GetInput(kIndex2);
  auto init_h = ib->GetInput(kIndex6);
  auto out = ib->GetInput(kIndex7);
  auto dout = ib->GetInput(kIndex8);
  auto y = ib->TupleGetItem(out, kIndex0);
  auto out_h = ib->TupleGetItem(out, kIndex1);
  auto update = ib->TupleGetItem(out, kIndex2);
  auto reset = ib->TupleGetItem(out, kIndex3);
  auto new_t = ib->TupleGetItem(out, kIndex4);
  auto hidden_new = ib->TupleGetItem(out, kIndex5);
  auto dy = ib->TupleGetItem(dout, kIndex0);
  auto dout_h = ib->TupleGetItem(dout, kIndex1);
  auto tmp = ib->Emit("DynamicGRUV2Grad",
                      {x, winput, whidden, y, init_h, out_h, dy, ib->TensorGetItem(dout_h, -1), update, reset, new_t,
                       hidden_new, ib->EmitValue(kNone), ib->EmitValue(kNone)},
                      {{"direction", ib->GetAttr("direction")},
                       {"cell_depth", ib->GetAttr("cell_depth")},
                       {"keep_prob", ib->GetAttr("keep_prob")},
                       {"cell_clip", ib->GetAttr("cell_clip")},
                       {"num_proj", ib->GetAttr("num_proj")},
                       {"time_major", ib->GetAttr("time_major")},
                       {"gate_order", ib->GetAttr("gate_order")},
                       {"reset_after", ib->GetAttr("reset_after")}});
  auto dw_input = ib->TupleGetItem(tmp, kIndex0);
  auto dw_hidden = ib->TupleGetItem(tmp, kIndex1);
  auto db_input = ib->TupleGetItem(tmp, kIndex2);
  auto db_hidden = ib->TupleGetItem(tmp, kIndex3);
  auto dx = ib->TupleGetItem(tmp, kIndex4);
  auto dh_prev = ib->TupleGetItem(tmp, kIndex5);
  constexpr int64_t zero = 0;
  return {dx, dw_input, dw_hidden, db_input, db_hidden, ib->ZerosLike(ib->Tensor(zero)), dh_prev};
});

REG_BPROP_BUILDER("AdaptiveMaxPool2D").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto index = ib->TupleGetItem(out, 1);
  auto dy = ib->TupleGetItem(dout, 0);
  auto dx = ib->Emit("AdaptiveMaxPool2DGrad", {dy, x, index});
  return {dx};
});

REG_BPROP_BUILDER("Conv2DTranspose").SetBody(Conv2DTransposeBpropExpander);
REG_BPROP_BUILDER("Conv2DBackpropInput").SetBody(Conv2DTransposeBpropExpander);

REG_BPROP_BUILDER("Conv2DBackpropFilter").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto dy = ib->GetInput(kIndex0);
  auto x = ib->GetInput(kIndex1);
  auto filter_size = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto x_shape = ib->GetShape(x);
  auto dw_dx = ib->Emit(kConv2DBackpropInputOpName, {dy, dout, ib->Value(x_shape)},
                        {{"mode", ib->GetAttr("mode")},
                         {"dilation", ib->GetAttr("dilation")},
                         {"stride", ib->GetAttr("stride")},
                         {"group", ib->GetAttr("group")},
                         {"groups", ib->GetAttr("group")},
                         {"format", ib->GetAttr("format")},
                         {"data_format", ib->GetAttr("format")},
                         {"out_channel", ib->GetAttr("out_channel")},
                         {"kernel_size", ib->GetAttr("kernel_size")},
                         {"pad_mode", ib->GetAttr("pad_mode")},
                         {"pad", ib->GetAttr("pad")},
                         {"pad_list", ib->GetAttr("pad_list")}});
  auto dw_dy = ib->Emit(kConv2DOpName, {x, dout},
                        {{"pad_mode", ib->GetAttr("pad_mode")},
                         {"pad", ib->GetAttr("pad")},
                         {"dilation", ib->GetAttr("dilation")},
                         {"stride", ib->GetAttr("stride")},
                         {"group", ib->GetAttr("group")},
                         {"groups", ib->GetAttr("group")},
                         {"format", ib->GetAttr("format")},
                         {"data_format", ib->GetAttr("format")},
                         {"out_channel", ib->GetAttr("out_channel")},
                         {"kernel_size", ib->GetAttr("kernel_size")},
                         {"mode", MakeValue(1)}});
  return {dw_dy, dw_dx, ib->ZerosLike(filter_size)};
});

REG_BPROP_BUILDER("BCEWithLogitsLoss").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto reduction = GetValue<std::string>(ib->GetAttr("reduction"));
  auto predict = ib->GetInput(kIndex0);
  auto target = ib->GetInput(kIndex1);
  auto weight = ib->GetInput(kIndex2);
  auto pos_weight = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto sigmoid_input = ib->Emit("Sigmoid", {predict});

  auto t = ib->Mul(target, pos_weight);
  auto dx =
    ib->Mul(ib->Sub(ib->Mul(ib->Sub(ib->Add(t, ib->Tensor(1, ib->GetDtype(t))), target), sigmoid_input), t), dout);
  auto grad_target = ib->Mul(ib->Sub(ib->Log(ib->Sub(ib->Tensor(1, ib->GetDtype(sigmoid_input)), sigmoid_input)),
                                     ib->Mul(pos_weight, ib->Log(sigmoid_input))),
                             dout);

  dx = ib->Mul(dx, weight);
  grad_target = ib->Mul(grad_target, weight);

  if (reduction == "mean") {
    dx = ib->RealDiv(dx, ib->Tensor(ib->GetSize(dx), ib->GetDtype(dx)));
    grad_target = ib->RealDiv(grad_target, ib->Tensor(ib->GetSize(target), ib->GetDtype(grad_target)));
  }
  return {dx, grad_target, ib->ZerosLike(weight), ib->ZerosLike(pos_weight)};
});

REG_BPROP_BUILDER("KLDivLoss").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto reduction = GetValue<std::string>(ib->GetAttr("reduction"));
  auto x = ib->GetInput(kIndex0);
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  NodePtr dx;
  if (reduction == "mean") {
    dx = ib->Emit("KLDivLossGrad", {dout, x, y}, {{"reduction", MakeValue("sum")}});
    dx = ib->RealDiv(dx, ib->Tensor(ib->GetSize(x), ib->GetDtype(dx)));
  } else {
    dx = ib->Emit("KLDivLossGrad", {dout, x, y}, {{"reduction", MakeValue(reduction)}});
  }
  return {dx, ib->ZerosLike(y)};
});

REG_BPROP_BUILDER("HShrink").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto features = ib->GetInput(kIndex0);
  auto gradients = ib->GetInput(kIndex2);
  auto dx = ib->Emit("HShrinkGrad", {gradients, features}, {{"lambd", ib->GetAttr("lambd")}});
  return {dx};
});

REG_BPROP_BUILDER("SoftShrink").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto input_x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("SoftShrinkGrad", {dout, input_x}, {{"lambd", ib->GetAttr("lambd")}});
  return {dx};
});

REG_BPROP_BUILDER("SoftMarginLoss").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto predict = ib->GetInput(kIndex0);
  auto label = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("SoftMarginLossGrad", {predict, label, dout}, {{"reduction", ib->GetAttr("reduction")}});
  auto dy = ib->Emit("SoftMarginLossGrad", {label, predict, dout}, {{"reduction", ib->GetAttr("reduction")}});
  return {dx, dy};
});

REG_BPROP_BUILDER("MultilabelMarginLoss").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto target = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("MultilabelMarginLossGrad", {ib->TupleGetItem(dout, 0), x, target, ib->TupleGetItem(out, 1)},
                     {{"reduction", ib->GetAttr("reduction")}});
  return {dx, ib->ZerosLike(target)};
});

REG_BPROP_BUILDER("Dilation2D").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto _filter = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("Dilation2DBackpropInput", {x, _filter, dout},
                     {{"stride", ib->GetAttr("stride")},
                      {"dilation", ib->GetAttr("dilation")},
                      {"pad_mode", ib->GetAttr("pad_mode")},
                      {"format", ib->GetAttr("format")}});
  auto dfilter = ib->Emit("Dilation2DBackpropFilter", {x, _filter, dout},
                          {{"stride", ib->GetAttr("stride")},
                           {"dilation", ib->GetAttr("dilation")},
                           {"pad_mode", ib->GetAttr("pad_mode")},
                           {"format", ib->GetAttr("format")}});
  return {dx, dfilter};
});

REG_BPROP_BUILDER("CeLU").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto alpha = GetValue<float>(ib->GetAttr("alpha"));
  auto x = ib->GetInput(kIndex0);
  auto x_dtype = ib->GetDtype(x);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto greater = ib->GreaterEqual(x, ib->Tensor(0.0, x_dtype));
  greater = ib->Cast(greater, x_dtype);
  auto lesser = ib->Less(x, ib->Tensor(0.0, x_dtype));
  lesser = ib->Cast(lesser, x_dtype);
  auto dx = ib->Mul(
    dout,
    (ib->Add((ib->Mul(greater, ib->Tensor(1.0, x_dtype))),
             (ib->Mul(lesser, (ib->Add((ib->RealDiv(out, ib->Tensor(alpha, x_dtype))), ib->Tensor(1.0, x_dtype))))))));
  return {dx};
});

REG_BPROP_BUILDER("Pdist").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("PdistGrad", {dout, x, out}, {{"p", ib->GetAttr("p")}});
  return {dx};
});

REG_BPROP_BUILDER("MultiMarginLoss").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto target = ib->GetInput(kIndex1);
  auto weight = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx =
    ib->Emit("MultiMarginLossGrad", {dout, x, target, weight},
             {{"p", ib->GetAttr("p")}, {"margin", ib->GetAttr("margin")}, {"reduction", ib->GetAttr("reduction")}});
  return {dx, ib->ZerosLike(target), ib->ZerosLike(weight)};
});

REG_BPROP_BUILDER("DropoutGenMask").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto shape = ib->GetInput(kIndex0);
  auto keep_prob = ib->GetInput(kIndex1);
  return {ib->ZerosLike(shape), ib->ZerosLike(keep_prob)};
});

REG_BPROP_BUILDER("DropoutDoMask").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto y = ib->GetInput(kIndex1);
  auto keep_prob = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  return {ib->Emit("DropoutDoMask", {dout, y, keep_prob}), ib->ZerosLike(y), ib->ZerosLike(keep_prob)};
});

REG_BPROP_BUILDER("ReluGrad").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto y = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dgrad = ib->Emit("ReluGrad", {dout, y});
  return {dgrad, ib->ZerosLike(y)};
});

REG_BPROP_BUILDER("GridSampler3D").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto input_x = ib->GetInput(kIndex0);
  auto grid = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto tmp = ib->Emit("GridSampler3DGrad", {dout, input_x, grid},
                      {{"interpolation_mode", ib->GetAttr("interpolation_mode")},
                       {"padding_mode", ib->GetAttr("padding_mode")},
                       {"align_corners", ib->GetAttr("align_corners")}});
  auto dx = ib->TupleGetItem(tmp, 0);
  auto dgrid = ib->TupleGetItem(tmp, 1);
  return {dx, dgrid};
});

REG_BPROP_BUILDER("ReLUV3").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dgrad = ib->Emit("ReluGrad", {dout, out});
  return {dgrad};
});

REG_BPROP_BUILDER("GridSampler2D").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto input_x = ib->GetInput(kIndex0);
  auto grid = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto tmp = ib->Emit("GridSampler2DGrad", {dout, input_x, grid},
                      {{"interpolation_mode", ib->GetAttr("interpolation_mode")},
                       {"padding_mode", ib->GetAttr("padding_mode")},
                       {"align_corners", ib->GetAttr("align_corners")}});
  auto dx = ib->TupleGetItem(tmp, 0);
  auto dgrid = ib->TupleGetItem(tmp, 1);
  return {dx, dgrid};
});

REG_BPROP_BUILDER("ResizeLinear1D").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto input_x = ib->GetInput(kIndex0);
  auto size = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("ResizeLinear1DGrad", {dout, input_x},
                     {{"coordinate_transformation_mode", ib->GetAttr("coordinate_transformation_mode")}});
  return {dx, ib->ZerosLike(size)};
});

REG_BPROP_BUILDER("MaxPool3DWithArgmax").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("MaxPool3DGradWithArgmax", {x, ib->TupleGetItem(dout, 0), ib->TupleGetItem(out, 1)},
                     {{"ksize", ib->GetAttr("ksize")},
                      {"strides", ib->GetAttr("strides")},
                      {"pads", ib->GetAttr("pads")},
                      {"dilation", ib->GetAttr("dilation")},
                      {"ceil_mode", ib->GetAttr("ceil_mode")},
                      {"format", ib->GetAttr("format")}});
  return {dx};
});

REG_BPROP_BUILDER("MaxUnpool2D").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto argmax = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("MaxUnpool2DGrad", {x, dout, argmax},
                     {{"ksize", ib->GetAttr("ksize")},
                      {"strides", ib->GetAttr("strides")},
                      {"pads", ib->GetAttr("pads")},
                      {"output_shape", ib->GetAttr("output_shape")},
                      {"format", ib->GetAttr("format")}});
  auto dargmax = ib->ZerosLike(argmax);
  return {dx, dargmax};
});

REG_BPROP_BUILDER("MaxUnpool3D").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto argmax = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("MaxUnpool3DGrad", {x, dout, argmax},
                     {{"ksize", ib->GetAttr("ksize")},
                      {"strides", ib->GetAttr("strides")},
                      {"pads", ib->GetAttr("pads")},
                      {"output_shape", ib->GetAttr("output_shape")},
                      {"format", ib->GetAttr("format")}});
  auto dargmax = ib->ZerosLike(argmax);
  return {dx, dargmax};
});

REG_BPROP_BUILDER("NthElement").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto input_x = ib->GetInput(kIndex0);
  auto n = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto indicators = ib->Equal(ib->ExpandDims(out, -1), input_x, ib->GetDtype(input_x));
  dout = ib->ExpandDims(dout, -1);
  auto num_select = ib->ExpandDims(ib->ReduceSum(indicators, {-1}), -1);
  return {ib->Mul(ib->Div(indicators, num_select), dout), ib->ZerosLike(n)};
});

REG_BPROP_BUILDER("AdaptiveAvgPool3D").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto x_shape = ib->Tensor(ib->GetShape(x));
  auto dx = ib->Emit("AdaptiveAvgPool3DGrad", {dout, ib->Cast(x_shape, kInt32)});
  return {dx};
});

REG_BPROP_BUILDER("AdaptiveAvgPool2DV1").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("AdaptiveAvgPool2DGradV1", {dout}, {{"orig_input_shape", MakeValue(ib->GetShape(x))}});
  return {dx};
});

REG_BPROP_BUILDER("FractionalMaxPool").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit(
    "FractionalMaxPoolGrad",
    {x, ib->TupleGetItem(out, 0), ib->TupleGetItem(dout, 0), ib->TupleGetItem(out, 1), ib->TupleGetItem(out, 2)},
    {{"overlapping", ib->GetAttr("overlapping")}});

  return {dx};
});

REG_BPROP_BUILDER("FractionalMaxPool3DWithFixedKsize").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto random_samples = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("FractionalMaxPool3DGradWithFixedKsize", {x, ib->TupleGetItem(dout, 0), ib->TupleGetItem(out, 1)},
                     {{"format", ib->GetAttr("format")}});
  return {dx, ib->ZerosLike(random_samples)};
});

REG_BPROP_BUILDER("FractionalAvgPool").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto x_shape = ib->GetShape(x);

  auto dx = ib->Emit("FractionalAvgPoolGrad",
                     {ib->Cast(ib->Tensor(x_shape), kInt64), ib->TupleGetItem(dout, 0), ib->TupleGetItem(out, 1),
                      ib->TupleGetItem(out, 2)},
                     {{"overlapping", ib->GetAttr("overlapping")}, {"max_length", MakeValue<int64_t>(1000000)}});

  return {dx};
});

REG_BPROP_BUILDER("PSROIPooling").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto spatial_scale = ib->GetAttr("spatial_scale");
  auto group_size = ib->GetAttr("group_size");
  auto output_dim = ib->GetAttr("output_dim");
  auto x = ib->GetInput(kIndex0);
  auto rois = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto shape = ib->GetShape(x);
  ShapeVector input_size;
  for (size_t i = 2; i < shape.size(); i++) {
    input_size.push_back(shape[i]);
  }
  auto dx = ib->Emit("PSROIPoolingGrad", {dout, rois},
                     {
                       {"input_size", MakeValue(input_size)},
                       {"spatial_scale", spatial_scale},
                       {"group_size", group_size},
                       {"output_dim", output_dim},
                     });
  return {dx, ib->ZerosLike(rois)};
});

REG_BPROP_BUILDER("AvgPoolV1").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto orig_input_shape = ib->Value<ShapeVector>(ib->GetShape(x));
  auto dx = ib->Emit("AvgPoolGradV1", {orig_input_shape, dout},
                     {
                       {"kernel_size", ib->GetAttr("kernel_size")},
                       {"strides", ib->GetAttr("strides")},
                       {"pad_mode", ib->GetAttr("pad_mode")},
                       {"format", ib->GetAttr("format")},
                     });
  return {dx};
});

REG_BPROP_BUILDER("MaxPoolV1").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("MaxPoolGradV1", {x, out, dout},
                     {
                       {"kernel_size", ib->GetAttr("kernel_size")},
                       {"strides", ib->GetAttr("strides")},
                       {"pad_mode", ib->GetAttr("pad_mode")},
                       {"format", ib->GetAttr("format")},
                     });
  return {dx};
});

REG_BPROP_BUILDER("CTCLossV2").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto log_probs = ib->GetInput(kIndex0);
  auto targets = ib->GetInput(kIndex1);
  auto input_lengths = ib->GetInput(kIndex2);
  auto target_lengths = ib->GetInput(kIndex3);
  auto out = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex5);
  auto grad = ib->Emit("CTCLossV2Grad",
                       {ib->TupleGetItem(dout, 0), log_probs, targets, input_lengths, target_lengths,
                        ib->TupleGetItem(out, 0), ib->TupleGetItem(out, 1)},
                       {{"blank", ib->GetAttr("blank")},
                        {"reduction", ib->GetAttr("reduction")},
                        {"zero_infinity", ib->GetAttr("zero_infinity")}});
  return {grad, ib->ZerosLike(targets), ib->ZerosLike(input_lengths), ib->ZerosLike(target_lengths)};
});

REG_BPROP_BUILDER("InstanceNormV2").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto gamma = ib->GetInput(kIndex1);
  auto mean = ib->GetInput(kIndex3);
  auto variance = ib->GetInput(kIndex4);
  auto out = ib->GetInput(kIndex5);
  auto dout = ib->GetInput(kIndex6);
  auto saved_mean = ib->TupleGetItem(out, 1);
  auto saved_variance = ib->TupleGetItem(out, 2);
  auto grad_ops_out =
    ib->Emit("InstanceNormV2Grad", {ib->TupleGetItem(dout, 0), x, gamma, mean, variance, saved_mean, saved_variance},
             {{"is_training", ib->GetAttr("is_training")},
              {"epsilon", ib->GetAttr("epsilon")},
              {"momentum", ib->GetAttr("momentum")}});
  auto dx = ib->TupleGetItem(grad_ops_out, 0);
  auto dgamma = ib->TupleGetItem(grad_ops_out, 1);
  auto dbeta = ib->TupleGetItem(grad_ops_out, 2);
  return {dx, dgamma, dbeta, ib->ZerosLike(mean), ib->ZerosLike(variance)};
});

REG_BPROP_BUILDER("FractionalMaxPoolWithFixedKsize").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto random_samples = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("FractionalMaxPoolGradWithFixedKsize", {x, ib->TupleGetItem(dout, 0), ib->TupleGetItem(out, 1)},
                     {{"ksize", ib->GetAttr("ksize")},
                      {"output_shape", ib->GetAttr("output_shape")},
                      {"format", ib->GetAttr("format")}});
  return {dx, ib->ZerosLike(random_samples)};
});

REG_BPROP_BUILDER("AdaptiveAvgPool2D").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  MS_LOG(WARNING) << "Bprop Expander under testing: " << ib->name();
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("AdaptiveAvgPool2DGrad", {x, dout});
  return {dx};
});

REG_BPROP_BUILDER("SparseSoftmaxCrossEntropyWithLogitsV2").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto logits = ib->GetInput(kIndex0);
  auto labels = ib->GetInput(kIndex1);
  auto out = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto grad_loss = ib->TupleGetItem(dout, 0);
  auto softmax_grad = ib->TupleGetItem(out, 1);
  grad_loss = ib->ExpandDims(grad_loss, -1);
  auto grad = ib->Mul(grad_loss, softmax_grad);
  auto btmp = ib->TupleGetItem(dout, 1);
  if (ib->TupleGetItem(dout, 1) != nullptr) {
    auto softmax = ib->Emit("Softmax", {logits}, {{"axis", MakeValue(ShapeVector{1})}});
    auto x = ib->ExpandDims(ib->TupleGetItem(dout, 1), 1);
    auto y = ib->ExpandDims(softmax, 2);
    auto matmul_tmp = ib->BatchMatMul(x, y);
    grad =
      grad +
      (ib->TupleGetItem(dout, 1) - ib->Emit("Squeeze", {matmul_tmp}, {{"axis", MakeValue(ShapeVector{1})}})) * softmax;
  }
  return {grad, ib->ZerosLike(labels)};
});

REG_BPROP_BUILDER("DepthwiseConv2dNative").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto w = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("DepthwiseConv2dNativeBackpropInput", {ib->Value<ShapeVector>(ib->GetShape(x)), w, dout},
                     {{"channel_multiplier", ib->GetAttr("channel_multiplier")},
                      {"kernel_size", ib->GetAttr("kernel_size")},
                      {"pad_mode", ib->GetAttr("pad_mode")},
                      {"pad", ib->GetAttr("pad")},
                      {"pad_list", ib->GetAttr("pad_list")},
                      {"mode", ib->GetAttr("mode")},
                      {"stride", ib->GetAttr("stride")},
                      {"dilation", ib->GetAttr("dilation")},
                      {"group", ib->GetAttr("group")}});
  auto dw = ib->Emit("DepthwiseConv2dNativeBackpropFilter", {x, ib->Value<ShapeVector>(ib->GetShape(w)), dout},
                     {{"channel_multiplier", ib->GetAttr("channel_multiplier")},
                      {"kernel_size", ib->GetAttr("kernel_size")},
                      {"pad_mode", ib->GetAttr("pad_mode")},
                      {"pad", ib->GetAttr("pad")},
                      {"pad_list", ib->GetAttr("pad_list")},
                      {"mode", ib->GetAttr("mode")},
                      {"stride", ib->GetAttr("stride")},
                      {"dilation", ib->GetAttr("dilation")},
                      {"group", ib->GetAttr("group")}});
  return {dx, dw};
});

REG_BPROP_BUILDER("PadV3").SetBody([](const BpropIRBuilder *ib) -> NodePtrList {
  auto x = ib->GetInput(kIndex0);
  auto paddings = ib->GetInput(kIndex1);
  auto constant_values = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto mode = GetValue<std::string>(ib->GetAttr("mode"));
  NodePtr dx;
  auto pad = paddings->get()->abstract()->BuildValue();
  MS_EXCEPTION_IF_NULL(pad);
  std::vector<int64_t> pad_value;
  if (pad->isa<tensor::Tensor>()) {
    pad_value = CheckAndConvertUtils::CheckTensorIntValue("paddings value", pad, "PadV3");
  } else {
    pad_value = CheckAndConvertUtils::CheckTupleInt("paddings tuple value", pad, "PadV3");
  }
  if (mode == "constant") {
    (void)std::transform(pad_value.begin(), pad_value.end(), pad_value.begin(), [](const int64_t &c) { return -c; });
    dx = ib->Emit("PadV3", {dout, ib->Tensor(pad_value), ib->ZerosLike(constant_values)},
                  {{"mode", ib->GetAttr("mode")}, {"paddings_contiguous", ib->GetAttr("paddings_contiguous")}});
  } else {
    dx = ib->Emit("PadV3Grad", {dout, ib->Tensor(pad_value)},
                  {{"mode", ib->GetAttr("mode")}, {"paddings_contiguous", ib->GetAttr("paddings_contiguous")}});
  }
  return {dx, ib->ZerosLike(paddings), ib->ZerosLike(constant_values)};
});

REG_BPROP_BUILDER("MaximumGrad").SetBody(CommonMaxMinGradBprop);
REG_BPROP_BUILDER("MinimumGrad").SetBody(CommonMaxMinGradBprop);
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
