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

#include "plugin/device/cpu/kernel/eigen/matrix_solve_ls_cpu_kernel.h"
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <algorithm>

namespace mindspore {
namespace kernel {
namespace {
constexpr auto kInputNum = 3;
constexpr auto kOutputNum = 1;
constexpr int64_t kNum2 = 2;
constexpr char kFast[] = "fast";
constexpr bool kMatrixSolveLsComputeOk = true;
constexpr bool kMatrixSolveLsComputeFailed = false;

template <typename InputIt, typename T>
T GetNumElements(InputIt first, InputIt last, T init) {
  for (; first != last; ++first) {
    init = std::move(init) * (*first);
  }
  return init;
}
}  // namespace
template <typename T, int Major>
using EigenMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Major>;

template <typename T>
void MatrixSolveLsCpuKernelMod::RealCholeskySingleCompute(T *aptr, T *bptr, T *xptr, double *l2, int64_t m, int64_t k,
                                                          int64_t n) {
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a(m, k);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x(k, n);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> b(m, n);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a_copy;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a_b;

  for (int i = 0; i < m * k; i++) {
    *(a.data() + i) = *(aptr + i);
  }
  for (int i = 0; i < m * n; i++) {
    *(b.data() + i) = *(bptr + i);
  }

  if (m >= k) {
    a_copy = a.transpose() * a + ((T)*l2) * EigenMatrix<T, Eigen::RowMajor>::Identity(k, k);
    a_b = a.transpose() * b;
  } else {
    a_copy = a * a.transpose() + ((T)*l2) * EigenMatrix<T, Eigen::RowMajor>::Identity(m, m);
    a_b = b;
  }
  for (int64_t i = 0; i < n; i++) {
    EigenMatrix<T, Eigen::RowMajor> xi = a_copy.ldlt().solve(a_b.col(i));
    if (m < k) {
      xi = a.transpose() * xi;
    }
    x.col(i) = xi;
  }
  for (int64_t i = 0; i < k * n; i++) {
    *(xptr + i) = *(x.data() + i);
  }
}

template <typename T>
void MatrixSolveLsCpuKernelMod::ComplexCholeskySingleCompute(std::complex<T> *aptr, std::complex<T> *bptr,
                                                             std::complex<T> *xptr, double *l2, int64_t m, int64_t k,
                                                             int64_t n) {
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(kNum2 * m, kNum2 * k);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x(kNum2 * k, n);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> b(kNum2 * m, n);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a_copy;
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> a_b;
  auto l2value = abs(*l2);

  for (int64_t i = 0; i < k; i++) {
    for (int64_t j = 0; j < m; j++) {
      *(A.data() + i + j * kNum2 * k) = std::real(*(aptr + i + j * k));
    }
    for (int64_t j = 0; j < m; j++) {
      *(A.data() + (i + k) + (j + m) * kNum2 * k) = std::real(*(aptr + i + j * k));
    }
    for (int64_t j = 0; j < m; j++) {
      *(A.data() + (i + k) + j * kNum2 * k) = -std::imag(*(aptr + i + j * k));
    }
    for (int64_t j = 0; j < m; j++) {
      *(A.data() + i + (j + m) * kNum2 * k) = std::imag(*(aptr + i + j * k));
    }
  }
  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = 0; j < m; j++) {
      *(b.data() + i + j * n) = std::real(*(bptr + i + j * n));
      *(b.data() + i + (j + m) * n) = std::imag(*(bptr + i + j * n));
    }
  }

  if (m >= k) {
    a_copy =
      A.transpose() * A +
      ((T)l2value) * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Identity(kNum2 * k, kNum2 * k);
    a_b = A.transpose() * b;
  } else {
    a_copy =
      A * A.transpose() +
      ((T)l2value) * Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Identity(kNum2 * m, kNum2 * m);
    a_b = b;
  }

  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> xi;
  for (int64_t i = 0; i < n; i++) {
    xi = a_copy.ldlt().solve(a_b.col(i));
    if (m < k) {
      xi = A.transpose() * xi;
    }
    x.col(i) = xi;
    for (int64_t j = 0; j < k; j++) {
      (xptr + i + j * n)->real(*(x.data() + i + j * n));
      (xptr + i + j * n)->imag(*(x.data() + i + (j + k) * n));
    }
  }
}

template <typename T>
void MatrixSolveLsCpuKernelMod::RealQrSingleCompute(T *aptr, T *bptr, T *xptr, int64_t m, int64_t k, int64_t n) {
  EigenMatrix<T, Eigen::RowMajor> a(m, k);
  EigenMatrix<T, Eigen::RowMajor> x(k, n);
  EigenMatrix<T, Eigen::RowMajor> b(m, n);

  for (int i = 0; i < m * k; i++) {
    *(a.data() + i) = *(aptr + i);
  }
  for (int i = 0; i < m * n; i++) {
    *(b.data() + i) = *(bptr + i);
  }

  Eigen::CompleteOrthogonalDecomposition<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> qr_solve(a);

  for (int64_t i = 0; i < n; i++) {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> xi = qr_solve.solve(b.col(i));
    x.col(i) = xi;
  }

  for (int64_t i = 0; i < k * n; i++) {
    *(xptr + i) = *(x.data() + i);
  }
}

template <typename T>
void MatrixSolveLsCpuKernelMod::ComplexQrSingleCompute(std::complex<T> *aptr, std::complex<T> *bptr,
                                                       std::complex<T> *xptr, int64_t m, int64_t k, int64_t n) {
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(kNum2 * m, kNum2 * k);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x(kNum2 * k, n);
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> b(kNum2 * m, n);

  for (int64_t i = 0; i < k; i++) {
    for (int64_t j = 0; j < m; j++) {
      *(A.data() + i + j * kNum2 * k) = std::real(*(aptr + i + j * k));
    }
    for (int64_t j = 0; j < m; j++) {
      *(A.data() + (i + k) + (j + m) * kNum2 * k) = std::real(*(aptr + i + j * k));
    }
    for (int64_t j = 0; j < m; j++) {
      *(A.data() + (i + k) + j * kNum2 * k) = -std::imag(*(aptr + i + j * k));
    }
    for (int64_t j = 0; j < m; j++) {
      *(A.data() + i + (j + m) * kNum2 * k) = std::imag(*(aptr + i + j * k));
    }
  }
  for (int64_t i = 0; i < n; i++) {
    for (int64_t j = 0; j < m; j++) {
      *(b.data() + i + j * n) = std::real(*(bptr + i + j * n));
      *(b.data() + i + (j + m) * n) = std::imag(*(bptr + i + j * n));
    }
  }

  Eigen::CompleteOrthogonalDecomposition<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> qr_solve(A);

  for (int64_t i = 0; i < n; i++) {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> xi = qr_solve.solve(b.col(i));
    x.col(i) = xi;

    for (int64_t j = 0; j < k; j++) {
      (xptr + i + j * n)->real(*(x.data() + i + j * n));
      (xptr + i + j * n)->imag(*(x.data() + i + (j + k) * n));
    }
  }
}

template <typename T>
bool MatrixSolveLsCpuKernelMod::ComplexCholesky(const std::vector<kernel::AddressPtr> &inputs,
                                                const std::vector<kernel::AddressPtr> &outputs) {
  auto x_shape_vector = AnfAlgo::GetInputDeviceShape(node_wpt_, 0);
  auto dims = x_shape_vector.size();
  auto l2 = reinterpret_cast<double *>(inputs[2]->addr);
  auto aptr = reinterpret_cast<std::complex<T> *>(inputs[0]->addr);
  auto bptr = reinterpret_cast<std::complex<T> *>(inputs[1]->addr);
  auto xptr = reinterpret_cast<std::complex<T> *>(outputs[0]->addr);
  int64_t m = x_shape_vector[dims - kNum2];
  int64_t k = x_shape_vector[dims - 1];
  int64_t n = 1;
  auto b_shape_vector = AnfAlgo::GetInputDeviceShape(node_wpt_, 1);
  if (b_shape_vector.size() > 1) {
    n = b_shape_vector[dims - 1];
  }
  int64_t data_num = 1;
  data_num = GetNumElements(x_shape_vector.begin(), x_shape_vector.end(), data_num);
  const int64_t mat_size = m * k;
  const int64_t rhs_size = m * n;
  const int64_t res_size = n * k;
  const int64_t batch = data_num / mat_size;
  const int64_t kParallelDataNum = 16 * mat_size;
  if (data_num >= kParallelDataNum) {
    auto sharder_matrix_solve_ls = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        ComplexCholeskySingleCompute(aptr + i * mat_size, bptr + i * rhs_size, xptr + i * res_size, l2, m, k, n);
      }
    };
    ParallelLaunchAutoSearch(sharder_matrix_solve_ls, batch, this, &parallel_search_info_);
  } else {
    for (int64_t i = 0; i < batch; i++) {
      ComplexCholeskySingleCompute(aptr + i * mat_size, bptr + i * rhs_size, xptr + i * res_size, l2, m, k, n);
    }
  }
  return kMatrixSolveLsComputeOk;
}

template <typename T>
bool MatrixSolveLsCpuKernelMod::RealQr(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  auto x_shape_vector = AnfAlgo::GetInputDeviceShape(node_wpt_, 0);
  auto dims = x_shape_vector.size();
  auto aptr = reinterpret_cast<T *>(inputs[0]->addr);
  auto bptr = reinterpret_cast<T *>(inputs[1]->addr);
  auto xptr = reinterpret_cast<T *>(outputs[0]->addr);
  int64_t m = x_shape_vector[dims - kNum2];
  int64_t k = x_shape_vector[dims - 1];
  int64_t n = 1;
  auto b_shape_vector = AnfAlgo::GetInputDeviceShape(node_wpt_, 1);
  if (b_shape_vector.size() > 1) {
    n = b_shape_vector[dims - 1];
  }
  int64_t data_num = 1;
  data_num = GetNumElements(x_shape_vector.begin(), x_shape_vector.end(), data_num);
  const int64_t mat_size = m * k;
  const int64_t rhs_size = m * n;
  const int64_t res_size = n * k;
  const int64_t batch = data_num / mat_size;
  const int64_t kParallelDataNum = 16 * mat_size;
  if (data_num >= kParallelDataNum) {
    auto sharder_matrix_solve_ls = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        RealQrSingleCompute(aptr + i * mat_size, bptr + i * rhs_size, xptr + i * res_size, m, k, n);
      }
    };
    ParallelLaunchAutoSearch(sharder_matrix_solve_ls, batch, this, &parallel_search_info_);
  } else {
    for (int64_t i = 0; i < batch; i++) {
      RealQrSingleCompute(aptr + i * mat_size, bptr + i * rhs_size, xptr + i * res_size, m, k, n);
    }
  }
  return kMatrixSolveLsComputeOk;
}

template <typename T>
bool MatrixSolveLsCpuKernelMod::ComplexQr(const std::vector<kernel::AddressPtr> &inputs,
                                          const std::vector<kernel::AddressPtr> &outputs) {
  auto x_shape_vector = AnfAlgo::GetInputDeviceShape(node_wpt_, 0);
  auto dims = x_shape_vector.size();
  int64_t m = x_shape_vector[dims - kNum2];
  int64_t k = x_shape_vector[dims - 1];
  int64_t n = 1;
  auto b_shape_vector = AnfAlgo::GetInputDeviceShape(node_wpt_, 1);
  if (b_shape_vector.size() > 1) {
    n = b_shape_vector[dims - 1];
  }
  int64_t data_num = 1;
  data_num = GetNumElements(x_shape_vector.begin(), x_shape_vector.end(), data_num);
  const int64_t mat_size = m * k;
  const int64_t rhs_size = m * n;
  const int64_t res_size = n * k;
  const int64_t batch = data_num / mat_size;
  const int64_t kParallelDataNum = 16 * mat_size;
  auto aptr = reinterpret_cast<std::complex<T> *>(inputs[0]->addr);
  auto bptr = reinterpret_cast<std::complex<T> *>(inputs[1]->addr);
  auto xptr = reinterpret_cast<std::complex<T> *>(outputs[0]->addr);
  if (data_num >= kParallelDataNum) {
    auto sharder_matrix_solve_ls = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        ComplexQrSingleCompute(aptr + i * mat_size, bptr + i * rhs_size, xptr + i * res_size, m, k, n);
      }
    };
    ParallelLaunchAutoSearch(sharder_matrix_solve_ls, batch, this, &parallel_search_info_);
  } else {
    for (int64_t i = 0; i < batch; i++) {
      ComplexQrSingleCompute(aptr + i * mat_size, bptr + i * rhs_size, xptr + i * res_size, m, k, n);
    }
  }
  return kMatrixSolveLsComputeOk;
}

template <typename T>
bool MatrixSolveLsCpuKernelMod::RealCholesky(const std::vector<kernel::AddressPtr> &inputs,
                                             const std::vector<kernel::AddressPtr> &outputs) {
  auto x_shape_vector = AnfAlgo::GetInputDeviceShape(node_wpt_, 0);
  auto dims = x_shape_vector.size();
  auto aptr = reinterpret_cast<T *>(inputs[0]->addr);
  auto bptr = reinterpret_cast<T *>(inputs[1]->addr);
  auto xptr = reinterpret_cast<T *>(outputs[0]->addr);
  auto l2 = reinterpret_cast<double *>(inputs[2]->addr);
  int64_t m = x_shape_vector[dims - kNum2];
  int64_t k = x_shape_vector[dims - 1];
  int64_t n = 1;
  auto b_shape_vector = AnfAlgo::GetInputDeviceShape(node_wpt_, 1);
  if (b_shape_vector.size() > 1) {
    n = b_shape_vector[dims - 1];
  }
  int64_t data_num = 1;
  data_num = GetNumElements(x_shape_vector.begin(), x_shape_vector.end(), data_num);
  const int64_t mat_size = m * k;
  const int64_t rhs_size = m * n;
  const int64_t res_size = n * k;
  const int64_t batch = data_num / mat_size;
  const int64_t kParallelDataNum = 16 * mat_size;
  if (data_num >= kParallelDataNum) {
    auto sharder_matrix_solve_ls = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        RealCholeskySingleCompute(aptr + i * mat_size, bptr + i * rhs_size, xptr + i * res_size, l2, m, k, n);
      }
    };
    ParallelLaunchAutoSearch(sharder_matrix_solve_ls, batch, this, &parallel_search_info_);
  } else {
    for (int64_t i = 0; i < batch; i++) {
      RealCholeskySingleCompute(aptr + i * mat_size, bptr + i * rhs_size, xptr + i * res_size, l2, m, k, n);
    }
  }
  return kMatrixSolveLsComputeOk;
}

void MatrixSolveLsCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  node_wpt_ = kernel_node;
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  if (common::AnfAlgo::HasNodeAttr(kFast, kernel_node)) {
    qr_chole = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, kFast);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the attribute 'fast' does not exist.";
  }
  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

bool MatrixSolveLsCpuKernelMod::Resize(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs) {
  auto shapea = AnfAlgo::GetInputDeviceShape(node_wpt_, 0);
  auto shapeb = AnfAlgo::GetInputDeviceShape(node_wpt_, 1);
  auto shapel2 = AnfAlgo::GetInputDeviceShape(node_wpt_, 2);
  auto shapex = AnfAlgo::GetOutputDeviceShape(node_wpt_, 0);
  auto dims = shapea.size();

  if (shapeb.size() == 1) {
    if (shapea[dims - kNum2] != shapeb[0]) {
      MS_EXCEPTION(ValueError) << "For " << kernel_name_ << ", #Rows mismatch between A and rhs."
                               << "#Rows of A = [" << shapea[dims - kNum2] << "]"
                               << "#Rows of rhs = [" << shapeb[0] << "]";
      return kMatrixSolveLsComputeFailed;
    }
  } else {
    if (shapea[dims - kNum2] != shapeb[dims - kNum2]) {
      MS_EXCEPTION(ValueError) << "For " << kernel_name_ << "#Rows mismatch between A and rhs."
                               << "#Rows of A = [" << shapea[dims - kNum2] << "]"
                               << "#Rows of rhs = [" << shapeb[dims - kNum2] << "]";
      return kMatrixSolveLsComputeFailed;
    }
  }

  if (shapel2.size() != 0) {
    MS_EXCEPTION(ValueError) << "For " << kernel_name_ << "Tensor l2 should be a scalar.";
    return kMatrixSolveLsComputeFailed;
  }
  if (shapeb.size() == 1) {
    if ((shapex.size() != shapeb.size()) || (shapea[dims - 1] != shapex[0]) || (shapex.back() != shapeb[0])) {
      MS_EXCEPTION(ValueError) << "For " << kernel_name_ << "Tensor y shape mismatch.";
      return kMatrixSolveLsComputeFailed;
    }
  } else {
    if ((shapex.size() != shapeb.size()) || (shapea[dims - 1] != shapex[shapex.size() - kNum2]) ||
        (shapex.back() != shapeb.back())) {
      MS_EXCEPTION(ValueError) << "For " << kernel_name_ << "Tensor y shape mismatch.";
      return kMatrixSolveLsComputeFailed;
    }
  }
  return kMatrixSolveLsComputeOk;
}

template <typename T>
bool MatrixSolveLsCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                             const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);

  auto a_data_type = AnfAlgo::GetInputDeviceDataType(node_wpt_, 0);
  auto b_data_type = AnfAlgo::GetInputDeviceDataType(node_wpt_, 1);

  if (Resize(inputs, outputs) != true) {
    return kMatrixSolveLsComputeFailed;
  }

  if (a_data_type != b_data_type) {
    MS_EXCEPTION(TypeError) << "For " << kernel_name_ << "Tensor data type mismatch.";
    return kMatrixSolveLsComputeFailed;
  }
  if (a_data_type != kNumberTypeFloat32 && a_data_type != kNumberTypeFloat64 && a_data_type != kNumberTypeComplex64 &&
      a_data_type != kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For " << kernel_name_ << ", unsupported data type: " << TypeIdLabel(a_data_type) << ".";
    return kMatrixSolveLsComputeFailed;
  }

  if (qr_chole) {
    if (a_data_type == kNumberTypeComplex64) {
      return ComplexCholesky<float>(inputs, outputs);
    }
    if (a_data_type == kNumberTypeComplex128) {
      return ComplexCholesky<double>(inputs, outputs);
    }
    if (a_data_type == kNumberTypeFloat64) {
      return RealCholesky<double>(inputs, outputs);
    }
    if (a_data_type == kNumberTypeFloat32) {
      return RealCholesky<float>(inputs, outputs);
    }
  } else {
    if (a_data_type == kNumberTypeComplex64) {
      return ComplexQr<float>(inputs, outputs);
    }
    if (a_data_type == kNumberTypeComplex128) {
      return ComplexQr<double>(inputs, outputs);
    }
    if (a_data_type == kNumberTypeFloat64) {
      return RealQr<double>(inputs, outputs);
    }
    if (a_data_type == kNumberTypeFloat32) {
      return RealQr<float>(inputs, outputs);
    }
  }
  return kMatrixSolveLsComputeOk;
}  // un pass

std::vector<std::pair<KernelAttr, MatrixSolveLsCpuKernelMod::MatrixSolveLsFunc>> MatrixSolveLsCpuKernelMod::func_list_ =
  {{KernelAttr()
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat64)
      .AddOutputAttr(kNumberTypeFloat32),
    &MatrixSolveLsCpuKernelMod::LaunchKernel<float>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeFloat64)
      .AddInputAttr(kNumberTypeFloat64)
      .AddInputAttr(kNumberTypeFloat64)
      .AddOutputAttr(kNumberTypeFloat64),
    &MatrixSolveLsCpuKernelMod::LaunchKernel<double>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeComplex64)
      .AddInputAttr(kNumberTypeComplex64)
      .AddInputAttr(kNumberTypeFloat64)
      .AddOutputAttr(kNumberTypeComplex64),
    &MatrixSolveLsCpuKernelMod::LaunchKernel<std::complex<float>>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeComplex128)
      .AddInputAttr(kNumberTypeComplex128)
      .AddInputAttr(kNumberTypeFloat64)
      .AddOutputAttr(kNumberTypeComplex128),
    &MatrixSolveLsCpuKernelMod::LaunchKernel<std::complex<double>>}};

std::vector<KernelAttr> MatrixSolveLsCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MatrixSolveLsFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MatrixSolveLs, MatrixSolveLsCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
