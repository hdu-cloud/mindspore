# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""test graph getattr, hasattr"""
import pytest
import numpy as np

from mindspore import Tensor, jit, context

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_getattr_tensor():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tensor input.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        abs_func = getattr(x, "abs")
        return abs_func()

    out = foo(Tensor([-1, -2, -3]))
    assert np.all(out.asnumpy() == np.array([1, 2, 3]))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_getattr_tensor_with_concate_string():
    """
    Feature: Syntax getattr.
    Description: Graph syntax getattr support tensor input and concate string.
    Expectation: No exception.
    """

    @jit
    def foo(x):
        attr_str = "a" + "bs"
        abs_func = getattr(x, attr_str)
        return abs_func()

    out = foo(Tensor([-1, -2, -3]))
    assert np.all(out.asnumpy() == np.array([1, 2, 3]))
