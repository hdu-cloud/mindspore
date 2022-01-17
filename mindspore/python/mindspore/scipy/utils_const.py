# Copyright 2021 Huawei Technologies Co., Ltd
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
"""internal graph-compatible utility functions"""
from ..ops.primitive import constexpr
from .._c_expression import typing


@constexpr
def _callable_const(x):
    """Returns true if x is a function in graph mode."""
    return isinstance(x, typing.Function)


@constexpr
def _type_convert(new_type, obj):
    """
    Convert type of `obj` to `force`.
    """
    return new_type(obj)


@constexpr
def _raise_value_error(info):
    """
    Raise ValueError in both graph/pynative mode

    Args:
        info(str): info string to display
    """
    raise ValueError(info)


@constexpr
def _raise_type_error(info):
    """
    Raise TypeError in both graph/pynative mode

    Args:
        info(str): info string to display
    """
    raise TypeError(info)
