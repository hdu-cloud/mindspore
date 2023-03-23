mindspore.ops.NthElement
========================

.. py:class:: mindspore.ops.NthElement(reverse=False)

    寻找并返回输入Tensor最后一维第 :math:`n` 小的值。
    如果输入是Vector(rank为1)，寻找第n小的值并以Scalar Tensor类型输出结果。
    对于Matrix（rank大于1），计算最后一维每一行（各自可以看作一个Vector）第n小的值。因此，返回值 `values` 的shape满足 `values`.shape = `input`.shape[:-1]。

    参数：
        - **reverse** (bool，可选) - 可选参数，如果设为True，则寻找第 :math:`n` 大的值，如果设为False，则寻找第n小的值。默认值：False。

    输入：
        - **input** (Tensor) - 一维或者更高维度的Tensor，最后一维的大小必须大于等于 :math:`n+1` 。
        - **n** (Union[int, Tensor]) - 如果 :math:`n` 为Tensor，则必须是零维的，数据类型是int32。 :math:`n` 的有效范围是：:math:`[0, input.shape[-1])` 。

    输出：
        - **values** (Tensor) - 其shape满足： `values`.shape = `input`.shape[:-1]，数据类型与 `input` 一致。

    异常：
        - **TypeError** - `input` 的数据类型不在有效类型列表内。
        - **TypeError** - `n` 不是int32或者Tensor。
        - **ValueError** - `n` 不在 :math:`[0, input.shape[-1])` 范围内。
