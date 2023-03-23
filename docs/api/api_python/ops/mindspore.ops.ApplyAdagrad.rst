mindspore.ops.ApplyAdagrad
===========================

.. py:class:: mindspore.ops.ApplyAdagrad(update_slots=True)

    ����Adagrad�㷨������ز�����

    Adagrad�㷨������ `Adaptive Subgradient Methods for Online Learning and Stochastic Optimization <http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf>`_ ���������Բ�ͬ���������������ȵ����⣬����Ӧ��Ϊ�����������䲻ͬ��ѧϰ�ʡ�

    .. math::
        \begin{array}{ll} \\
            accum += grad * grad \\
            var -= lr * grad * \frac{1}{\sqrt{accum}}
        \end{array}

    `var` �� `accum` �� `grad` ��������ѭ��ʽ����ת������ʹ��������һ�¡�������Ǿ��в�ͬ���������ͣ��ϵ;��ȵ��������ͽ�ת��Ϊ�����߾��ȵ��������͡�

    ������
        - **update_slots** (bool) - �Ƿ���� `accum` ���������ΪTrue�� `accum` �����¡�Ĭ��ֵΪ��True��

    ���룺
        - **var** (Parameter) - Ҫ���µ�Ȩ�ء���������Ϊfloat32��float16��shape�� :math:`(N, *)` ������ :math:`*` ��ʾ���������ĸ���ά�ȡ�
        - **accum** (Parameter) - Ҫ���µ��ۻ���shape���������ͱ����� `var` ��ͬ��
        - **lr** (Union[Number, Tensor]) - ѧϰ�ʣ�������Scalar����������Ϊfloat32��float16��
        - **grad** (Tensor) - �ݶȣ�Ϊһ��Tensor��shape���������ͱ����� `var` ��ͬ��

    �����
        2��Tensor��ɵ�tuple�����º�����ݡ�

        - **var** (Tensor) - shape������������ `var` ��ͬ��
        - **accum** (Tensor) - shape������������ `accum` ��ͬ��

    �쳣��
        - **TypeError** - ��� `var` �� `accum` �� `lr` �� `grad` ���������ͼȲ���float16Ҳ����float32��
        - **TypeError** - ��� `lr` �Ȳ�����ֵ��Ҳ����Tensor��
        - **RuntimeError** - ��� `var` �� `accum` �� `grad` ��֧����������ת����