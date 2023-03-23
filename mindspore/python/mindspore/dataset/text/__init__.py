# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""
This module is to support text processing for NLP. It includes two parts:
text transforms and utils. text transforms is a high performance
NLP text processing module which is developed with ICU4C and cppjieba.
utils provides some general methods for NLP text processing.

Common imported modules in corresponding API examples are as follows:

.. code-block::

    import mindspore.dataset as ds
    import mindspore.dataset.text as text

See `Text Transforms
<https://www.mindspore.cn/tutorials/en/r2.0.0-alpha/beginner/transforms.html#text-transforms>`_ tutorial
for more details.

Descriptions of common data processing terms are as follows:

- TensorOperation, the base class of all data processing operations implemented in C++.
- TextTensorOperation, the base class of all text processing operations. It is a derived class of TensorOperation.

The data transform operation can be executed in the data processing pipeline or in the eager mode:

- Pipeline mode is generally used to process datasets. For examples, please refer to
  `introduction to data processing pipeline <https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/
  mindspore.dataset.html#introduction-to-data-processing-pipeline>`_ .
- Eager mode is generally used for scattered samples. Examples of text preprocessing are as follows:

  .. code-block::

      import mindspore.dataset.text as text
      from mindspore.dataset.text import NormalizeForm

      # construct vocab
      vocab_list = {"music": 1, "Opera": 2, "form": 3, "theatre": 4, "which": 5, "in": 6,
                    "fundamental": 7, "dramatic": 8, "component": 9, "taken": 10, "roles": 11, "singers": 12,
                    "is": 13, "are": 14, "of": 15, "UNK": 16}
      vocab = text.Vocab.from_dict(vocab_list)
      tokenizer_op = text.BertTokenizer(vocab=vocab, suffix_indicator='##', max_bytes_per_token=100,
                                        unknown_token='[UNK]', lower_case=False, keep_whitespace=False,
                                        normalization_form=NormalizeForm.NONE, preserve_unused_token=True,
                                        with_offsets=False)
      # tokenizer
      tokens = tokenizer_op("Opera is a form of theatre in which music is a fundamental "
                            "component and dramatic roles are taken by singers.")
      print("token: {}".format(tokens), flush=True)

      # token to ids
      ids = vocab.tokens_to_ids(tokens)
      print("token to id: {}".format(ids), flush=True)

      # ids to token
      tokens_from_ids = vocab.ids_to_tokens([15, 3, 7])
      print("token to id: {}".format(tokens_from_ids), flush=True)
"""
import platform

from . import transforms
from . import utils
from .transforms import JiebaTokenizer, Lookup, Ngram, PythonTokenizer, SentencePieceTokenizer, SlidingWindow, \
    ToNumber, ToVectors, TruncateSequencePair, UnicodeCharTokenizer, WordpieceTokenizer
from .utils import CharNGram, FastText, GloVe, JiebaMode, NormalizeForm, SentencePieceModel, SentencePieceVocab, \
    SPieceTokenizerLoadType, SPieceTokenizerOutType, Vectors, Vocab, to_bytes, to_str

if platform.system().lower() != 'windows':
    from .transforms import BasicTokenizer, BertTokenizer, CaseFold, FilterWikipediaXML, NormalizeUTF8, RegexReplace, \
        RegexTokenizer, UnicodeScriptTokenizer, WhitespaceTokenizer
