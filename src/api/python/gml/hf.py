# Copyright 2023- Gimlet Labs, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import glob
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, TextIO

import gml.proto.src.api.corepb.v1.model_exec_pb2 as modelexecpb
import torch
from gml.model import GenerationConfig, Model, TorchModel
from gml.tensor import (
    AttentionMaskDimension,
    BatchDimension,
    TensorSemantics,
    TokensDimension,
    UnusedTensorSemantics,
    VocabLogitsDimension,
)
from transformers import Pipeline, PreTrainedModel, PreTrainedTokenizer


class HuggingFaceTokenizer(Model):
    def __init__(self, tokenizer: PreTrainedTokenizer, name: Optional[str] = None):
        if name is None:
            name = tokenizer.name_or_path + ".tokenizer"
        super().__init__(
            name=name,
            kind=modelexecpb.ModelInfo.MODEL_KIND_HUGGINGFACE_TOKENIZER,
            storage_format=modelexecpb.ModelInfo.MODEL_STORAGE_FORMAT_OPAQUE,
            input_tensor_semantics=[],
            output_tensor_semantics=[],
        )
        self.tokenizer = tokenizer

    def _collect_assets(self) -> Dict[str, TextIO | BinaryIO | Path]:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.tokenizer.save_pretrained(tmpdir)
            paths = [Path(f) for f in glob.glob(tmpdir + "/*")]
            yield {p.name: p for p in paths}


class HuggingFaceGenerationConfig(GenerationConfig):

    def __init__(self, model: PreTrainedModel):
        config = model.generation_config
        eos_tokens = config.eos_token_id
        if eos_tokens is None:
            eos_tokens = []
        if not isinstance(eos_tokens, list):
            eos_tokens = [eos_tokens]
        super().__init__(eos_tokens)


def flatten(items):
    flattened = []
    if isinstance(items, torch.Tensor) or not isinstance(items, Iterable):
        flattened.append(items)
    else:
        for x in items:
            flattened.extend(flatten(x))
    return flattened


class HuggingFaceTextGenerationPipeline:
    def __init__(
        self,
        pipeline: Pipeline,
        name: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        dynamic_seqlen: bool = True,
        max_seqlen: int = 65536,
    ):
        self.tokenizer_model = HuggingFaceTokenizer(pipeline.tokenizer, tokenizer_name)
        if name is None:
            name = pipeline.model.name_or_path
        self.language_model = TorchModel(
            name,
            torch_module=pipeline.model,
            **self._guess_model_spec(pipeline, dynamic_seqlen, max_seqlen)
        )

    def _guess_model_spec(
        self, pipeline: Pipeline, dynamic_seqlen: bool, max_seqlen: int
    ) -> Dict:
        input_dict = pipeline.preprocess("this is a prompt! Test test test?")
        if "input_ids" not in input_dict:
            raise ValueError(
                'HuggingFaceTextGenerationPipeline expects preprocessed inputs to have an "input_ids" tensor'
            )

        inputs = []
        input_tensor_semantics = []

        # This currently assumes that all HF language models have inputs that are [B, NUM_TOKENS].
        inputs.append(input_dict["input_ids"])
        input_tensor_semantics.append(
            TensorSemantics(
                dimensions=[
                    BatchDimension(),
                    TokensDimension(),
                ],
            )
        )

        try:
            outputs = pipeline.model(*inputs, return_dict=False, use_cache=False)
            needs_attention_mask = False
        except Exception:
            needs_attention_mask = True

        if needs_attention_mask and "attention_mask" in input_dict:
            inputs.append(input_dict["attention_mask"])
            input_tensor_semantics.append(
                TensorSemantics(
                    dimensions=[
                        BatchDimension(),
                        AttentionMaskDimension(),
                    ],
                )
            )

        if needs_attention_mask:
            outputs = pipeline.model(*inputs, return_dict=False, use_cache=False)

        # Determine output semantics.
        # TODO(james): support caching
        output_tensor_semantics = []
        seqlen = inputs[0].shape[1]
        found_logits = False
        for tensor in flatten(outputs):
            if not isinstance(tensor, torch.Tensor):
                continue

            if (
                not found_logits
                and len(tensor.shape) == 3
                and tensor.shape[0] == 1
                and tensor.shape[1] == seqlen
            ):
                # This should be the logits tensor.
                output_tensor_semantics.append(
                    TensorSemantics(
                        dimensions=[
                            BatchDimension(),
                            TokensDimension(),
                            VocabLogitsDimension(),
                        ],
                    )
                )
                found_logits = True
            else:
                output_tensor_semantics.append(UnusedTensorSemantics())

        dynamic_shapes = None
        if dynamic_seqlen:
            dynamic_shapes = [dict() for _ in inputs]
            seqlen = torch.export.Dim("seqlen", max=max_seqlen)
            for tensor_idx, semantics in enumerate(input_tensor_semantics):
                for dim_idx, dim in enumerate(semantics.dimensions):
                    if isinstance(dim, TokensDimension):
                        dynamic_shapes[tensor_idx][dim_idx] = seqlen

        return {
            "example_inputs": inputs,
            "dynamic_shapes": dynamic_shapes,
            "input_tensor_semantics": input_tensor_semantics,
            "output_tensor_semantics": output_tensor_semantics,
            "generation_config": HuggingFaceGenerationConfig(pipeline.model),
        }

    def models(self) -> List[Model]:
        return [self.tokenizer_model, self.language_model]


def import_huggingface_pipeline(pipeline: Pipeline, **kwargs) -> List[Model]:
    if pipeline.framework != "pt":
        raise ValueError(
            "unimplemented: hugging face pipeline framework: {}".format(
                pipeline.framework
            )
        )

    if pipeline.task == "text-generation":
        return HuggingFaceTextGenerationPipeline(pipeline, **kwargs).models()
    raise ValueError(
        "unimplemnted: hugging face pipeline task: {}".format(pipeline.task)
    )
