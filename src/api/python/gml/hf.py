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
from typing import Any, BinaryIO, Dict, List, Optional, TextIO, Tuple

import gml.proto.src.api.corepb.v1.model_exec_pb2 as modelexecpb
import torch
import transformers
from gml.asset_manager import AssetManager
from gml.model import GenerationConfig, Model, TorchModel
from gml.preprocessing import (
    ImagePreprocessingStep,
    ImageToFloatTensor,
    LetterboxImage,
    ResizeImage,
    StandardizeTensor,
)
from gml.tensor import (
    AttentionKeyValueCacheTensorSemantics,
    BatchDimension,
    BoundingBoxFormat,
    DetectionNumCandidatesDimension,
    DetectionOutputDimension,
    ImageChannelDimension,
    ImageHeightDimension,
    ImageWidthDimension,
    SegmentationMaskChannel,
    TensorSemantics,
    TokensDimension,
    VocabLogitsDimension,
)
from transformers import (
    BaseImageProcessor,
    Pipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
)


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

    def _collect_assets(
        self, weight_manager: Optional[AssetManager] = None
    ) -> Dict[str, TextIO | BinaryIO | Path]:
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


class WrapWithFunctionalCache(torch.nn.Module):

    def __init__(self, model: transformers.PreTrainedModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids, cache):
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )

        return outputs.logits, outputs.past_key_values


class HuggingFaceTextGenerationPipeline:
    def __init__(
        self,
        pipeline: Pipeline,
        name: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        dynamic_seqlen: bool = False,
    ):
        self.pipeline = pipeline
        self.tokenizer_model = HuggingFaceTokenizer(pipeline.tokenizer, tokenizer_name)
        self._cache_length_for_tracing = 32
        if name is None:
            name = pipeline.model.name_or_path

        self.model = pipeline.model
        self.model = self.model.to(torch.float16)
        self.model = WrapWithFunctionalCache(pipeline.model)

        self.language_model = TorchModel(
            name,
            torch_module=self.model,
            **self._guess_model_spec(dynamic_seqlen),
        )

    def _initialize_key_value_cache(self):
        cache = []
        config = self.pipeline.model.config
        head_dim = (
            config.head_dim
            if hasattr(config, "head_dim")
            else config.hidden_size // config.num_attention_heads
        )
        num_key_value_heads = (
            config.num_attention_heads
            if config.num_key_value_heads is None
            else config.num_key_value_heads
        )
        cache_shape = (1, num_key_value_heads, self._cache_length_for_tracing, head_dim)
        for _ in range(config.num_hidden_layers):
            cache.append(
                [
                    torch.zeros(cache_shape).to(torch.float16),
                    torch.zeros(cache_shape).to(torch.float16),
                ]
            )
        return cache

    def _guess_model_spec(self, dynamic_seqlen: bool) -> Dict:
        input_dict = self.pipeline.preprocess("this is a prompt! Test test test?")
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

        # Assume that the model supports a KeyValue cache.
        cache_values = self._initialize_key_value_cache()
        inputs.append(cache_values)
        for _ in cache_values:
            input_tensor_semantics.append(AttentionKeyValueCacheTensorSemantics())
            input_tensor_semantics.append(AttentionKeyValueCacheTensorSemantics())

        outputs = self.model(*inputs)

        # Determine output semantics.
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
                output_tensor_semantics.append(AttentionKeyValueCacheTensorSemantics())

        dynamic_shapes = None
        seqlen = torch.export.Dim("seqlen", min=2, max=9223372036854775096)

        cache_length = torch.export.Dim("cache_length", min=2, max=9223372036854775096)
        dynamic_shapes = [
            {1: seqlen},
            [[{2: cache_length}, {2: cache_length}] for _ in cache_values],
        ]

        return {
            "example_inputs": inputs,
            "dynamic_shapes": dynamic_shapes,
            "input_tensor_semantics": input_tensor_semantics,
            "output_tensor_semantics": output_tensor_semantics,
            "generation_config": HuggingFaceGenerationConfig(self.pipeline.model),
        }

    def models(self) -> List[Model]:
        return [self.tokenizer_model, self.language_model]


class HuggingFaceImageProcessor:

    def __init__(
        self,
        model: PreTrainedModel,
        processor: BaseImageProcessor,
    ):
        self.model = model
        self.processor = processor

    def input_spec(self) -> Dict[str, Any]:
        target_size = None
        image_preprocessing_steps = []
        if (
            hasattr(self.processor, "do_resize")
            and self.processor.do_resize
            and hasattr(self.processor, "size")
        ):
            target_size, preprocessing_step = self._convert_resize()
            image_preprocessing_steps.append(preprocessing_step)

        if (
            hasattr(self.processor, "do_rescale")
            and self.processor.do_rescale
            and hasattr(self.processor, "rescale_factor")
        ):
            image_preprocessing_steps.append(
                ImageToFloatTensor(
                    scale=True, scale_factor=self.processor.rescale_factor
                )
            )
        else:
            image_preprocessing_steps.append(ImageToFloatTensor(scale=False))

        if hasattr(self.processor, "do_normalize") and self.processor.do_normalize:
            image_preprocessing_steps.append(
                StandardizeTensor(self.processor.image_mean, self.processor.image_std)
            )

        channels_first = True
        if (
            hasattr(self.processor, "input_data_format")
            and self.processor.input_data_format == "channels_last"
        ):
            channels_first = False

        # Assume RGB for now.
        # TODO(james): figure out if this is specified anywhere in the huggingface pipeline.
        channel_format = "rgb"

        dimensions = [
            BatchDimension(),
        ]
        input_shape = [1]
        if channels_first:
            dimensions.append(ImageChannelDimension(channel_format))
            input_shape.append(3)
        dimensions.append(ImageHeightDimension())
        input_shape.append(target_size[0])
        dimensions.append(ImageWidthDimension())
        input_shape.append(target_size[1])
        if not channels_first:
            dimensions.append(ImageChannelDimension(channel_format))
            input_shape.append(3)

        example_input = torch.rand(input_shape)
        input_tensor_semantics = [TensorSemantics(dimensions)]
        return {
            "example_inputs": [example_input],
            "input_tensor_semantics": input_tensor_semantics,
            "image_preprocessing_steps": image_preprocessing_steps,
        }

    def output_spec_segmentation(self) -> Dict[str, Any]:
        if not hasattr(self.processor, "post_process_semantic_segmentation"):
            raise NotImplementedError(
                "only semantic segmentation is currently supported"
            )
        dimensions = [
            BatchDimension(),
            # TODO(james): verify all semantic segmentation in hugging face output a logits mask.
            SegmentationMaskChannel("logits_mask"),
            ImageHeightDimension(),
            ImageWidthDimension(),
        ]
        output_tensor_semantics = [
            TensorSemantics(dimensions),
        ]
        id_to_label = self.model.config.id2label
        max_id = max(id_to_label)
        labels = []
        for i in range(max_id):
            if i not in id_to_label:
                labels.append("")
                continue
            labels.append(id_to_label[i])
        return {
            "output_tensor_semantics": output_tensor_semantics,
            "class_labels": labels,
        }

    def output_spec_object_detection(self) -> Dict[str, Any]:
        if not hasattr(self.processor, "post_process_object_detection"):
            raise NotImplementedError(
                "processor must have post_process_object_detection set"
            )

        id_to_label = self.model.config.id2label
        max_id = max(id_to_label)
        labels = []
        for i in range(max_id):
            if i not in id_to_label:
                labels.append("")
                continue
            labels.append(id_to_label[i])
        num_classes = max_id + 1

        # TODO(james): verify assumptions made here apply broadly.
        output_tensor_semantics = []
        # We assume that ObjectDetectionWrapper is used to ensure that logits are the first tensor and boxes are the second.
        logits_dimensions = [
            BatchDimension(),
            DetectionNumCandidatesDimension(is_nms=False),
            DetectionOutputDimension(
                scores_range=(0, num_classes),
                scores_are_logits=True,
            ),
        ]
        output_tensor_semantics.append(TensorSemantics(logits_dimensions))

        box_dimensions = [
            BatchDimension(),
            DetectionNumCandidatesDimension(is_nms=False),
            DetectionOutputDimension(
                coordinates_start_index=0,
                box_format=BoundingBoxFormat("cxcywh", is_normalized=True),
            ),
        ]
        output_tensor_semantics.append(TensorSemantics(box_dimensions))
        return {
            "output_tensor_semantics": output_tensor_semantics,
            "class_labels": labels,
        }

    def _convert_resize(self) -> Tuple[Tuple[int, int], ImagePreprocessingStep]:
        size = self.processor.size
        target_size = None
        preprocess_step = None
        if "height" in size and "width" in size:
            target_size = [size["height"], size["width"]]
            preprocess_step = ResizeImage()
        elif (
            "shortest_edge" in size
            or "longest_edge" in size
            or "max_height" in size
            or "max_width" in size
        ):
            shortest_edge = size.get("shortest_edge")
            longest_edge = size.get("longest_edge")
            max_height = size.get("max_height")
            max_width = size.get("max_width")

            min_size = None
            for edge_size in [shortest_edge, longest_edge, max_height, max_width]:
                if not edge_size:
                    continue
                if not min_size or edge_size < min_size:
                    min_size = edge_size

            target_size = [min_size, min_size]
            preprocess_step = LetterboxImage()
        else:
            raise ValueError(
                "could not determine target size for resize from model config"
            )
        return target_size, preprocess_step


class HuggingFaceImageSegmentationPipeline:
    def __init__(
        self,
        pipeline: Pipeline,
        name: Optional[str] = None,
    ):
        self.pipeline = pipeline
        if name is None:
            name = pipeline.model.name_or_path

        self.model = TorchModel(
            name,
            torch_module=self.pipeline.model,
            **self._guess_model_spec(),
        )

    def _guess_model_spec(self) -> Dict:
        if self.pipeline.image_processor is None:
            raise ValueError(
                "Could not determine image preprocessing for pipeline with image_processor=None"
            )
        if self.pipeline.tokenizer is not None:
            raise NotImplementedError(
                "HuggingFaceImageSegmentationPipeline does not yet support token inputs"
            )

        image_processor = HuggingFaceImageProcessor(
            self.pipeline.model, self.pipeline.image_processor
        )
        spec = image_processor.input_spec()
        spec.update(image_processor.output_spec_segmentation())
        return spec

    def models(self) -> List[Model]:
        return [self.model]


class ObjectDetectionWrapper(torch.nn.Module):
    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        return outputs.logits, outputs.pred_boxes


class HuggingFaceObjectDetectionPipeline:
    def __init__(
        self,
        pipeline: Pipeline,
        name: Optional[str] = None,
    ):
        self.pipeline = pipeline
        if name is None:
            name = pipeline.model.name_or_path

        self.model = TorchModel(
            name,
            torch_module=ObjectDetectionWrapper(self.pipeline.model),
            **self._guess_model_spec(),
        )

    def _guess_model_spec(self) -> Dict:
        if self.pipeline.image_processor is None:
            raise ValueError(
                "Could not determine image preprocessing for pipeline with image_processor=None"
            )
        if self.pipeline.tokenizer is not None:
            raise NotImplementedError(
                "HuggingFaceObjectDetectionPipeline does not yet support token inputs"
            )

        image_processor = HuggingFaceImageProcessor(
            self.pipeline.model, self.pipeline.image_processor
        )
        spec = image_processor.input_spec()
        spec.update(image_processor.output_spec_object_detection())
        return spec

    def models(self) -> List[Model]:
        return [self.model]


def import_huggingface_pipeline(pipeline: Pipeline, **kwargs) -> List[Model]:
    if pipeline.framework != "pt":
        raise ValueError(
            "unimplemented: hugging face pipeline framework: {}".format(
                pipeline.framework
            )
        )

    if pipeline.task == "text-generation":
        return HuggingFaceTextGenerationPipeline(pipeline, **kwargs).models()
    elif pipeline.task == "image-segmentation":
        return HuggingFaceImageSegmentationPipeline(pipeline, **kwargs).models()
    elif pipeline.task == "object-detection":
        return HuggingFaceObjectDetectionPipeline(pipeline, **kwargs).models()
    raise ValueError(
        "unimplemented: hugging face pipeline task: {} (supported tasks: [{}])".format(
            pipeline.task, ["text-generation", "image-segmentation", "object-detection"]
        )
    )
