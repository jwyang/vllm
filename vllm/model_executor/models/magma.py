# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from functools import cached_property
from typing import (Final, Iterable, List, Literal, Mapping, Optional,
                    Protocol, Set, Tuple, TypedDict, TypeVar, Union, Sequence)
import re
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers import BatchFeature, MagmaConfig, MagmaProcessor
from transformers.models.magma.modeling_magma import MagmaVisionModel
from typing_extensions import NotRequired

from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import (ImageEmbeddingItems, ImageProcessorItems,
                                   ImageSize, MultiModalDataItems)
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig, MultiModalKwargs)
from vllm.multimodal.parse import ImageSize
from vllm.sequence import IntermediateTensors
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, ProcessingCache,
                                        PromptReplacement, PromptUpdate,
                                        PromptUpdateDetails)

from .clip import CLIPVisionModel
from .interfaces import MultiModalEmbeddings, SupportsMultiModal, SupportsPP
from .siglip import SiglipVisionModel
from .utils import (AutoWeightsLoader, embed_multimodal, flatten_bn,
                    init_vllm_registered_model, maybe_prefix, merge_multimodal_embeddings)
from .vision import get_vision_encoder_info


class MagmaImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: Union[torch.Tensor, list[torch.Tensor]]
    """
    Shape:
    `(batch_size * num_images, 1 + num_patches, num_channels, height, width)`

    Note that `num_patches` may be different per batch and image,
    in which case the data is passed as a list instead of a batched tensor.
    """

    image_sizes: NotRequired[torch.Tensor]
    """
    Shape: `(batch_size * num_images, 2)`

    This should be in `(height, width)` format.
    """


class MagmaImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


MagmaImageInputs = Union[MagmaImagePixelInputs,
                             MagmaImageEmbeddingInputs]


class MagmaLikeConfig(Protocol):
    vision_config: Final[PretrainedConfig]
    image_token_id: Final[int]
    vision_feature_select_strategy: Final[str]
    vision_feature_layer: Final[Union[int, list[int]]]

class MagmaLikeProcessor(Protocol):
    image_token: Final[str]

class BaseMagmaProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self) -> MagmaLikeConfig:
        return self.ctx.get_hf_config(MagmaConfig)

    def get_vision_encoder_info(self):
        return get_vision_encoder_info(self.get_hf_config())

    @abstractmethod
    def get_hf_processor(self, **kwargs: object) -> MagmaLikeProcessor:
        raise NotImplementedError

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def _apply_feature_select_strategy(
        self,
        strategy: str,
        encoder_num_image_tokens: int,
    ) -> int:
        if strategy == "default":
            return encoder_num_image_tokens - 1
        if strategy == "full":
            return encoder_num_image_tokens

        msg = f"Unexpected feature select strategy: {strategy!r}"
        raise NotImplementedError(msg)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_config = self.get_hf_config()
        vision_encoder_info = self.get_vision_encoder_info()

        return self._apply_feature_select_strategy(
            hf_config.vision_feature_select_strategy,
            vision_encoder_info.get_num_image_tokens(
                image_width=image_width,
                image_height=image_height,
            ),
        )

    def get_image_size_with_most_features(self) -> ImageSize:
        return ImageSize(width=512, height=512)

_I = TypeVar("_I", bound=BaseMagmaProcessingInfo)

class MagmaProcessingInfo(BaseMagmaProcessingInfo):

    def get_hf_config(self) -> MagmaLikeConfig:
        return self.ctx.get_hf_config(MagmaConfig)

    def get_hf_processor(self, **kwargs: object):
        hf_processor = self.ctx.get_hf_processor(MagmaProcessor, **kwargs)
        return hf_processor

_I = TypeVar("_I", bound=MagmaProcessingInfo)

class MagmaDummyInputsBuilder(BaseDummyInputsBuilder[_I]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = processor.image_token

        return image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = \
            self.info.get_image_size_with_most_features()

        return {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images)
        }


class BaseMagmaMultiModalProcessor(BaseMultiModalProcessor[_I]):

    # Copied from BaseMultiModalProcessor
    @abstractmethod
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        raise NotImplementedError

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> Sequence[PromptUpdate]:
        
        image_token_id = self.info.get_hf_config().image_token_id
        hf_processor = self.info.get_hf_processor()

        def get_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems))

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                image = images.get(item_idx)
                image_inputs = hf_processor.image_processor(image)
                image_sizes = image_inputs['image_sizes'].view(-1).tolist()
                num_image_tokens = image_sizes[0] * image_sizes[1] * 256 + image_sizes[0] * 16
            return [image_token_id] * num_image_tokens
    
        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            ),
        ]

class MagmaMultiModalProcessor(
        BaseMagmaMultiModalProcessor[MagmaProcessingInfo]):

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )

        input_ids = processed_outputs["input_ids"]

        if "pixel_values" in processed_outputs and "image_sizes" in processed_outputs:
            # image_token_id = self.info.get_hf_config().image_token_id
            # # replace the image_token_id with the number of image tokens that is equal to the number of image tokens in the image
            # image_sizes = processed_outputs["image_sizes"]  # (batch_size, num_images, 2)
            # assert input_ids.shape[0] == image_sizes.shape[0]
            # input_ids_list = input_ids.tolist()
            # assert len(input_ids_list) == image_sizes.shape[0]
            # input_ids_list_filled = []
            # for i in range(input_ids.shape[0]):
            #     input_ids_orig = input_ids[i].tolist()
            #     input_ids_list_filled.append([])
            #     img_idx = 0
            #     for id in input_ids_orig:
            #         if id == image_token_id:
            #             # replace the image_token_id with the number of image tokens that is equal to the number of image tokens in the image
            #             assert image_sizes[i][img_idx][0] != 0 and image_sizes[i][img_idx][1] != 0, "some mismatch happens, please double check your prompt processor"
            #             num_image_tokens = image_sizes[i][img_idx][0] * image_sizes[i][img_idx][1] * 256 + image_sizes[i][img_idx][0] * 16
            #             input_ids_list_filled[i].extend([image_token_id] * num_image_tokens)                        
            #             img_idx += 1
            #         else:
            #             input_ids_list_filled[i].append(id)
            # processed_outputs["input_ids"] = torch.tensor(input_ids_list_filled)
            # processed_outputs["attention_mask"] = torch.ones_like(processed_outputs["input_ids"])

            image_sizes = processed_outputs["image_sizes"].squeeze(0)
            pixel_values = processed_outputs["pixel_values"].squeeze(0)

            # NOTE: this assumes the images for a single sample are all with the same size
            num_images = image_sizes.shape[0]
            pixel_values = torch.stack(pixel_values.split(pixel_values.shape[0] // num_images, dim=0))
            image_sizes = torch.cat(image_sizes.split(num_images, dim=0))
            processed_outputs["image_sizes"] = image_sizes
            processed_outputs["pixel_values"] = pixel_values

        return processed_outputs
    
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_sizes=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )

class MagmaMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        projector_type = config.mm_projector_type
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            self.proj = nn.Sequential(*modules)

        if "mm_use_row_seperator" not in config or config.mm_use_row_seperator:
            # define a row seperator
            self.row_seperator = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        if 'mm_use_im_start_end' in config and config.mm_use_im_start_end:
            self.img_start_seperator = nn.Parameter(torch.zeros(1, config.hidden_size))
            self.img_end_seperator = nn.Parameter(torch.zeros(1, config.hidden_size))                        

    def forward(self, x):
        return self.proj(x)

@MULTIMODAL_REGISTRY.register_processor(MagmaMultiModalProcessor,
                                        info=MagmaProcessingInfo,
                                        dummy_inputs=MagmaDummyInputsBuilder)
class MagmaForCausalLM(nn.Module, SupportsMultiModal,
                                        SupportsPP):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        self.vision_tower = MagmaVisionModel(config.vision_config, require_pretrained=False)
        config.vision_config.mm_hidden_size = config.vision_config.mm_hidden_size \
            if 'mm_hidden_size' in config.vision_config else self.vision_tower.hidden_size
        config.vision_config.hidden_size = config.vision_config.hidden_size \
            if 'hidden_size' in config.vision_config else self.config.text_config.hidden_size
        self.multi_modal_projector = MagmaMultiModalProjector(config.vision_config)

        if hasattr(config.text_config, 'auto_map'):
            del config.text_config.auto_map


        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def _validate_image_sizes(self, data: torch.Tensor) -> torch.Tensor:
        expected_dims = (2, )

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)

            if actual_dims != expected_dims:
                expected_expr = str(expected_dims)
                raise ValueError(
                    f"The expected shape of image sizes per image per batch "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _validate_pixel_values(
        self, data: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:

        h = w = self.config.vision_config.img_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape[1:])

            if actual_dims != expected_dims:
                expected_expr = ("num_patches", *map(str, expected_dims))
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}.")

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[MagmaImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            if not isinstance(image_sizes, (torch.Tensor, list)):
                raise ValueError("Incorrect type of image sizes. "
                                 f"Got type: {type(image_sizes)}")

            return MagmaImagePixelInputs(
                type="pixel_values",
                pixel_values=self._validate_pixel_values(
                    flatten_bn(pixel_values)),
                image_sizes=self._validate_image_sizes(
                    flatten_bn(image_sizes, concat=True)),
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeds. "
                                 f"Got type: {type(image_embeds)}")

            return MagmaImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds),
            )

        raise AssertionError("This line should be unreachable.")

    def _process_image_pixels(
        self,
        inputs: MagmaImagePixelInputs,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        assert self.vision_tower is not None

        image_sizes = inputs["image_sizes"].unsqueeze(1)
        if isinstance(inputs["pixel_values"], torch.Tensor):
            pixel_values = inputs["pixel_values"]
        else:
            pixel_values = torch.nn.utils.rnn.pad_sequence(inputs["pixel_values"], batch_first=True)  # padding the first dimension
                
        target_device = self.multi_modal_projector.proj[0].weight.device
        target_dtype = self.multi_modal_projector.proj[0].weight.dtype

        # convert pixel_values to target data type on the same device
        pixel_values = pixel_values.to(dtype=target_dtype)

        image_num_patches = [(imsize[imsize.sum(1) > 0,0] * imsize[imsize.sum(1) > 0,1]).tolist() for imsize in image_sizes]   

        # figure out if pixel_values is concatenated or stacked
        if pixel_values.dim() == 5:
            # stacking when input is (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [
                pix_val[:sum(num_patch)].split(num_patch, dim=0) for pix_val, num_patch in zip(pixel_values, image_num_patches)
            ]
            _image_sizes_list = [image_size[image_size.sum(-1) > 0].tolist() for image_size in image_sizes]
        else:
            # otherwise has to be stacked from list of (num_patches, num_channels, height, width)
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be 5 dimensions")

        # calculate number of crops for each instance in the batch given _image_sizes_list
        _image_sizes_list_temp = sum(_image_sizes_list, [])
        # concate nate all images in _pixel_values_list
        _pixel_values_list_temp = sum(_pixel_values_list, ())
        _pixel_values_list_temp = torch.cat(_pixel_values_list_temp, dim=0)
        image_features = self.vision_tower(_pixel_values_list_temp).permute(0, 2, 3, 1)
        image_features = self.multi_modal_projector(image_features)

        num_crops_list = [_image_size[0]*_image_size[1] for _image_size in _image_sizes_list_temp]
        image_features_split = torch.split(image_features, num_crops_list, dim=0)
        flattened_image_features = []
        for image_feature, image_size in zip(image_features_split, _image_sizes_list_temp):
            image_feature = image_feature.view(image_size[0], image_size[1], *image_feature.shape[1:])
            image_feature = image_feature.permute(0, 2, 1, 3, 4).flatten(2, 3).flatten(0, 1)
            if "mm_use_row_seperator" not in self.config.vision_config or self.config.vision_config.mm_use_row_seperator:
                image_feature = torch.cat((image_feature, self.multi_modal_projector.row_seperator.repeat(image_feature.shape[0],1,1)), dim=1)
            flattened_image_features.append(image_feature.flatten(0, 1))
        return flattened_image_features

    def _process_image_input(
        self,
        image_input: MagmaImageInputs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if image_input["type"] == "image_embeds":
            return [image_input["data"]]
        return self._process_image_pixels(image_input)

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        vision_features = self._process_image_input(image_input)
        return vision_features

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                self.config.image_token_id,
            )
        return inputs_embeds.squeeze(0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, MagmaForCausalLM

        >>> model = MagmaForCausalLM.from_pretrained("microsoft/magma-8b")
        >>> processor = AutoProcessor.from_pretrained("microsoft/magma-8b")

        >>> convs = [
        >>>     {"role": "system", "content": "You are agent that can see, talk and act."},            
        >>>     {"role": "user", "content": "<image_start><image><image_end>\nWhat is the letter on the robot?"},
        >>> ]
        >>> url = "https://microsoft.github.io/Magma/static/images/logo.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> prompt = processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(images=[image], texts=prompt, return_tensors="pt")
        >>> inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
        >>> inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)     

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The letter on the robot is \"M\"."
        ```"""

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility
        elif inputs_embeds is None:        
            # 1. Extract the input embeddings
            vision_features = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, vision_features)
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)

        return hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
