"""
openvla.py

This module defines the OpenVLA model for VLLM inference.
Ref: https://github.com/vllm-project/vllm/blob/v0.6.6.post1/vllm/model_executor/models/llava.py
"""

from typing import (Iterable, List, Literal, Mapping, Optional, Protocol, Set,
                    Tuple, TypedDict, Union, Any, Callable)

import torch
import torch.nn as nn
from transformers import (BatchFeature, ProcessorMixin)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig, PrismaticConfig, VISION_BACKBONE_TO_TIMM_ID
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.inputs import InputContext
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import NestedTensors
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        MultiModalDataItems, ProcessorInputs,
                                        PromptReplacement)
from vllm.sequence import IntermediateTensors
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    flatten_bn,
    init_vllm_registered_model,
    maybe_prefix,
    merge_multimodal_embeddings,
)
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.siglip import (
    dummy_image_for_siglip,
    dummy_seq_data_for_siglip,
    get_max_siglip_image_tokens,
)
from functools import partial
from timm.models.vision_transformer import LayerScale


# === Data Structures for Image Inputs ===
class OpenVLAImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """
    Shape: (batch_size * num_images, num_channels, height, width)
    """

class OpenVLAImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """
    Shape: (batch_size * num_images, image_feature_size, hidden_size)
    """

OpenVLAImageInputs = Union[OpenVLAImagePixelInputs, OpenVLAImageEmbeddingInputs]

# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result
    return wrapper

# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor

def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


# === MultiModal Projector for OpenVLA ===
class OpenVLAMultiModalProjector(nn.Module):
    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        use_fused_vision_backbone: bool,
        projector_hidden_act: str = "gelu",
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        
        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = ColumnParallelLinear(
                vision_hidden_size,
                text_hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc1" if prefix else "fc1",
            )
            self.fc2 = RowParallelLinear(
                text_hidden_size,
                text_hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc2" if prefix else "fc2",
            )
            self.act_fn1 = get_act_fn(projector_hidden_act)
        else:
            initial_projection_dim = 4 * vision_hidden_size
            self.fc1 = ColumnParallelLinear(
                vision_hidden_size,
                initial_projection_dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc1" if prefix else "fc1",
            )
            self.fc2 = ColumnParallelLinear(
                initial_projection_dim,
                text_hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc2" if prefix else "fc2",
            )
            self.fc3 = RowParallelLinear(
                text_hidden_size,
                text_hidden_size,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc3" if prefix else "fc3",
            )
            self.act_fn1 = get_act_fn(projector_hidden_act)
            self.act_fn2 = get_act_fn(projector_hidden_act)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            hidden_states, _ = self.fc1(image_features)
            hidden_states = self.act_fn1(hidden_states)
            hidden_states, _ = self.fc2(hidden_states)
        else:
            hidden_states, _ = self.fc1(image_features)
            hidden_states = self.act_fn1(hidden_states)
            hidden_states, _ = self.fc2(hidden_states)
            hidden_states = self.act_fn2(hidden_states)
            hidden_states, _ = self.fc3(hidden_states)
        return hidden_states


# === Fused Vision Tower for Dinosiglip ===
class FusedVisionTower(nn.Module):
    def __init__(self, hf_config: OpenVLAConfig, num_hidden_layers_override: Optional[int] = None):
        super().__init__()
        import timm
        self.use_fused = True
        # Primary featurizer
        self.featurizer = timm.create_model(
            hf_config.timm_model_ids[0],
            pretrained=False,
            num_classes=0,
            img_size=hf_config.image_sizes[0],
            act_layer=hf_config.timm_override_act_layers[0],
        )
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks)-2})
        )
        self.embed_dim = self.featurizer.embed_dim

        # Fused featurizer
        self.fused_featurizer = timm.create_model(
            hf_config.timm_model_ids[1],
            pretrained=False,
            num_classes=0,
            img_size=hf_config.image_sizes[1],
            act_layer=hf_config.timm_override_act_layers[1],
        )
        self.fused_featurizer.forward = unpack_tuple(
            partial(self.fused_featurizer.get_intermediate_layers, n={len(self.fused_featurizer.blocks)-2})
        )
        self.embed_dim += self.fused_featurizer.embed_dim

        # Patch LayerScale modules to be HF-compatible
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        for module in self.fused_featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Expecting pixel_values shape: (batch_size, 6, H, W) where channels are stacked (3 for primary, 3 for fused)
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        patches = self.featurizer(img)
        patches_fused = self.fused_featurizer(img_fused)
        return torch.cat([patches, patches_fused], dim=2)


def _init_vision_tower(hf_config: OpenVLAConfig, quant_config: Optional[QuantizationConfig] = None):
    backbone_id = hf_config.vision_backbone_id
    vision_feature_layer = getattr(hf_config, "vision_feature_layer", None)
    if vision_feature_layer is not None:
        if vision_feature_layer < 0:
            num_hidden_layers = hf_config.num_hidden_layers + vision_feature_layer + 1
        else:
            num_hidden_layers = vision_feature_layer + 1
    else:
        num_hidden_layers = None

    if hf_config.use_fused_vision_backbone:
        if backbone_id.startswith("dinosiglip"):
            return FusedVisionTower(hf_config, num_hidden_layers_override=num_hidden_layers)
        else:
            raise NotImplementedError(f"Fused vision backbone not supported for {backbone_id}; only dinosiglip is supported.")
    else:
        raise NotImplementedError(f"Unsupported vision backbone: {backbone_id}; only dinosiglip is supported.")


# === Utility Functions (only support dinosiglip) ===
def get_max_openvla_image_tokens(ctx: InputContext) -> int:
    hf_config = ctx.model_config.hf_config
    backbone_id = hf_config.vision_backbone_id
    if backbone_id.startswith("dinosiglip"):
        timm_model_ids = VISION_BACKBONE_TO_TIMM_ID[backbone_id]    # e.g., ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224"]
        hf_config.image_size = hf_config.image_sizes[0]
        hf_config.patch_size = int(timm_model_ids[0].split("patch")[1].split("_")[0])   # HACK: get patch_size from timm_model_ids
        num_image_tokens = get_max_siglip_image_tokens(hf_config)
    else:
        raise NotImplementedError(f"Unsupported vision backbone: {backbone_id}; only dinosiglip is supported.")
    return num_image_tokens


# === MultiModal Processor for OpenVLA ===
class OpenVLAMultiModalProcessor(BaseMultiModalProcessor):
    """
    A multi-modal processor for OpenVLA.
    This class handles the processing of image inputs for OpenVLA models.
    """

    def _get_hf_processor(self) -> ProcessorMixin:
        processor = PrismaticProcessor.from_pretrained(self.ctx.model_config.tokenizer, trust_remote_code=True)
        assert isinstance(processor, ProcessorMixin)
        return processor

    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_inputs: BatchFeature,
        mm_processor_kwargs: Mapping[str, object],
    ) -> list[PromptReplacement]:
        """Replacement for image tokens"""
        hf_config = self.ctx.model_config.hf_config
        image_token_id = hf_config.pad_token_id
        max_image_tokens = get_max_openvla_image_tokens(self.ctx)
        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=[image_token_id] * max_image_tokens,
            )
        ]

    def _get_dummy_mm_inputs(
        self,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        """Placeholder for dummy image inputs"""
        hf_config = self.ctx.model_config.hf_config
        hf_config.image_size = hf_config.image_sizes[0]
        num_images = mm_counts["image"]
        if hf_config.vision_backbone_id.startswith("dinosiglip"):
            data = dummy_image_for_siglip(hf_config, num_images)
        else:
            msg = f"Unsupported vision backbone: {hf_config.vision_backbone_id}; only dinosiglip is supported."
            raise NotImplementedError(msg)
        image_token = "<PAD>"
        return ProcessorInputs(
            prompt_text=image_token * num_images,
            mm_data=data,
            mm_processor_kwargs={},
        )

# === Main Model Class ===
# @MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_openvla_image_tokens)
@MULTIMODAL_REGISTRY.register_processor(OpenVLAMultiModalProcessor)
class OpenVLAForActionPrediction(nn.Module, SupportsMultiModal):
    """
    OpenVLA model for VLLM inference.
    """
    # BitandBytes specific attributes
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        # Extract configuration from vllm_config
        config = vllm_config.model_config.hf_config  # type: OpenVLAConfig
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        # Instantiate vision tower (only dinosiglip is supported)
        self.vision_backbone = _init_vision_tower(config, quant_config=quant_config)

        # Use the hidden size from the vision tower for projection.
        self.projector = OpenVLAMultiModalProjector(
            vision_hidden_size=self.vision_backbone.embed_dim,
            text_hidden_size=config.text_config.hidden_size,
            use_fused_vision_backbone=config.use_fused_vision_backbone,
            projector_hidden_act="gelu",
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "projector"),
        )

        # Initialize the language model with vllm_config
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config.text_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["LlamaForCausalLM"], # HACK: openvla do not have architectures key in text_config
        )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
        h = w = self.config.image_sizes[0]
        expected_dims = (3, h, w) if not self.config.use_fused_vision_backbone else (6, h, w)
        actual_dims = tuple(data.shape[1:])
        if actual_dims != expected_dims:
            expected_expr = ("batch_size", *map(str, expected_dims))
            raise ValueError(
                f"The expected shape of pixel values is {expected_expr}. You supplied {tuple(data.shape)}."
            )
        return data

    def _parse_and_validate_image_input(self, **kwargs: object) -> Optional[dict]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError(f"Incorrect type of pixel values. Got type: {type(pixel_values)}")
            return {"type": "pixel_values", "data": self._validate_pixel_values(flatten_bn(pixel_values, concat=True))}
        if image_embeds is not None:
            if not isinstance(image_embeds, (torch.Tensor, list)):
                raise ValueError(f"Incorrect type of image embeddings. Got type: {type(image_embeds)}")
            return {"type": "image_embeds", "data": flatten_bn(image_embeds, concat=True)}
        raise AssertionError("This line should be unreachable.")

    def _image_pixels_to_features(self, vision_tower: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(torch.bfloat16)  # HACK: openvla supports bfloat16 originally
        image_features = vision_tower(pixel_values)
        return image_features

    def _process_image_pixels(self, inputs: dict) -> torch.Tensor:
        assert self.vision_backbone is not None
        pixel_values = inputs["data"]
        return self._image_pixels_to_features(self.vision_backbone, pixel_values)

    def _process_image_input(self, image_input: dict) -> torch.Tensor:
        if image_input["type"] == "image_embeds":
            return image_input["data"]
        assert self.vision_backbone is not None
        image_features = self._process_image_pixels(image_input)
        return self.projector(image_features)

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        return self._process_image_input(image_input)

    def get_input_embeddings(
        self, input_ids: torch.Tensor, multimodal_embeddings: Optional[NestedTensors] = None
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings, self.config.pad_token_id
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> SamplerOutput:
        """
        Rewrite the forward() method of your model to remove any unnecessary code, such as training-specific code. 
        Modify the input parameters to treat input_ids and positions as flattened tensors with a single batch size dimension, 
        without a max-sequence length dimension.
        """
        if intermediate_tensors is not None:
            inputs_embeds = None
        elif inputs_embeds is None:
            multimodal_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, multimodal_embeddings)
            input_ids = None

        hidden_states = self.language_model.model(
            input_ids, positions, kv_caches, attn_metadata, intermediate_tensors, inputs_embeds=inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states, sampling_metadata)

    def sample(
        self, logits: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
