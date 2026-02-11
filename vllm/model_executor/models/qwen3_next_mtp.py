# # SPDX-License-Identifier: Apache-2.0
# # SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# """Inference-only Qwen3Next MTP model."""
# from collections.abc import Iterable
# from typing import Optional

# import torch
# from torch import nn
# from torch._dynamo.convert_frame import output_codes

# from vllm.compilation.decorators import support_torch_compile
# from vllm.config import VllmConfig
# from vllm.distributed.parallel_state import get_pp_group
# from vllm.logger import init_logger
# from vllm.model_executor.layers.fused_moe import FusedMoE
# from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
# from vllm.model_executor.layers.logits_processor import LogitsProcessor
# from vllm.model_executor.layers.vocab_parallel_embedding import (
#     DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
# from vllm.model_executor.model_loader.weight_utils import default_weight_loader
# from vllm.model_executor.models.qwen3_next import (Qwen3NextDecoderLayer,
#                                                    Qwen3NextRMSNorm)
# from vllm.sequence import IntermediateTensors
# from vllm.transformers_utils.configs import Qwen3NextConfig

# from .interfaces import SupportsPP
# from .utils import (AutoWeightsLoader, is_pp_missing_parameter,
#                     make_empty_intermediate_tensors_factory, maybe_prefix)

# logger = init_logger(__name__)

# KVCache = tuple[torch.Tensor, torch.Tensor]



# @support_torch_compile
# class Qwen3NextMultiTokenPredictor(nn.Module):

#     def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
#         super().__init__()

#         model_config = vllm_config.model_config
#         quant_config = vllm_config.quant_config
#         lora_config = vllm_config.lora_config
#         config: Qwen3NextConfig = model_config.hf_config

#         self.config = config
#         lora_vocab = ((lora_config.lora_extra_vocab_size *
#                        (lora_config.max_loras or 1)) if lora_config else 0)
#         self.vocab_size = config.vocab_size + lora_vocab
#         self.org_vocab_size = config.vocab_size

#         self.mtp_start_layer_idx = config.num_hidden_layers
#         self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)
#         if self.num_mtp_layers == 1 and hasattr(config, "layer_types") and len(config.layer_types) > config.num_hidden_layers:
#             self.num_mtp_layers = len(config.layer_types) - config.num_hidden_layers

#         # FIXME: num_mtp_layers should be set to the number of mtp layers in the config
#         self.num_mtp_layers = 4
#         self.embed_tokens = VocabParallelEmbedding(
#             self.vocab_size,
#             config.hidden_size,
#             org_num_embeddings=config.vocab_size,
#         )

#         self.fc = ColumnParallelLinear(self.config.hidden_size * 2,
#                                        self.config.hidden_size,
#                                        gather_output=True,
#                                        bias=False,
#                                        return_bias=False,
#                                        quant_config=quant_config,
#                                        prefix=f'{prefix}.fc')

#         self.layers = torch.nn.ModuleList(
#             Qwen3NextDecoderLayer(
#                 vllm_config,
#                 layer_type='full_attention',
#                 prefix=f'{prefix}.layers.{idx}',
#             ) for idx in range(self.num_mtp_layers))

#         self.make_empty_intermediate_tensors = (
#             make_empty_intermediate_tensors_factory(
#                 ["hidden_states", "residual"], config.hidden_size))

#         self.norm = Qwen3NextRMSNorm(config.hidden_size,
#                                      eps=config.rms_norm_eps)
#         self.pre_fc_norm_hidden = Qwen3NextRMSNorm(config.hidden_size,
#                                                    eps=config.rms_norm_eps)
#         self.pre_fc_norm_embedding = Qwen3NextRMSNorm(config.hidden_size,
#                                                       eps=config.rms_norm_eps)


#     def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
#         return self.embed_tokens(input_ids)

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         positions: torch.Tensor,
#         hidden_states: torch.Tensor,
#         intermediate_tensors: Optional[IntermediateTensors] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         spec_step_idx: int = 0,
#     ) -> torch.Tensor:
#         if get_pp_group().is_first_rank:
#             if inputs_embeds is None:
#                 inputs_embeds = self.get_input_embeddings(input_ids)
#             assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
#             inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
#             hidden_states = self.pre_fc_norm_hidden(hidden_states)
#             hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
#             hidden_states = self.fc(hidden_states)
#             residual = None
#         else:
#             assert intermediate_tensors is not None
#             hidden_states = intermediate_tensors["hidden_states"]
#             residual = intermediate_tensors["residual"]

#         current_step_idx = (spec_step_idx % self.num_mtp_layers)
#         hidden_states, residual = self.layers[current_step_idx](
#             positions=positions,
#             hidden_states=hidden_states,
#             residual=residual,
#         )

#         if not get_pp_group().is_last_rank:
#             return IntermediateTensors({
#                 "hidden_states": hidden_states,
#                 "residual": residual
#             })

#         hidden_states, _ = self.norm(hidden_states, residual)
#         return hidden_states

#     def load_weights(self, weights: Iterable[tuple[str,
#                                                    torch.Tensor]]) -> set[str]:
#         stacked_params_mapping = [
#             # (param_name, shard_name, shard_id)
#             ("qkv_proj", "q_proj", "q"),
#             ("qkv_proj", "k_proj", "k"),
#             ("qkv_proj", "v_proj", "v"),
#             ("gate_up_proj", "gate_proj", 0),
#             ("gate_up_proj", "up_proj", 1),
#         ]

#         # Params for weights, fp8 weight scales, fp8 activation scales
#         # (param_name, weight_name, expert_id, shard_id)
#         expert_params_mapping = FusedMoE.make_expert_params_mapping(
#             ckpt_gate_proj_name="gate_proj",
#             ckpt_down_proj_name="down_proj",
#             ckpt_up_proj_name="up_proj",
#             num_experts=self.config.num_experts)

#         params_dict = dict(self.named_parameters())
#         loaded_params: set[str] = set()
#         for name, loaded_weight in weights:
#             if "rotary_emb.inv_freq" in name:
#                 continue

#             for param_name, weight_name, shard_id in stacked_params_mapping:
#                 if weight_name not in name:
#                     continue

#                 if "mlp.experts" in name:
#                     continue

#                 name = name.replace(weight_name, param_name)
#                 # Skip loading extra bias for GPTQ models.
#                 if name.endswith(".bias") and name not in params_dict:
#                     continue
#                 # Skip layers on other devices.
#                 if is_pp_missing_parameter(name, self):
#                     continue
#                 if name not in params_dict:
#                     continue
#                 param = params_dict[name]
#                 weight_loader = param.weight_loader
#                 weight_loader(param, loaded_weight, shard_id)
#                 break
#             else:
#                 for mapping in expert_params_mapping:
#                     param_name, weight_name, expert_id, shard_id = mapping
#                     if weight_name not in name:
#                         continue
#                     name = name.replace(weight_name, param_name)
#                     # Skip layers on other devices.
#                     if is_pp_missing_parameter(name, self):
#                         continue
#                     # Skip loading extra bias for GPTQ models.
#                     if ((name.endswith(".bias") or name.endswith("_bias"))
#                             and name not in params_dict):
#                         continue
#                     param = params_dict[name]
#                     weight_loader = param.weight_loader
#                     weight_loader(param,
#                                   loaded_weight,
#                                   name,
#                                   shard_id=shard_id,
#                                   expert_id=expert_id)
#                     break
#                 else:
#                     # Skip loading extra bias for GPTQ models.
#                     if name.endswith(".bias") and name not in params_dict:
#                         continue
#                     if is_pp_missing_parameter(name, self):
#                         continue

#                     param = params_dict[name]
#                     weight_loader = getattr(param, "weight_loader",
#                                             default_weight_loader)
#                     weight_loader(param, loaded_weight)
#             loaded_params.add(name)
#         return loaded_params


# @support_torch_compile
# class Qwen3NextMTP(nn.Module, SupportsPP):
#     packed_modules_mapping = {
#         "qkv_proj": [
#             "q_proj",
#             "k_proj",
#             "v_proj",
#         ],
#         "gate_up_proj": ["up_proj", "down_proj"]
#     }

#     def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
#         config = vllm_config.model_config.hf_config
#         self.vllm_config = vllm_config
#         cache_config = vllm_config.cache_config
#         assert not cache_config.enable_prefix_caching, \
#             "Qwen3NextMTP currently does not support prefix caching"

#         self.quant_config = vllm_config.quant_config

#         super().__init__()
#         self.config = config
#         self.model = Qwen3NextMultiTokenPredictor(vllm_config=vllm_config,
#                                                   prefix=maybe_prefix(
#                                                       prefix, "mtp"))
#         self.unpadded_vocab_size = config.vocab_size
#         self.lm_head = ParallelLMHead(self.unpadded_vocab_size,
#                                       config.hidden_size,
#                                       org_num_embeddings=config.vocab_size,
#                                       padding_size=DEFAULT_VOCAB_PADDING_SIZE,
#                                       prefix=maybe_prefix(prefix, "lm_head"))
#         self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
#                                                 config.vocab_size)
#         self.make_empty_intermediate_tensors = (
#             self.model.make_empty_intermediate_tensors)

#     def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
#         return self.model.get_input_embeddings(input_ids)

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         positions: torch.Tensor,
#         hidden_states: torch.Tensor,
#         intermediate_tensors: Optional[IntermediateTensors] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         **kwargs: object,
#     ):
#         hidden_states = self.model(input_ids, positions, hidden_states,
#                                    intermediate_tensors, inputs_embeds)
#         return hidden_states

#     def compute_logits(
#         self,
#         hidden_states: torch.Tensor,
#         spec_step_idx: int = 0,
#     ) -> Optional[torch.Tensor]:
#         return self.logits_processor(self.lm_head, hidden_states)

#     def load_weights(self, weights: Iterable[tuple[str,
#                                                    torch.Tensor]]) -> set[str]:
#         shared_weight_names = ["embed_tokens", "lm_head"]

#         def remap_weight_names(weights):
#             for name, weight in weights:
#                 if name.startswith("mtp."):
#                     name = name.replace("mtp.", "model.")
#                 elif not any(key in name for key in shared_weight_names):
#                     continue
#                 yield name, weight

#         loader = AutoWeightsLoader(self)
#         return loader.load_weights(remap_weight_names(weights))
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3Next MTP model."""
from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen3_next import (Qwen3NextDecoderLayer,
                                                   Qwen3NextRMSNorm)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import Qwen3NextConfig

from .interfaces import SupportsPP
from .utils import (AutoWeightsLoader, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, maybe_prefix)
from .adapter import AutoMTPStopHeadMid

logger = init_logger(__name__)


class TolerantAutoWeightsLoader(AutoWeightsLoader):
    """
    A tolerant version of AutoWeightsLoader that skips unmatched weights
    instead of raising ValueError.
    """
    
    def _load_module(
        self,
        base_prefix: str,
        module: nn.Module,
        weights: Iterable[tuple[str, torch.Tensor]],
    ):
        """Override to skip unmatched weights instead of raising ValueError."""
        from vllm.model_executor.models.utils import PPMissingLayer
        
        if isinstance(module, PPMissingLayer):
            return

        # Avoid infinite recursion since this function is typically
        # called inside load_weights of the module itself
        if module != self.module:
            module_load_weights = getattr(module, "load_weights", None)
            if callable(module_load_weights):
                loaded_params = module_load_weights(weights)
                if loaded_params is None:
                    logger.warning(
                        "Unable to collect loaded parameters "
                        "for module %s", module)
                else:
                    yield from map(
                        lambda x: self._get_qualname(base_prefix, x),
                        loaded_params,
                    )
                    return

        child_modules = dict(module.named_children())
        child_params = dict(module.named_parameters(recurse=False))

        # Add missing tensors the weight loader needs to be able to load
        # that aren't registered as params, e.g., batchnorm statistics.
        self._add_loadable_non_param_tensors(module, child_params)

        for child_prefix, child_weights in self._groupby_prefix(weights):
            prefix = self._get_qualname(base_prefix, child_prefix)

            if child_prefix in child_modules:
                if self._can_skip(prefix + "."):
                    logger.debug("Skipping module %s", prefix)
                    continue

                yield from self._load_module(prefix,
                                             child_modules[child_prefix],
                                             child_weights)
            elif child_prefix in child_params:
                if self._can_skip(prefix):
                    logger.debug("Skipping param %s", prefix)
                    continue

                yield from self._load_param(prefix, child_params[child_prefix],
                                            child_weights)
            else:
                can_skip_module = self._can_skip(prefix + ".")
                can_skip_param = self._can_skip(prefix)
                if can_skip_module or can_skip_param:
                    logger.debug("Skipping missing %s", prefix)
                    continue

                can_ignore_module = self._can_ignore_unexpected(prefix + ".")
                can_ignore_param = self._can_ignore_unexpected(prefix)
                if can_ignore_module or can_ignore_param:
                    logger.debug("Ignoring missing %s", prefix)
                    continue

                # NOTE: Instead of raising ValueError, log warning and skip
                logger.debug(
                    f"Skipping unmatched weight '{prefix}' "
                    f"(not found in {type(self.module).__name__})"
                )
                continue

KVCache = tuple[torch.Tensor, torch.Tensor]


@support_torch_compile
class Qwen3NextMultiTokenPredictor(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        model_config = vllm_config.model_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        config: Qwen3NextConfig = model_config.hf_config

        self.config = config
        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)

        # FIXME: model_configs to set num_nextn_predict_layers
        self.num_mtp_layers = 1
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        self.fc = ColumnParallelLinear(self.config.hidden_size * 2,
                                       self.config.hidden_size,
                                       gather_output=True,
                                       bias=False,
                                       return_bias=False,
                                       quant_config=quant_config,
                                       prefix=f'{prefix}.fc')

        self.layers = torch.nn.ModuleList(
            Qwen3NextDecoderLayer(
                vllm_config,
                layer_type="full_attention",
                prefix=f'{prefix}.layers.{idx}',
            ) for idx in range(self.num_mtp_layers))

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        self.norm = Qwen3NextRMSNorm(config.hidden_size,
                                     eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3NextRMSNorm(config.hidden_size,
                                                   eps=config.rms_norm_eps)
        self.pre_fc_norm_embedding = Qwen3NextRMSNorm(config.hidden_size,
                                                      eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings(input_ids)
            assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
            hidden_states = self.pre_fc_norm_hidden(hidden_states)
            hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
            hidden_states = self.fc(hidden_states)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        current_step_idx = (spec_step_idx % self.num_mtp_layers)
        hidden_states, residual = self.layers[current_step_idx](
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
                        continue
                    # Skip missing parameters (tolerant loading)
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip missing parameters (tolerant loading)
                    if name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

class MTPStopHead(nn.Module):
    def __init__(self, hidden_size, intermediate_size=None, num_classes=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size if intermediate_size else self.hidden_size // 2
        self.num_classes = num_classes
        # SwiGLU MLP
        self.gate_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # Classification head: hidden_size -> num_classes
        self.output_proj = torch.nn.Linear(self.hidden_size, self.num_classes, bias=False)
        self.act_fn = torch.nn.functional.silu

    def forward(self, x):
        # SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        hidden = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        # Classification output: [SeqLen, Batch, num_classes]
        logits = self.output_proj(hidden)
        return logits 

@support_torch_compile
class Qwen3NextMTP(nn.Module, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["up_proj", "down_proj"]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        cache_config = vllm_config.cache_config
        assert not cache_config.enable_prefix_caching, \
            "Qwen3NextMTP currently does not support prefix caching"

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen3NextMultiTokenPredictor(vllm_config=vllm_config,
                                                  prefix=maybe_prefix(
                                                      prefix, "mtp"))
        self.unpadded_vocab_size = config.vocab_size
        self.lm_head = ParallelLMHead(self.unpadded_vocab_size,
                                      config.hidden_size,
                                      org_num_embeddings=config.vocab_size,
                                      padding_size=DEFAULT_VOCAB_PADDING_SIZE,
                                      prefix=maybe_prefix(prefix, "lm_head"))
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)

        # NOTE: AutoMTP
        self.adapter = None
        # self.adapter = None
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)


    # @torch.inference_mode()  
    def should_stop(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.adapter is None:
            if hidden_states.dim() == 1:
                return torch.tensor(False, device=hidden_states.device)
            return torch.zeros(hidden_states.shape[0], dtype=torch.bool, device=hidden_states.device)

        # Compute entropy using torch.special.entr (更高效的 CUDA 实现)
        # entr(p) = -p * log(p)，所以 entropy = sum(entr(p))
        probs = torch.nn.functional.softmax(logits.float(), dim=-1)
        entropy = torch.special.entr(probs).sum(dim=-1, keepdim=True)  # [num_tokens, 1]
        entropy = entropy.to(hidden_states.dtype)
        
        # [num_tokens, hidden_size] + [num_tokens, 1] → [num_tokens, hidden_size + 1]
        hidden_states = torch.cat([hidden_states, entropy], dim=-1)

        stop_logits = self.adapter(hidden_states)  # [num_tokens, 1] 或 [1]
        stop_prob = torch.sigmoid(stop_logits).squeeze(-1)
        
        # 返回布尔 tensor: [num_tokens] 或标量
        return True if stop_prob >= 0.5 else False
        # return True

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)


    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ):
        hidden_states = self.model(input_ids, positions, hidden_states,
                                   intermediate_tensors, inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> Optional[torch.Tensor]:
        return self.logits_processor(self.lm_head, hidden_states)

    # def load_weights(self, weights: Iterable[tuple[str,
    #                                                torch.Tensor]]) -> set[str]:
    #     shared_weight_names = ["embed_tokens", "lm_head"]

    #     def remap_weight_names(weights):
    #         for name, weight in weights:
    #             if name.startswith("mtp."):
    #                 name = name.replace("mtp.", "model.")
    #             elif not any(key in name for key in shared_weight_names):
    #                 continue
    #             yield name, weight

    #     # NOTE: Use TolerantAutoWeightsLoader to skip unmatched weights
    #     # instead of raising ValueError
    #     loader = TolerantAutoWeightsLoader(self)
    #     return loader.load_weights(remap_weight_names(weights))


    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        # Convert to list to allow multiple iterations
        weights_list = list(weights)
        
        shared_weight_names = ["embed_tokens", "lm_head"]
        
        # Check if adapter weights exist in checkpoint
        has_adapter_weights = any(name.startswith("mtp.adapter.") 
                                 for name, _ in weights_list)
        
        # Create adapter module if weights exist
        if has_adapter_weights:
            self.adapter = AutoMTPStopHeadMid(self.config.hidden_size)
            # Move to the same device as the model (get device from lm_head)
            device = next(self.lm_head.parameters()).device
            self.adapter = self.adapter.to(device)

        def remap_weight_names(weights):
            for name, weight in weights:
                # Handle adapter weights: mtp.adapter.xxx -> adapter.xxx
                if name.startswith("mtp.adapter."):
                    # Remove "mtp." prefix to get "adapter.xxx"
                    name = name.replace("mtp.adapter.", "adapter.")
                    yield name, weight
                # Remap mtp. prefix to model. (for MTP layers)
                elif name.startswith("mtp."):
                    name = name.replace("mtp.", "model.")
                    yield name, weight
                # Only include shared weights (embed_tokens, lm_head)
                elif any(key in name for key in shared_weight_names):
                    yield name, weight
                # Skip other weights that don't match

        # NOTE: Use TolerantAutoWeightsLoader to skip unmatched weights
        # instead of raising ValueError
        loader = TolerantAutoWeightsLoader(self)
        return loader.load_weights(remap_weight_names(weights_list))