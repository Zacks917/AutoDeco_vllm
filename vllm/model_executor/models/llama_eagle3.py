# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import Optional
from pydantic_core.core_schema import NoneSchema
import atexit
import os
import torch
import torch.nn as nn
from transformers import LlamaConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.llama import (LlamaDecoderLayer,
                                              LlamaForCausalLM)

from .utils import AutoWeightsLoader, maybe_prefix

from .adapter import AutoMTPStopHeadV4, AutoMTPStopHeadMid, AutoMTPStopHeadSimple, AutoMTPStopHeadDeep
logger = init_logger(__name__)


class LlamaDecoderLayer(LlamaDecoderLayer):

    def __init__(self,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 config: Optional[LlamaConfig] = None) -> None:
        super().__init__(vllm_config, prefix=prefix, config=config)

        config = config or vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        # override qkv
        self.self_attn.qkv_proj = QKVParallelLinear(
            2 * self.hidden_size,
            self.self_attn.head_dim,
            self.self_attn.total_num_heads,
            self.self_attn.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "qkv_proj"),
        )

        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if getattr(config, "norm_before_residual", False):
            self._residual_norm = self._norm_before_residual
        else:
            self._residual_norm = self._norm_after_residual

    def _norm_before_residual(
            self,
            hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.hidden_norm(hidden_states)
        residual = hidden_states
        return hidden_states, residual

    def _norm_after_residual(
            self,
            hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        return hidden_states, residual

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:

        embeds = self.input_layernorm(embeds)

        hidden_states, residual = self._residual_norm(
            hidden_states=hidden_states)

        hidden_states = torch.cat([embeds, hidden_states], dim=-1)
        
        # Self Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)

        # Fully Connected
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@support_torch_compile
class LlamaModel(nn.Module):

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config. \
            speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size

        current_vllm_config = get_current_vllm_config()

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                current_vllm_config,
                prefix=maybe_prefix(prefix, f"layers.{start_layer_id}"),
                config=self.config,
            )
        ])
        if hasattr(self.config, "target_hidden_size"):
            self.fc = torch.nn.Linear(self.config.target_hidden_size * 3,
                                      self.config.hidden_size,
                                      bias=False)
        else:
            self.fc = torch.nn.Linear(self.config.hidden_size * 3,
                                      self.config.hidden_size,
                                      bias=False)
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_embeds = self.embed_tokens(input_ids)
        assert hidden_states.shape[-1] == input_embeds.shape[-1]

        residual = None
        hidden_states, residual = self.layers[0](
            positions,
            input_embeds,
            hidden_states,
            residual,
        )

        hidden_states, hidden_prenorm = self.norm(hidden_states, residual)
        return hidden_states, hidden_prenorm

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if 'midlayer.' in name:
                name = name.replace('midlayer.', 'layers.0.')
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Eagle3LlamaForCausalLM(LlamaForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config. \
            speculative_config.draft_model_config.hf_config
        # Ensure draft_vocab_size is set
        # default to the base vocab size when absent
        if getattr(self.config, "draft_vocab_size", None) is None:
            base_vocab_size = getattr(self.config, "vocab_size", None)
            self.config.draft_vocab_size = base_vocab_size
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config)

        # Store target layer count in draft config for
        # proper layer_types indexing in draft models
        self.config.target_layer_count = target_layer_num
        self.model = LlamaModel(vllm_config=vllm_config,
                                prefix="model",
                                start_layer_id=target_layer_num)

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            org_num_embeddings=self.config.draft_vocab_size,
            padding_size=(DEFAULT_VOCAB_PADDING_SIZE),
            prefix=maybe_prefix(prefix, "lm_head"))
        self.logits_processor = LogitsProcessor(self.config.draft_vocab_size,
                                                scale=logit_scale)
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
            requires_grad=False,
        )

        self.adapter = AutoMTPStopHeadMid(
            hidden_size=self.config.hidden_size,
        )
        # 先创建原始 adapter，等权重加载后再 compile
        # self.adapter = AutoMTPStopHeadSimple(hidden_size=self.config.hidden_size)
        self._adapter_compiled = False  # 标记是否已 compile
        # self.adapter = AutoMTPStopHeadDeep(hidden_size=self.config.hidden_size)
        self._init_should_stop_profiler()

    def _init_should_stop_profiler(self) -> None:
        enabled = os.getenv("PROFILE_SHOULD_STOP", "0") in ("1", "true", "True")
        self._profile_should_stop_enabled = enabled
        if not enabled:
            return

        self._profile_log_interval = int(
            os.getenv("PROFILE_SHOULD_STOP_LOG_INTERVAL", "200"))
        self._profile_threshold = float(
            os.getenv("PROFILE_SHOULD_STOP_THRESHOLD", "0.5"))

        self._profile_calls = 0
        self._profile_stop_true = 0
        self._profile_ms_total = 0.0
        self._profile_ms_entropy = 0.0
        self._profile_ms_cat = 0.0
        self._profile_ms_adapter = 0.0
        self._profile_ms_post = 0.0

        atexit.register(self._log_should_stop_profile_summary)

    def _log_should_stop_profile_summary(self) -> None:
        if not getattr(self, "_profile_should_stop_enabled", False):
            return
        calls = getattr(self, "_profile_calls", 0)
        if calls == 0:
            logger.info("[should_stop profile] no should_stop calls recorded.")
            return
        stop_rate = self._profile_stop_true / calls
        logger.info(
            "[should_stop profile][summary] calls=%d stop_rate=%.4f "
            "avg_total=%.4fms entropy=%.4fms cat=%.4fms adapter=%.4fms post=%.4fms",
            calls,
            stop_rate,
            self._profile_ms_total / calls,
            self._profile_ms_entropy / calls,
            self._profile_ms_cat / calls,
            self._profile_ms_adapter / calls,
            self._profile_ms_post / calls,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is not None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support multimodal inputs yet."
            )
        return self.model(input_ids, positions, hidden_states)

    # @torch.inference_mode()  
    def should_stop(
        self,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        threshold = getattr(self, "_profile_threshold", 0.5)

        if self.adapter is None:
            if hidden_states.dim() == 1:
                return torch.tensor(False, device=hidden_states.device)
            return torch.zeros(hidden_states.shape[0], dtype=torch.bool, device=hidden_states.device)

        profiling_enabled = (
            getattr(self, "_profile_should_stop_enabled", False)
            and hidden_states.is_cuda
            and logits.is_cuda
        )

        if profiling_enabled:
            ev_total_start = torch.cuda.Event(enable_timing=True)
            ev_entropy_start = torch.cuda.Event(enable_timing=True)
            ev_entropy_end = torch.cuda.Event(enable_timing=True)
            ev_cat_end = torch.cuda.Event(enable_timing=True)
            ev_adapter_end = torch.cuda.Event(enable_timing=True)
            ev_post_end = torch.cuda.Event(enable_timing=True)

            ev_total_start.record()
            ev_entropy_start.record()

        # Compute entropy using torch.special.entr (更高效的 CUDA 实现)
        # entr(p) = -p * log(p)，所以 entropy = sum(entr(p))
        probs = torch.nn.functional.softmax(logits.float(), dim=-1)
        entropy = torch.special.entr(probs).sum(dim=-1, keepdim=True)  # [num_tokens, 1]
        entropy = entropy.to(hidden_states.dtype)
        if profiling_enabled:
            ev_entropy_end.record()

        # [num_tokens, hidden_size] + [num_tokens, 1] → [num_tokens, hidden_size + 1]
        hidden_states = torch.cat([hidden_states, entropy], dim=-1)
        if profiling_enabled:
            ev_cat_end.record()

        stop_logits = self.adapter(hidden_states)  # [num_tokens, 1] 或 [1]
        if profiling_enabled:
            ev_adapter_end.record()

        stop_prob = torch.sigmoid(stop_logits).squeeze(-1)
        stop_mask = stop_prob >= threshold

        if profiling_enabled:
            ev_post_end.record()
            ev_post_end.synchronize()

            self._profile_calls += 1
            self._profile_ms_total += ev_total_start.elapsed_time(ev_post_end)
            self._profile_ms_entropy += ev_entropy_start.elapsed_time(
                ev_entropy_end)
            self._profile_ms_cat += ev_entropy_end.elapsed_time(ev_cat_end)
            self._profile_ms_adapter += ev_cat_end.elapsed_time(ev_adapter_end)
            self._profile_ms_post += ev_adapter_end.elapsed_time(ev_post_end)
            if bool(stop_mask.any().item()):
                self._profile_stop_true += 1

            if self._profile_calls % self._profile_log_interval == 0:
                avg_total = self._profile_ms_total / self._profile_calls
                logger.info(
                    "[should_stop profile] calls=%d avg_total=%.4fms "
                    "entropy=%.4fms cat=%.4fms adapter=%.4fms post=%.4fms "
                    "stop_rate=%.4f",
                    self._profile_calls,
                    avg_total,
                    self._profile_ms_entropy / self._profile_calls,
                    self._profile_ms_cat / self._profile_calls,
                    self._profile_ms_adapter / self._profile_calls,
                    self._profile_ms_post / self._profile_calls,
                    self._profile_stop_true / self._profile_calls,
                )

        return stop_mask

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            assert logits.shape[1] == self.config.vocab_size, \
                "Expected logits to have shape " \
                f"(*, {self.config.vocab_size}), but got {logits.shape}"
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full((
            logits.shape[0],
            self.config.vocab_size,
        ), float('-inf'))
        logits_new[:, targets] = logits
        return logits_new

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # combine multiple auxiliary hidden states returned by eagle3
        return self.model.fc(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        includes_adapter = False
        for name, loaded_weight in weights:
            if "t2d" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            if 'draft_id_to_target_id' in name:
                includes_draft_id_mapping = True
            elif "lm_head" not in name and "adapter" not in name: # NOTE: AUTOMTP
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            if 'adapter' in name:
                includes_adapter = True
            model_weights[name] = loaded_weight
        
        if not includes_adapter:
            self.adapter = None
        skip_substrs = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            skip_substrs=skip_substrs,
        )
        loader.load_weights(model_weights.items())
        
        # 权重加载完成后，compile adapter 以加速推理
        if self.adapter is not None and not self._adapter_compiled:
            self.adapter = torch.compile(
                self.adapter,
                mode="reduce-overhead",  # 减少运行时开销，适合小模型
                fullgraph=True,  # 强制编译整个图，避免 graph breaks
            )
            self._adapter_compiled = True
            
            # 预热编译：预热多个常见 batch size，避免运行时重新编译
            device = next(self.adapter.parameters()).device
            dtype = next(self.adapter.parameters()).dtype
            warmup_batch_sizes = [1, 2, 4, 8]  # 常见的 batch sizes
            with torch.inference_mode():
                for bs in warmup_batch_sizes:
                    dummy_input = torch.randn(
                        bs, self.config.hidden_size + 1, 
                        device=device, dtype=dtype
                    )
                    _ = self.adapter(dummy_input)
            logger.info(f"Adapter compiled and warmed up for batch sizes {warmup_batch_sizes}")
