# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only Mixtral model."""
from typing import List, Optional, Tuple

import os
import numpy as np
import json

import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch import nn
from copy import deepcopy
from transformers import MixtralConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               ReplicatedLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead)
from vllm.model_executor.parallel_utils.communication_op import (
    tensor_model_parallel_all_reduce)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]


class MixtralMLP(nn.Module):

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.ffn_dim = intermediate_size
        self.hidden_dim = hidden_size

        self.w1 = ReplicatedLinear(self.hidden_dim,
                                   self.ffn_dim,
                                   bias=False,
                                   linear_method=linear_method)
        self.w2 = ReplicatedLinear(self.ffn_dim,
                                   self.hidden_dim,
                                   bias=False,
                                   linear_method=linear_method)
        self.w3 = ReplicatedLinear(self.hidden_dim,
                                   self.ffn_dim,
                                   bias=False,
                                   linear_method=linear_method)

        # TODO: Use vllm's SiluAndMul
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        w1_out, _ = self.w1(hidden_states)
        w1_out = self.act_fn(w1_out)
        w3_out, _ = self.w3(hidden_states)
        current_hidden_states = w1_out * w3_out
        current_hidden_states, _ = self.w2(current_hidden_states)
        return current_hidden_states


def all_gather_dict(local_dict):
    dist.barrier()

    # Convert the local_dict to a list of tuples for all_gather_object
    local_list = list(local_dict.items())

    # All gather the list of tuples
    gathered_list = [None for _ in range(dist.get_world_size())]
    torch.distributed.all_gather_object(gathered_list, local_list)

    # Convert the gathered list of tuples back to a dictionary
    def tuple2dict(tuple_items):
        out_dict = {}
        for (key, val) in tuple_items:
            out_dict[key] = val
        return out_dict
    gathered_dict = tuple2dict(deepcopy(gathered_list[0]))
    for i, tuple_items in enumerate(gathered_list):
        if i==0:
            continue
        temp_dict = tuple2dict(tuple_items)
        # gathered_dict.update(temp_dict)
        for key in gathered_dict.keys():
            gathered_dict[key].update(temp_dict[key])

    # Ensure that the gathered dictionary has the same structure on all processes
    dist.barrier()
    return gathered_dict


class MixtralMoE(nn.Module):
    # for tracking the global id of this layer
    layer_count = 0
    all_layer_expert_inputs_info = {}

    def __init__(
        self,
        config: MixtralConfig,
        linear_method: Optional[LinearMethodBase] = None,
        return_profile: bool = False,
    ):
        super().__init__()
        self.config = config
        self.rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        if self.tp_size > self.num_total_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.num_total_experts}.")
        # Split experts equally between ranks
        self.expert_indicies = np.array_split(range(
            self.num_total_experts), self.tp_size)[self.rank].tolist()
        if not self.expert_indicies:
            raise ValueError(
                f"Rank {self.rank} has no experts assigned to it.")

        self.experts = nn.ModuleList([
            MixtralMLP(self.num_total_experts,
                       config.hidden_size,
                       config.intermediate_size,
                       linear_method=linear_method)
            if idx in self.expert_indicies else None
            for idx in range(self.num_total_experts)
        ])
        self.gate = ReplicatedLinear(config.hidden_size,
                                     self.num_total_experts,
                                     bias=False,
                                     linear_method=None)
        self.return_profile = return_profile
        self.layer_id = self.__class__.layer_count
        self.__class__.layer_count += 1
        self.expert_inputs_info = {}
        self.__class__.all_layer_expert_inputs_info.update({self.layer_id: {}})

    def forward_origin(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits, _ = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights,
                                                       self.top_k,
                                                       dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros_like(hidden_states)
        for expert_idx in self.expert_indicies:
            expert_layer = self.experts[expert_idx]
            expert_mask = (selected_experts == expert_idx)

            if self.return_profile:
                # profile token distributions
                num_tokens = expert_mask.sum().item()
                # 更新当前专家历史输入信息
                if expert_idx not in self.expert_inputs_info:
                    self.expert_inputs_info[expert_idx] = []
                self.expert_inputs_info[expert_idx].append(num_tokens)

            if expert_mask.sum() == 0:
                continue
            expert_weights = (routing_weights * expert_mask).sum(dim=-1,
                                                                 keepdim=True)

            current_hidden_states = expert_layer(hidden_states).mul_(
                expert_weights)
            if final_hidden_states is None:
                final_hidden_states = current_hidden_states
            else:
                final_hidden_states.add_(current_hidden_states)

        if self.return_profile:
            assert self.layer_id in self.__class__.all_layer_expert_inputs_info
            self.__class__.all_layer_expert_inputs_info[self.layer_id].update(self.expert_inputs_info)
            # 如果是最后一层，则计算所有层的专家输入统计信息
            if self.layer_id == self.__class__.layer_count - 1:
                gathered_dict = all_gather_dict(deepcopy(self.__class__.all_layer_expert_inputs_info))
                # 将统计信息保存为 JSON 文件和 CSV 文件
                with open(f"expert_input_statistics.json", 'w') as f:
                    json.dump(gathered_dict, f, indent=4)

        return tensor_model_parallel_all_reduce(final_hidden_states).view(
            batch_size, sequence_length, hidden_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # 计算路由权重
        router_logits, _ = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        # 选择 TOP-K 专家
        top_k_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # tracing expert choices for each token
        if os.environ.get("EXPERT_TRACE", "0") == "1":
            if hasattr(self, 'token2experts'):
                selected_experts_tmp = selected_experts.view(batch_size, sequence_length, -1)
                for seq_idx in range(batch_size):
                    length = min(len(self.token2experts[seq_idx]), sequence_length)
                    for token_idx in range(length):
                        expert_indices = selected_experts_tmp[seq_idx, token_idx, :].cpu().numpy().tolist()
                        if self.layer_id not in self.token2experts[seq_idx][token_idx]:
                            self.token2experts[seq_idx][token_idx][self.layer_id] = []
                        self.token2experts[seq_idx][token_idx][self.layer_id].append(expert_indices)
        final_hidden_states = torch.zeros_like(hidden_states)
        ep_size = len(self.expert_indicies)
        world_size = dist.get_world_size()
        num_tokens_all = len(hidden_states) * 2
        num_tokens_per_ep_group = num_tokens_all // world_size
        for local_idx, expert_idx in enumerate(self.expert_indicies):
            expert_layer = self.experts[expert_idx]

            #########################################
            #  method 1: selectedd_experts
            #########################################

            # # Long-tailed
            # selected_experts = torch.zeros_like(selected_experts) # (b,top_k)
            # selected_experts[:, 0] = 1

            # # uniform情况
            # interval_per_expert = len(selected_experts) * 2 // (dist.get_world_size() * len(self.expert_indicies))
            # selected_experts = torch.zeros_like(selected_experts)
            # selected_experts -= 1
            # selected_experts[:interval_per_expert, 0] *= (-1 * expert_idx)

            # expert_mask = (selected_experts == expert_idx)


            #########################################
            #  method 2: expert_mask
            #########################################
            # 正常情况
            expert_mask = (selected_experts == expert_idx)
            
            # # 长尾情况
            # expert_mask = torch.zeros_like(selected_experts).bool()
            # if ep_size == 1:
            #     if self.rank in [0, 1]:
            #         expert_mask[:, 0] = True
            # else:
            #     if self.rank == 0 and local_idx < self.top_k:
            #         expert_mask[:, 0] = True

            # # uniform情况
            # expert_mask = torch.zeros_like(selected_experts)
            # if ep_size == 1:
            #     expert_mask[:num_tokens_per_ep_group, 0] = 1
            # else:
            #     if local_idx == 0:
            #         expert_mask[:num_tokens_per_ep_group, 0] = 1
            # expert_mask = expert_mask.bool()

            if self.return_profile:
                # profile token distributions
                num_tokens = expert_mask.sum().item()
                # 更新当前专家历史输入信息
                if expert_idx not in self.expert_inputs_info:
                    self.expert_inputs_info[expert_idx] = []
                self.expert_inputs_info[expert_idx].append(num_tokens)
            # 创建专家掩码并进行选择
            indices = torch.nonzero(expert_mask)

            # 仅对选中的 hidden_states 进行计算
            if indices.numel() > 0:
                cur_weight = top_k_weights[indices[:, 0], indices[:, 1]].view(-1,1)
                token_mask = expert_mask.sum(-1).bool().view(-1,1)
                selected_hidden_states = torch.masked_select(hidden_states, token_mask).view(-1, hidden_dim)
                current_hidden_states = expert_layer(selected_hidden_states)

                # 应用权重并累加结果
                current_hidden_states.mul_(cur_weight)
                final_hidden_states.index_add_(0, indices[:, 0], current_hidden_states)

        if self.return_profile:
            assert self.layer_id in self.__class__.all_layer_expert_inputs_info
            self.__class__.all_layer_expert_inputs_info[self.layer_id].update(self.expert_inputs_info)
            # 如果是最后一层，则计算所有层的专家输入统计信息
            if self.layer_id == self.__class__.layer_count - 1:
                gathered_dict = all_gather_dict(deepcopy(self.__class__.all_layer_expert_inputs_info))
                self.__class__.gathered_dict = gathered_dict
        return tensor_model_parallel_all_reduce(final_hidden_states).view(
            batch_size, sequence_length, hidden_dim)


class MixtralAttention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 rope_theta: float = 10000,
                 linear_method: Optional[LinearMethodBase] = None,
                 sliding_window: Optional[int] = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            sliding_window=self.sliding_window,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class MixtralDecoderLayer(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = MixtralAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            sliding_window=config.sliding_window,
            linear_method=linear_method)
        self.block_sparse_moe = MixtralMoE(config=config,
                                           linear_method=linear_method)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)
        return hidden_states, residual


class MixtralModel(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList([
            MixtralDecoderLayer(config, linear_method=linear_method)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        rank = get_tensor_model_parallel_rank()
        print(f"rank{rank}: [{input_ids.size(0)}, {input_ids.size(1)}],")
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states,
                                            kv_caches[i], input_metadata,
                                            residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        moe_layer = self.layers[-1].block_sparse_moe
        # for layer_id, layer_info in moe_layer.gathered_dict.items():
        #     for expert_id, expert_info in layer_info.items():
        #         print(f"layer{layer_id} exeprt{expert_id}: {expert_info[1:]}")
        # print('\n======\n')
        return hidden_states


class MixtralForCausalLM(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.model = MixtralModel(config, linear_method)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata)
        return hidden_states

    def sample(
        self,
        hidden_states: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path,
                cache_dir,
                load_format,
                revision,
                fall_back_to_pt=False):
            if "rotary_emb.inv_freq" in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip experts that are not assigned to this worker.
                if ("block_sparse_moe.experts." in name
                        and name not in params_dict):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
