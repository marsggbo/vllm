from collections import deque
from typing import Deque
import torch
import os

from vllm.sequence import SequenceGroup


class Policy:

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        raise NotImplementedError

    def sort_by_priority(
        self,
        now: float,
        seq_groups: Deque[SequenceGroup],
    ) -> Deque[SequenceGroup]:
        return deque(
            sorted(
                seq_groups,
                key=lambda seq_group: self.get_priority(now, seq_group),
                reverse=True,
            ))


class FCFS(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return now - seq_group.arrival_time


class UniformTokenDistribution(Policy):

    def get_priority(
        self,
        now: float,
        seq_group: SequenceGroup,
    ) -> float:
        return now - seq_group.arrival_time

    def get_expert_choice(self, seq_group):
        t2e = seq_group.token2experts
        num_layers = len(t2e[0]) - 1
        num_experts = 8
        expert_choices = torch.zeros((num_layers, num_experts)).to(torch.cuda.current_device())
        for i_token in t2e:
            for layer_id in t2e[i_token]:
                if isinstance(layer_id, int):
                    experts = t2e[i_token][layer_id][0]
                    for expert_id in experts:
                        expert_choices[layer_id][expert_id] += 1
        return expert_choices

    def sort_by_priority(
        self,
        now: float,
        seq_groups: Deque[SequenceGroup],
    ) -> Deque[SequenceGroup]:
        expert_choice_list = []
        if os.environ.get("EXPERT_TRACE", "0") == "1" and len(seq_groups) > 0:
            for seq_group in seq_groups:
                expert_choice = self.get_expert_choice(seq_group)
                expert_choice_list.append(expert_choice)
            expert_choice_list = torch.stack(expert_choice_list, dim=0) # (num_token, num_layer, num_experts)
            similarities = self.compute_similarity(
                expert_choice_list[:, None, :, :], expert_choice_list[None, :, :, :])
            min_mean_max = lambda x: (x.min(), x.mean(), x.max())
            print("similarities", min_mean_max(similarities))
        return deque(
            sorted(
                seq_groups,
                key=lambda seq_group: self.get_priority(now, seq_group),
                reverse=True,
            ))

    def compute_similarity(self, matrix1, matrix2, epsilon=1e-10):
        # 将每一行的数据归一化为概率分布
        distributions1 = (matrix1 + epsilon) / (torch.sum(matrix1, dim=-1, keepdim=True) + epsilon)
        distributions2 = (matrix2 + epsilon) / (torch.sum(matrix2, dim=-1, keepdim=True) + epsilon)

        # 计算平均分布
        average_distribution = 0.5 * (distributions1 + distributions2)

        # 计算KL散度
        kl_divergence1 = torch.sum(distributions1 * torch.log(distributions1 / (average_distribution + epsilon)), dim=-1)
        kl_divergence2 = torch.sum(distributions2 * torch.log(distributions2 / (average_distribution + epsilon)), dim=-1)

        # 计算Jensen-Shannon散度
        js_divergence = 0.5 * (kl_divergence1 + kl_divergence2)
        similarity = 1 - 2 * js_divergence
        return similarity


class PolicyFactory:

    _POLICY_REGISTRY = {
        'fcfs': FCFS,
        'utd': UniformTokenDistribution,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)
