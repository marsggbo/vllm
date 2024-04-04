import types
import logging
import pathlib
import typing
import json
import random
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
from safetensors.torch import load_file

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    Trainer
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from fastchat.train.train import (
    DataArguments,
    ModelArguments,
    TrainingArguments
)

from peft import get_peft_model, LoraConfig

num_layers = 32
num_experts_per_layer = 8
NUM_LABELS = num_layers * num_experts_per_layer
PADDING_SIDE = 'right'
model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"


@dataclass
class CustomArguments:
    ckpt_path: str = field(default=None, metadata={"help": "The checkpoint path for evaluation"})


@dataclass
class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features):
        if not hasattr(self, 'padding_side'):
            self.padding_side = 'right'
        assert self.padding_side in ['left', 'right'], "Padding should be on one side (left or right)"
        non_label_features =[]
        for feature in features:
            item = {key: val for key, val in feature.items() if key in ['input_ids', 'attention_mask']}
            non_label_features.append(item)
        batch = super().__call__(non_label_features)
        return batch


def prepare_dataset(
        data_list: Optional[int] = 0,
        data_size: int = 2000,
        sort_by_len: bool = True,
    ):
    '''
    Args:
        data_list:
            list[string]
            0: alpaca data list
            1: yizhongw data list
            2: alpaca-yizhongw combined data list
    '''
    def load_json(file):
        with open(file, 'r') as f:
            data = json.load(f)
        return data

    if data_list == 0:
        alpaca_data = load_json("/home/nus-hx/code/Sequence-Scheduling/data/alpaca-train-10k.json")
        data_list = []
        for i in range(data_size):
            data_list.append(alpaca_data[i]['conversations'][0]['value'])
    elif data_list == 1:
        yizhongw_data = load_dataset("yizhongw/self_instruct", "super_natural_instructions")
        data_prompts = yizhongw_data['train']['prompt']
        data_list = []
        for i in range(data_size):
            data_list.append(data_prompts[i])
    elif data_list == 2:
        data_list = []
        alpaca_data = load_json("/home/nus-hx/code/Sequence-Scheduling/data/alpaca-train-10k.json")
        for i in range(data_size):
            data_list.append(alpaca_data[i]['conversations'][0]['value'])

        yizhongw_data = load_dataset("yizhongw/self_instruct", "super_natural_instructions")
        data_prompts = yizhongw_data['train']['prompt']
        for i in range(data_size):
            data_list.append(data_prompts[i])

    if sort_by_len:
        data_list = sorted(data_list, key=len)
    data = {"sentence": data_list}
    dataset = Dataset.from_dict(data)
    return dataset


def new_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = []
        for i in range(len(self.lm_head)):
            logits_per_expert = self.lm_head[i](hidden_states).float() # (bs, seq_len, num_experts)
            logits.append(logits_per_expert)
        logits = torch.stack(logits, dim=-2) # # (bs, seq_len, num_layers, num_experts)

        loss = None
        if labels is not None:
            # masked BCE loss
            labels = labels.to(logits.device).float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.view(-1, num_experts_per_layer),
                labels.view(-1, num_experts_per_layer),
                reduction='none')
            loss_mask = labels.view(-1, num_experts_per_layer).sum(-1) != 0
            loss = loss[loss_mask].sum() / loss_mask.sum()
            # BCE loss (not recommended)
            # labels = labels.to(logits.device).float().view(logits.shape)
            # loss = nn.BCEWithLogitsLoss()(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, CustomArguments)
    )
    (
        model_args,
        training_args,
        custom_args
    ) = parser.parse_args_into_dataclasses()
    
    ################################
    # 实例化 model
    ################################
    print('building model...')
    predictor_num_layers = 2
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        cache_dir="/data/common/mixtral/")
    model.model.layers = model.model.layers[:predictor_num_layers]
    model.forward = types.MethodType(new_forward, model)
    model.lm_head = nn.ModuleList([
        nn.Linear(model.config.hidden_size, 8, bias=False) for i in range(32)
    ])
    print('loading weights for prediction...')
    loaded_weights = load_file(custom_args.ckpt_path)
    model.load_state_dict(loaded_weights)
    
    ################################
    # 实例化 tokenizer 和 dataset
    ################################
    print('building tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    preprocess_function = lambda examples: tokenizer(
            examples["sentence"], padding=True, return_tensors="pt", return_attention_mask=True)

    print('building dataset and dataloader')
    dataset = prepare_dataset(data_list=2, sort_by_len=True)
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = CustomDataCollatorWithPadding(tokenizer=tokenizer)
    data_loader = DataLoader(
        tokenized_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    ################################
    # 开始预测
    ################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = model.to(dtype=torch.bfloat16)
    model.eval()
    
    pattern_predictions = []
    batch_times = []
    for batch_id, batch_data in enumerate(data_loader):
        input_ids = batch_data['input_ids'].to(device)
        attention_mask = batch_data['attention_mask'].to(device)
        start_time = time.time()
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        top2_indices = logits.topk(2, dim=-1).indices
        top2_one_hot_pattern_predictions = torch.zeros_like(logits)
        top2_one_hot_pattern_predictions.scatter_(-1, top2_indices, 1)
        extended_attention_mask = attention_mask[:, :, None, None]
        masked_output = top2_one_hot_pattern_predictions * extended_attention_mask
        sequence_level_patterns = torch.sum(masked_output, dim=1)
        end_time = time.time()
        batch_time = end_time - start_time
        batch_times.append(batch_time)
        print(f"Batch-{batch_id+1} takes {batch_time:.4f}s {input_ids.shape}")
        pattern_predictions.append(sequence_level_patterns.cpu().numpy())
    print(f"Each batch averagely takes {np.mean(batch_times[1:-2]):.4f}s")
    pattern_predictions = np.concatenate(pattern_predictions, axis=0)
    return pattern_predictions


if __name__ == "__main__":
    main()


# TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0 python predict_pattern.py \
# --ckpt_path /home/nus-hx/code/vllm/examples/ckpts/finetuneAll_AlltrainMax512StartRandom_EvalMax512_2Layer_bceIgnore_baselr2e-5wd1e-4_head1e-3wd5e-3/checkpoint-5625/model.safetensors \
# --output_dir ./predictions \
# --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
# --bf16 True \
# --do_predict True \
# --per_device_eval_batch_size 64 \
# --bf16_full_eval True