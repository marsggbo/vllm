import types
import logging
import pathlib
import typing
import random
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    Trainer
)
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput

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
class TokenClassificationLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    labels: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    
@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    inference_mode: bool = False


# 定义 MoEPatternDataset 类
class MoEPatternDataset(Dataset):
    def __init__(self, data_file: str, training=False):
        self.data_file = data_file
        self.data = torch.load(data_file) # merged_data.pt
        self.training = training
        self.truncate_ratio = 1.

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.training:
            self.truncate_ratio = random.uniform(0.1, 1)
        seq_data = self.data[idx]
        input_ids = np.array(seq_data['data'])[:,0] # 第 idx 个数据的 token indices, (seq_len,)
        seq_len = len(input_ids)
        labels = np.stack(np.array(seq_data['data'])[:,1]).reshape(seq_len, -1) # 第 idx 个数据所有 token 的 pattern matrix, (seq_len, L*E)
        attention_mask = torch.ones(len(input_ids)) # (seq_len,)
        
        truncate_length = int(seq_len * self.truncate_ratio)
        start_index = random.randint(0, seq_len - truncate_length)
        end_index = start_index + truncate_length
        return {
            'input_ids': input_ids[start_index:end_index],
            'attention_mask': attention_mask[start_index:end_index],
            'labels': labels[start_index:end_index]
        }


@dataclass
class PatternDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features):
        if not hasattr(self, 'padding_side'):
            self.padding_side = 'right'
        assert self.padding_side in ['left', 'right'], "Padding should be on one side (left or right)"
        non_label_features =[]
        for feature in features:
            item = {key: val for key, val in feature.items() if key in ['input_ids', 'attention_mask']}
            non_label_features.append(item)
        batch = super().__call__(non_label_features)
        label_features = [feature['labels'] for feature in features] # (seq_len, L*E) 可以是 0/1，也可是 weighted values
        # 计算最大长度以进行padding
        max_length = max(len(label) for label in label_features)
        # 进行padding：对于不足max_length的部分，用全0的pattern填充
        padded_labels = []
        for label in label_features:
            # 创建一个足够大的全 0 tensor来存放padded的labels
            label = torch.tensor(label)
            padded_label = torch.zeros((max_length, label.shape[-1]))
            # 将实际的label值复制到全0tensor的前面
            if self.padding_side == 'left':
                padded_label[-1*len(label):, :] = label # padding left
            else:
                padded_label[:len(label), :] = label # padding right
            padded_labels.append(padded_label)
        # 将padded_labels转换为一个tensor，并加入到batch中
        batch['labels'] = torch.stack(padded_labels, dim=0)
        
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch


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
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            labels = labels.to(logits.device).float()
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return TokenClassificationLMOutputWithPast(
            loss=loss,
            logits=logits,
            labels=labels,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def compute_metrics(outputs):
    true_labels = outputs.labels
    pred_labels = outputs.logits
    true_indices = torch.nonzero(true_labels)[:,-1].view(true_labels.size(0), -1, 2)
    pred_indices = pred_labels.topk(2)[1].sort()[0]
    mask = (pred_indices == true_indices)
    acc = mask.sum() / true_indices.numel()
    return {
        'accuracy': acc
    }


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    
    ################################
    # 实例化 tokenizer 和 model
    ################################
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path
    )
    # # for debug
    config.num_hidden_layers = 8
    config.hidden_size = 1024
    config.intermediate_size = 2048
    model = AutoModelForCausalLM.from_config(config)
    ## for real training
    # model = AutoModelForCausalLM.from_pretrained(
    #     "mistralai/Mistral-7B-Instruct-v0.2")
    model.forward = types.MethodType(new_forward, model)
    model.lm_head = nn.Linear(model.config.hidden_size, num_layers*num_experts_per_layer, bias=False)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side=PADDING_SIDE,
    )
    ## 方案 1：设置 pad_token
    tokenizer.pad_token = tokenizer.unk_token
    # # 方案 2：设置 pad_token
    # # 确保tokenizer已经设置了pad_token，如果没有，我们添加一个
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': '<pad>'})
    # # 需要更新模型的词汇表大小
    # model.resize_token_embeddings(len(tokenizer))
    
    ################################
    # 定义 LoRA 配置
    ################################
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        inference_mode=lora_args.inference_mode,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.lm_head.weight.requires_grad = True
    if training_args.local_rank == 0:
        model.print_trainable_parameters()

    ################################
    # 实例化DataCollatorWithPadding
    ################################
    print('loading PatternDataCollatorWithPadding')
    data_collator = PatternDataCollatorWithPadding(tokenizer=tokenizer)
    data_collator.padding_side = PADDING_SIDE
    ################################
    # 实例化 MoEPatternDataset
    ################################
    print('loading MoEPatternDataset')
    train_dataset = MoEPatternDataset('./merged_data.pt', training=True)
    eval_dataset = MoEPatternDataset('./merged_data.pt', training=False)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    model.config.use_cache = False
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    if training_args.local_rank == 0:
        # model.save_pretrained(training_args.output_dir, state_dict=state_dict)
        model.save_pretrained(training_args.output_dir)

def test_dataset():
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.unk_token
    data_collator = PatternDataCollatorWithPadding(tokenizer=tokenizer)
    dataset = MoEPatternDataset('./merged_data.pt')
    data_loader = DataLoader(dataset, batch_size=8, collate_fn=data_collator)
    for idx, batch in enumerate(data_loader):
        if idx > 5:
            break
        print("Batch input_ids shape:", batch["input_ids"].shape)  # 打印每个batch的input_ids的shape
        print("Batch attention_mask shape:", batch["attention_mask"].shape)  # 打印每个batch的attention_mask的shape
        print("Batch labels shape:", batch["labels"].shape)  # 打印每个batch的attention_mask的shape
        print('\n\n=========')

def test_model():
    config = AutoConfig.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2")
    # for debug
    config.num_hidden_layers = 4
    config.hidden_size = 1024
    config.intermediate_size = 2048
    model = AutoModelForCausalLM.from_config(config)
    model.forward = types.MethodType(new_forward, model)
    model.lm_head = nn.Linear(model.config.hidden_size, NUM_LABELS, bias=False)
    lora_args = LoraArguments()
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        inference_mode=lora_args.inference_mode,
        task_type="CAUSAL_LM",
    )
    # lora_config = LoraConfig(
    #     r=8,
    #     lora_alpha=16,
    #     target_modules=field(
    #         default_factory=lambda: ["q_proj", "v_proj"]
    #     ),
    #     lora_dropout=0.05,
    #     bias="none",
    #     inference_mode=False,
    #     task_type="CAUSAL_LM",
    # )
    model = get_peft_model(model, lora_config)
    model.lm_head.weight.requires_grad = True
    model.print_trainable_parameters()
    for named, param in model.named_parameters():
        if param.requires_grad:
            print(named, param.shape)
    x = torch.randint(0, 100, (4, 64))
    labels = torch.ones((4, 64, NUM_LABELS))
    output = model(x, labels=labels)
    print(output.logits.shape)
    print(output.loss)

if __name__ == "__main__":
    train()
    # test_dataset()
    # test_model()
