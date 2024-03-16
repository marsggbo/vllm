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
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    inference_mode: bool = False


# 定义 MoEPatternDataset 类
class MoEPatternDataset(Dataset):
    def __init__(self, data_path_or_file: str, training=False, num_evaluation=1000):
        self.data_path_or_file = data_path_or_file
        if isinstance(data_path_or_file, str) and data_path_or_file.endswith(".pt"):
            self.data = torch.load(data_path_or_file) # merged_data.pt
        else:
            self.data = data_path_or_file
        self.training = training
        self.truncate_ratio = 1.
        
        # evaluate a part of data
        self.num_evaluation = num_evaluation
        if not self.training:
            if num_evaluation == 1:
                self.data = self.data[:num_evaluation]
            else:
                N = len(self.data)
                step = (N - 1) / (num_evaluation - 1)
                indices = np.round(np.arange(0, N - 1, step)).astype(int)
                if len(indices) < num_evaluation:  # 确保返回K个样本，特别是当N非常接近K时
                    indices = np.append(indices, N-1)
                self.data = [self.data[i] for i in indices]

    def __len__(self):
        if self.training:
            return len(self.data)
            # return 4000 # for debug
        else:
            return self.num_evaluation
            # return 50 # for debug
    
    def __getitem__(self, idx):
        seq_data = self.data[idx]
        input_ids = np.array(seq_data['data'])[:,0] # 第 idx 个数据的 token indices, (seq_len,)
        seq_len = len(input_ids)
        labels = np.stack(np.array(seq_data['data'])[:,1]).reshape(seq_len, -1) # 第 idx 个数据所有 token 的 pattern matrix, (seq_len, L*E)
        attention_mask = torch.ones(len(input_ids)) # (seq_len,)
        
        if self.training:
            self.truncate_ratio = random.uniform(0.3, 1)
            truncate_length = int(seq_len * self.truncate_ratio)
            # truncate_length = min(int(seq_len * self.truncate_ratio), 512)
            # start_index = random.randint(0, seq_len - truncate_length)
        else:
            truncate_length = min(int(seq_len * self.truncate_ratio), 256)
            # start_index = 0
        start_index = 0
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
        logits = []
        for i in range(len(self.lm_head)):
            logits_per_expert = self.lm_head[i](hidden_states).float() # (bs, seq_len, num_experts)
            logits.append(logits_per_expert)
        logits = torch.stack(logits, dim=-2) # # (bs, seq_len, num_layers, num_experts)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device).float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.view(-1, num_experts_per_layer),
                labels.view(-1, num_experts_per_layer),
                reduction='none')
            loss_mask = labels.view(-1, num_experts_per_layer).sum(-1) != 0
            loss = loss[loss_mask].sum() / loss_mask.sum()

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


def acc_precision_recall_f1(y_true, y_pred):
    # 真正例 (True Positives)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    
    # 假正例 (False Positives)
    FP = np.sum((y_true == 0) & (y_pred == 1))
    
    # 假负例 (False Negatives)
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    # 真负例 (True Negatives)
    TN = np.sum((y_true == 0) & (y_pred == 0))
    
    y_true = y_true.reshape(-1, 256)
    y_pred = y_pred.reshape(-1, 256)
    print(f"origin y_true.shape={y_true.shape}")
    indices = np.any(y_true, axis=-1)
    y_true = y_true[indices]
    y_pred = y_pred[indices]
    print(f"filtered y_true.shape={y_true.shape}")
    
    # 准确率
    num_tokens = y_true.shape[0]
    accuracy = TP / (num_tokens*64)
    recall = 0
    precision = 0
    f1 = 0
    print(f"non-padding ratio: {indices.sum()}/{len(indices)}={indices.sum()/len(indices)}\n")

    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1,
    }


def compute_metrics(outputs):    
    true_labels = outputs.label_ids
    pred_labels = outputs.predictions
    if len(pred_labels.shape) == 3:
        bs, seq_len, dim = pred_labels.shape
    elif len(pred_labels.shape)==4:
        bs, seq_len, num_layer, num_experts = pred_labels.shape
        dim = num_layer * num_experts
    assert dim == NUM_LABELS, "Dimension of predictions should be {} but got {}".format(NUM_LABELS, dim)
    true_labels = true_labels.reshape(-1, num_experts_per_layer)
    pred_labels = pred_labels.reshape(-1, num_experts_per_layer)
        
    # Convert predictions to top-2 one-hot encoding
    preds_one_hot = np.zeros_like(pred_labels)
    top2_indices = np.argsort(pred_labels, axis=1)[:, -2:]
    rows = np.arange(pred_labels.shape[0])[:, None]
    preds_one_hot[rows, top2_indices] = 1

    return acc_precision_recall_f1(
        true_labels, preds_one_hot
    )


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
    # config.num_hidden_layers = 8
    # config.hidden_size = 1024
    # config.intermediate_size = 2048
    # model = AutoModelForCausalLM.from_config(config)
    ## for real training
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        cache_dir="/data/common/mixtral/")
    model.model.layers = model.model.layers[:8]
    model.forward = types.MethodType(new_forward, model)
    model.lm_head = nn.ModuleList([
        nn.Linear(model.config.hidden_size, 8, bias=False) for i in range(32)
    ])
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=256,
        padding_side=PADDING_SIDE,
        truncation_side=PADDING_SIDE,
        padding=True,
        truncation=True
    )
    ## 方案 1：设置 pad_token
    tokenizer.pad_token = tokenizer.eos_token
    # # 方案 2：设置 pad_token
    # # 确保tokenizer已经设置了pad_token，如果没有，我们添加一个
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': '<pad>'})
    # # 需要更新模型的词汇表大小
    # model.resize_token_embeddings(len(tokenizer))
    
    ################################
    # 定义 LoRA 配置
    ################################
    # lora_config = LoraConfig(
    #     r=lora_args.lora_r,
    #     lora_alpha=lora_args.lora_alpha,
    #     target_modules=lora_args.lora_target_modules,
    #     lora_dropout=lora_args.lora_dropout,
    #     bias=lora_args.lora_bias,
    #     inference_mode=lora_args.inference_mode,
    #     task_type="CAUSAL_LM",
    # )
    # model = get_peft_model(model, lora_config)
    # model.model.model.norm.weight.requires_grad = True
    # for module in model.lm_head:
    #     module.weight.requires_grad = True
    # if training_args.local_rank == 0:
    #     model.print_trainable_parameters()

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
    origin_data = torch.load('./merged_data.pt')
    train_dataset = MoEPatternDataset(origin_data, training=True)
    eval_dataset = MoEPatternDataset(origin_data, training=False)

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
    print('Start training')
    output_dir = training_args.output_dir
    if training_args.run_name:
        output_dir += f'_{training_args.run_name}'
    if list(pathlib.Path(output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    if training_args.local_rank == 0:
        # model.save_pretrained(output_dir, state_dict=state_dict)
        model.save_pretrained(output_dir)

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
