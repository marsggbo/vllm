import torch
import numpy as np
import random
import types
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # Accelerator,  # This is used for easy mixed precision and multi-GPU support
)
from safetensors.torch import load_file
from train_pattern_predictor import (
    new_forward,
    MoEPatternDataset,
    PatternDataCollatorWithPadding,
    acc_precision_recall_f1,
)

# Initialize the Accelerator for mixed precision and multi-GPU
# accelerator = Accelerator(mixed_precision="bf16")  # Use mixed_precision="fp16" if bf16 is not supported

# Constants and configurations
model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"
cache_dir = "/data/common/mixtral/"
ckpt_path = "/home/nus-hx/code/vllm/examples/ckpts/finetuneAll_AlltrainMax512StartRandom_EvalMax512_2Layer_bceIgnore_baselr2e-5wd1e-4_head1e-3wd5e-3/checkpoint-5625"
file_path = ckpt_path + "/model.safetensors"
num_layers, num_experts_per_layer = 32, 8
predictor_num_layers = 2
device = "cuda:1"

# Load model and tokenizer
print('loading model and tokenizer')
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

# Modify the model as needed
model.model.layers = model.model.layers[:predictor_num_layers]
model.forward = types.MethodType(new_forward, model)
model.lm_head = nn.ModuleList([nn.Linear(model.config.hidden_size, 8, bias=False) for _ in range(num_layers)])

# Load the model weights
loaded_weights = load_file(file_path)
model.load_state_dict(loaded_weights)

# Dataset and DataLoader setup
print('loading dataset')
origin_data = torch.load('./merged_data.pt')
eval_dataset = MoEPatternDataset(origin_data, training=False, eval_max_seq_size=None, data_type=2)
data_collator = PatternDataCollatorWithPadding(tokenizer=tokenizer)
loader = DataLoader(eval_dataset, batch_size=32, collate_fn=data_collator, shuffle=False)

# # Prepare everything with accelerator
# model, loader = accelerator.prepare(model, loader)
model = model.to(device)

# Evaluation loop example
results = {
    'accuracy': [],
    'recall': [],
    'precision': [],
    'f1': [],
}
print('start evaluation...')
for i, sample in enumerate(loader):
    if i == -1:  # Just an example to break early
        break
    sample = {key: val.to(device) for key, val in sample.items()}
    output = model(**sample)
    true_labels = sample['labels']
    pred_labels = output.logits
    
    true_labels = true_labels.reshape(-1, num_experts_per_layer)
    pred_labels = pred_labels.detach().reshape(-1, num_experts_per_layer)
        
    # Convert predictions to top-2 one-hot encoding
    preds_one_hot = np.zeros_like(pred_labels.cpu().numpy())
    top2_indices = np.argsort(pred_labels.cpu().numpy(), axis=1)[:, -2:]
    rows = np.arange(pred_labels.shape[0])[:, None]
    preds_one_hot[rows, top2_indices] = 1
    result = acc_precision_recall_f1(true_labels.cpu().numpy(), preds_one_hot)
    print('step', i, result['accuracy'])
    for key in result.keys():
        results[key].append(result[key])

# Aggregate and print results
for key, val in results.items():
    print(f"{key}: {np.mean(val)}")

# Note: Modifications in result calculation may be required to handle tensors efficiently with Accelerator
