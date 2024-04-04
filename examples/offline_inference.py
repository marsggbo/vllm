from vllm import LLM, SamplingParams
import time
import os
import json
import torch
import numpy as np
import random
from datasets import load_dataset
from typing import List, Optional, Tuple, Union, Dict


def set_seed(seed):
    # 设置PyTorch的随机数种子
    torch.manual_seed(seed)

    # 设置CUDA的随机数种子（如果使用GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 设置Python内置的随机数种子
    random.seed(seed)

    # 设置NumPy的随机数种子
    np.random.seed(seed)

set_seed(666)
models = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "facebook/opt-125m",
    "/home/wangyuxin/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/",
    "/home/nus-hx/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/125c431e2ff41a156b9f9076f744d2f35dd6e67a/",
    # 'deepseek-ai/deepseek-moe-16b-base'
]


def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def prepare_data(dataset_list: Dict[str,int]):
    data = []
    # alpaca_data
    if 'alpaca' in dataset_list:
        alpaca_data = load_json("/home/nus-hx/code/Sequence-Scheduling/data/alpaca-train-10k.json")
        num_samples = dataset_list['alpaca']
        for i in range(num_samples):
            data.append(alpaca_data[i]['conversations'][0]['value'])

    # sst2
    if 'sst2' in dataset_list:
        sst2_data = load_dataset("stanfordnlp/sst2")['train'] # contain 67349 samples
        prefix_for_sst2 = '''For each given sentence, determine the sentiment expressed. If the sentiment is positive, return "positive". If the sentiment is negative, return "negative". Consider only these two categories for sentiment analysis. Please analyze the sentiment of the following sentence:'''
        num_samples = dataset_list['sst2']
        for i in range(num_samples):
            data.append(prefix_for_sst2 + sst2_data[i]['sentence'])

    # mrpc
    if 'mrpc' in dataset_list:
        mrpc_data  = load_dataset("SetFit/mrpc")["train"] # contain 3668 samples
        prefix_for_mrpc = '''Given two sentences, determine whether they express the same meaning. If they are paraphrases of each other, return "equivalent". If they are not, return "not equivalent". Please evaluate the following sentence pair:\n
        Sentence 1: "{}"
        Sentence 2: "{}"'''
        num_samples = dataset_list['mrpc']
        for i in range(num_samples):
            sample = mrpc_data[i]
            data.append(prefix_for_mrpc.format(sample['text1'], sample['text2']))

    # # yizhongw/self_instruct
    if 'yizhongw' in dataset_list:
        dataset = load_dataset("yizhongw/self_instruct", "super_natural_instructions")
        data_prompts = dataset['train']['prompt']
        num_samples = dataset_list['yizhongw']
        for i in range(num_samples):
            data.append(data_prompts[i])

    if 'tick666-math' in dataset_list:
        dataset = load_dataset("TICK666/Basic-Math-Chinese-1M-V1.1")['train'] # contains 1000000 samples
        num_samples = dataset_list['tick666-math']
        for i in range(num_samples):
            data.append(dataset[i]['text'])
    print(f"The data contains {len(data)} samples.")
    return data

dataset_list = {
    'alpaca': 10000,
    # 'sst2': 10000,
    # 'mrpc': 2000,
    'tick666-math': 10000,
    # 'yizhongw': 10000
}
print(f'Building dataset including {dataset_list}')
data = prepare_data(dataset_list)

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_k=100,
                                 max_tokens=64
                                 ) # max_tokens=1 disables decoding

# Create an LLM.
llm = LLM(model=models[-1], tensor_parallel_size=2, enforce_eager=True, seed=666
        #   , trust_remote_code=True
          ) # mistral
balance_prefilling = False
if balance_prefilling:
    tokenizer = llm.get_tokenizer()
    num_tokens = []
    for sample in data:
        num_tokens.append(len(tokenizer.encode(sample)))
    sorted_indices = sorted(range(len(num_tokens)), key=lambda idx: num_tokens[idx])
    num_tokens = [num_tokens[idx] for idx in sorted_indices]
    data = [data[idx] for idx in sorted_indices]

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
all_time = []
# prompts = data # one by one inference
prompts = [data] * 1 # batch continuous inference
for idx, prompt in enumerate(prompts):
    start = time.perf_counter()
    outputs = llm.generate(prompt, sampling_params)
    end = time.perf_counter()
    cost_iter = end - start
    all_time.append(cost_iter)
    if os.environ.get("EXPERT_TRACE", "0") == "1":
        saved_name = ""
        for name, length in dataset_list.items():
            saved_name += f"{name}_{length}"
        torch.save(outputs, f"./data/{saved_name}.pt")
    print(f"{idx} seq ====> Time elapsed: {cost_iter} seconds")
    # Print the outputs.
    for output in outputs[-2:]:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
avg = np.mean(all_time)
std = np.std(all_time)
_all = np.sum(all_time)
print(f"Total time elapsed: {_all:.4f}sec {avg:.4f}±{std:.4f} seconds")


# EXPERT_TRACE=1 CUDA_VISIBLE_DEVICES=1,2 python examples/offline_inference.py # 实测可运行，但是更长的句子还未测试 

# PATTERN_SORT=1 CUDA_VISIBLE_DEVICES=0 python examples/offline_inference.py # 开启 pattern sort
