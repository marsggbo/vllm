from vllm import LLM, SamplingParams
import time
import os
import json
import torch
import numpy as np
import random

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
    "/home/nus-hx/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/125c431e2ff41a156b9f9076f744d2f35dd6e67a/"
]


def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

# alpaca_data
alpaca_data = load_json("/home/nus-hx/code/Sequence-Scheduling/data/alpaca-train-10k.json")
num_samples = 800
data = []
for i in range(num_samples):
    data.append(alpaca_data[i]['conversations'][0]['value'])
# Create a sampling params object.
sampling_params = SamplingParams(temperature=1, top_k=100,
                                #  max_tokens=500000
                                 ) # max_tokens=1 disables decoding

# Create an LLM.
llm = LLM(model=models[-1], tensor_parallel_size=2, enforce_eager=True, seed=666) # mistral
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
        for i in range(num_samples):
            prompt_len = outputs[i].outputs[0].seq_group.get_seqs()[0].get_prompt_len()
            output_len = outputs[i].outputs[0].seq_group.get_seqs()[0].get_output_len()
            print(prompt_len, output_len, prompt_len+output_len)
        # print(outputs[1].outputs[0].seq_group.token2experts, outputs[1].outputs[0].seq_group.get_seqs()[0].get_len())
        torch.save(outputs, f"outputs_{idx}.pt")
    print(f"{idx} seq ====> Time elapsed: {cost_iter} seconds")
    # Print the outputs.
    # for output in outputs[-3:]:
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
avg = np.mean(all_time)
std = np.std(all_time)
_all = np.sum(all_time)
print(f"Total time elapsed: {_all:.4f}sec {avg:.4f}±{std:.4f} seconds")


# EXPERT_TRACE=1 CUDA_VISIBLE_DEVICES=1,2 python examples/offline_inference.py # 实测可运行，但是更长的句子还未测试 