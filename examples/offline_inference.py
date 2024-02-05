from vllm import LLM, SamplingParams
import time
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
    "/home/wangyuxin/.cache/huggingface/hub/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/"
]


def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

alpaca_data = load_json("/home/wangyuxin/xinhe/Sequence-Scheduling/data/alpaca-train-10k.json")
num_samples = 800
data = []
for i in range(num_samples):
    data.append(alpaca_data[i]['conversations'][0]['value'])
# Sample prompts.
# data = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
#     """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nDescribe a time when you had to make a difficult decision.\n\n### Response:""",
#     """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nExplain the concept of a bubble sort algorithm to a non-technical audience.\n\n### Response:""",
#     """In this task, you are given an input list A. You need to extract and sort the unique digits used in the list in ascending order. Return -1 if there is no digit in the list. Input: ['425', '29', 'i', '255', '201', 'r', '29', '191', '441', '239', 'o'] Output:""",
#     """In this task, you are given a set of context paragraphs, some supporting facts and an answer of a question. Your task is to generate question for given answer based on set of context paragraphs, supporting facts and an answer. Input: Context_1 : Miss Seventeen is a reality television show on MTV that aired from October 17, 2005 to December 19, 2005. The show consisted of 17 young women competing for an internship at and a college scholarship. Atoosa Rubenstein was the main judge, she was the youngest editor-in-chief ever to run "Seventeen magazine". They picked 17 girls from around the United States who were not only photogenic but also had been at the top of their class, to provide a role model for young women. The girls were flown to New York, where they would take part in a contest similar in format to The Apprentice — they would be given tasks to be done by Atoosa, and in each episode one of the girls would be eliminated from the competition. The winner would get her face on the cover of "Seventeen magazine", a college scholarship and would be offered an internship job on the magazine. Context_2 : The Sancy, a pale yellow diamond of 55.23 carat, was once reputed to have belonged to the Mughals of antiquity, but is more likely of Indian origin owing to its cut, which is unusual by Western standards. Context_3 : The Spirit of de Grisogono is the world's largest cut black diamond and the world's fifth largest diamond overall. Starting at an uncut weight of 587 carat, it was taken from its origin in western Central African Republic and cut by Swiss jeweler De Grisogono. The resulting mogul-cut diamond weighs 312.24 carat and is set in a white gold ring with 702 smaller white diamonds totaling 36.69 carat. The ring is said to have been sold. Context_4 : Love & Letter, also known as First Love & Letter, is the first studio album by South Korean boy group Seventeen released on April 29, 2016. The album is a follow-up to the group's two EPs, "17 Carat" and "Boys Be" (2015). Context_5 : Rules of origin are used to determine the country of origin of a product for purposes of international trade. There are two common types of rules of origin depending upon application, the preferential and non-preferential rules of origin (19 CFR 102). The exact rules vary from country to country, from agreement to agreement. Context_6 : 17 Carat is the debut extended play by South Korean boy group Seventeen. It was released on May 29, 2015 by Pledis Entertainment and distributed by LOEN Entertainment. "Adore U" serves as the lead single for the extended play. Context_7 : Seventeen (Hangul: 세븐틴 ), also stylized as SEVENTEEN or SVT, is a South Korean boy group formed by Pledis Entertainment in 2015. The group consists of thirteen members who are separated into three sub-units, each with different areas of specialization: a 'Hip-Hop Unit', 'Vocal Unit', and 'Performance Unit'. They have released one studio album and four extended plays. Context_8 : "Fourteen Carat Mind" is a song written by Dallas Frazier and Larry Lee, and recorded by American country music artist Gene Watson. It was released in September 1981 as the first single from the album "Old Loves Never Die". "Fourteen Carat Mind" was Gene Watson's twentieth country hit and his only song to hit number one on the country chart. The single stayed at number one for one week and spent a total of fifteen weeks on the country chart. Context_9 : The Aurora Green Diamond is a 5.03 carat vivid green diamond with VS2 clarity. In May 2016, the Aurora Green became the largest ever vivid green diamond to ever sell at auction. The record was previous held by a 2.54 carat Fancy Vivid Green VS1 diamond that was sold by Sotheby’s on November 17, 2009 for $1.22 million per carat according to the Diamond Investment & Intelligence Center. On May 31, 2016, the diamond, which was originally owned by Scarselli Diamonds was sold by Christie's for a record price per carat of $3.3 million to Chinese jewelry company Chow Tai Fook, totaling $16.8 million. Context_10 : The South Korean boy band Seventeen embarked on their first concert tour entitled Seventeen 1st Asia Tour 2016 Shining Diamonds in July through September of 2016, performing at venues including Singapore, Australia, New Zealand and China. The string of concerts began in South Korea where 13,000 tickets were sold. They have also held four showcases, the most notable being their debut showcase, "Seventeen 1st Mini Album '17 Carat' Showcase""",
#     """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nConvert the following sentence into the present continuous tense\n\n### Input:\nHe reads books\n\n### Response:""",
#     """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nGive an example of a metaphor that uses the following object\n\n### Input:\nStars\n\n### Response:""",
#     """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nExplain the concept of a bubble sort algorithm to a non-technical audience.\n\n### Response:""",
#     """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nGenerate a creative birthday wish for a friend.\n\n### Response:"""
# ]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_k=50, max_tokens=1) # max_tokens=1 disables decoding

# Create an LLM.
# llm = LLM(model=models[1], load_format='dummy', tensor_parallel_size=2) # test
llm = LLM(model=models[-1], tensor_parallel_size=8, enforce_eager=True, seed=666) # mistral
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
prompts = [data] * 10 # batch continuous inference
for idx, prompt in enumerate(prompts):
    start = time.perf_counter()
    outputs = llm.generate(prompt, sampling_params)
    end = time.perf_counter()
    cost_iter = end - start
    all_time.append(cost_iter)
    # print(f"{idx} seq ====> Time elapsed: {cost_iter} seconds")
    # Print the outputs.
    # for output in outputs[-3:]:
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
avg = np.mean(all_time)
std = np.std(all_time)
print(f"Total time elapsed: {avg:.4f}±{std:.4f} seconds")


# CUDA_VISIBLE_DEVICES=2,3 python examples/offline_inference.py # 实测可运行，但是更长的句子还未测试 