import os
import json
import random
import re
import string
import tqdm
import argparse
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from gpt3_api import make_requests as make_gpt3_requests

from sentence_transformers import SentenceTransformer, SimilarityFunction
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("./all-MiniLM-L6-v2")


random.seed(42)


def encode_prompt(prompt_instructions, classification=False):
    """Encode multiple prompt instructions into a single string."""
    if classification:
        prompt = ("Come up with a series of classification tasks. Try to specify the possible output labels when "
                  "possible. The requirement is for all these tasks to be related to "
                  "vulnerability descriptions:\n")
    else:
        prompt = ("Come up with a series of new instructions as diverse as possible(just instructions). The "
                  "requirement is for all these instructions to be related to the completion of the vulnerability "
                  "description: \n")

    for idx, instruction in enumerate(prompt_instructions):
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        prompt += f"{idx + 1}. {instruction}\n"
    prompt += f"{len(prompt_instructions) + 1}."
    return prompt


def sample_machine_instructions(machine_instructions, similarities, n):
    # 使用min函数，为了满足machine_instructions为0条的初始情况，为0时即为空，即不抽取，完全使用种子指令
    return random.sample(machine_instructions, min(n, len(machine_instructions)))


def find_word_in_string(w, s):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search(s)


def post_process_gpt3_response(response):
    insList = []
    pattern = re.compile(r'^\d+\.\s+')
    taskList = response.content.split('\n')[1:-1]
    for ins in taskList:
        ins = pattern.sub('', ins)
        insList.append(ins)

    instructions = []
    for inst in insList:
        # 避免开头为 ** 的指令被过滤掉
        inst = re.sub(r"\s+", " ", inst).strip().strip("**")
        inst = inst.strip().capitalize()
        if inst == "":
            continue
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        if any(find_word_in_string(word, inst) for word in
               ["image", "images", "graph", "graphs", "picture", "pictures", "file", "files", "map", "maps", "draw",
                "plot", "go to"]):
            continue
        if inst.startswith("Write a program"):
            continue
        if inst[0] in string.punctuation:
            continue
        if not inst[0].isascii():
            continue
        instructions.append(inst)
    return instructions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        default="data/gpt3_generations/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        required=True,
        default="data/seed_tasks.jsonl",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--num_instructions_to_generate",
        type=int,
        default=100,
        help="th",
    )
    parser.add_argument(
        "--use_clf_seed_tasks_only",
        action="store_true",
        help="If specified, we will only use the classification seed tasks to prompt new instructions. This will lead "
             "to more classification instructions.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="davinci",
        help="The engine to use."
    )
    parser.add_argument(
        "--num_prompt_instructions",
        type=int,
        default=8,
        help="The number of instructions to use in the prompt."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send to GPT3 at a time."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]
    if args.use_clf_seed_tasks_only:
        seed_tasks = [t for t in seed_tasks if t["is_classification"]]
    seed_instructions = [t["instruction"] for t in seed_tasks]
    print(f"Loaded {len(seed_instructions)} human-written seed instructions")

    os.makedirs(args.batch_dir, exist_ok=True)
    request_idx = 0
    machine_instructions = []
    if os.path.exists(os.path.join(args.batch_dir, "origin_combined_all_instructions.jsonl")):
        with open(os.path.join(args.batch_dir, "origin_combined_all_instructions.jsonl"), "r") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_instructions.append(instruction_info["instruction"])
                request_idx = int(instruction_info["request_idx"]) + 1
        print(f"Loaded {len(machine_instructions)} machine-generated instructions")

    # 获取每个种子指令及目前已经生成得到的新指令对应的特征向量
    all_instructions_vectors = []
    for ins in tqdm.tqdm(seed_instructions + machine_instructions):
        vector = model.encode(ins, convert_to_tensor=True)
        all_instructions_vectors.append(vector)

    # 创建RougeL指标计算对象
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # 开始生成
    progress_bar = tqdm.tqdm(total=args.num_instructions_to_generate)
    if machine_instructions:
        progress_bar.update(len(machine_instructions))

    with open(os.path.join(args.batch_dir, "origin_combined_all_instructions.jsonl"), "a") as fout:
        while len(machine_instructions) < args.num_instructions_to_generate:
            batch_inputs = []
            for _ in range(args.request_batch_size):
                # 动态抽取新生成指令中的两条，再结合种子指令随机抽样形成新的prompt
                prompt_instructions = sample_machine_instructions(
                    machine_instructions,
                    similarities=None,
                    n=2)

                # 从池中抽取人工指令
                prompt_instructions += random.sample(seed_instructions,
                                                     args.num_prompt_instructions - len(prompt_instructions))
                random.shuffle(prompt_instructions)
                prompt = encode_prompt(prompt_instructions, classification=args.use_clf_seed_tasks_only)
                # 添加到本batch中
                batch_inputs.append(prompt)
            results = make_gpt3_requests(
                engine=args.engine,
                prompts=batch_inputs,
                max_tokens=1024,
                temperature=0.4,
                top_p=0.4,
                system_ins='You are now an expert in generating instructions for large models.'
            )
            instructions = []
            all_metadata = []
            for result in results:
                new_instructions = post_process_gpt3_response(result["response"])
                # if len(new_instructions) == 0:
                #     continue
                instructions += new_instructions
                all_metadata += [result] * len(new_instructions)


            all_instructions = seed_instructions + machine_instructions

            for inst, metadata in zip(instructions, all_metadata):
                rouge_scores = []
                for allOfOne in all_instructions:
                    rouge_scores.append(scorer.score(inst, allOfOne)["rougeL"].fmeasure)

                most_similar_instructions = {
                    all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
                }

                if min(rouge_scores) > 0.7:
                    continue

                fout.write(json.dumps(dict({
                    "instruction": inst,
                    "most_similar": most_similar_instructions,
                    "avg_similarity_score": float(np.mean(rouge_scores)),
                    "metadata": str(metadata),
                    "request_idx": str(request_idx)
                })) + "\n")
                # 一次外层while循环中会进行inst与metadata的内层循环，
                # 该内层循环会涉及多条指令的写入，即在一次外层循环后，len(machine_instructions)的变化可能会大于1
                # 因此会出现最后写入到文件的指令数大于50的情况，即上一次while循环len < 50，经历下一次while循环的内层循环时，len值增量大于1，此时len > 50，经while条件判断，跳出循环
                machine_instructions.append(inst)

                progress_bar.update(1)
            request_idx += 1
