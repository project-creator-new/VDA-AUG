import difflib
import os
import json
import random
import tqdm
import re
import argparse
import pandas as pd
from collections import OrderedDict
from gpt3_api import make_requests as make_gpt3_requests
from templates.gen_vulnerability_instance_template import output_first_template_for_clf, input_first_template_for_gen, new_input_template_for_gen


random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="machine_generated_instructions.jsonl"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="machine_generated_instances.jsonl",
    )
    parser.add_argument(
        "--num_instructions",
        type=int,
        help="if specified, only generate instance input for this many instructions",
    )
    parser.add_argument(
        "--max_instances_to_generate",
        type=int,
        default=5,
        help="The max number of instances to generate for each instruction.",
    )
    parser.add_argument(
        "--generation_tasks_only",
        action="store_true",
        help="If specified, only do for generation tasks.",
    )
    parser.add_argument(
        "--classification_tasks_only",
        action="store_true",
        help="If specified, only do for classification tasks.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="davinci",
        help="The engine to use."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,  # 默认为5
        help="The number of requests to send in a batch."
    )
    return parser.parse_args()


def fuzzy_match(s1, s2, threshold=0.9):
    if s1 == s2:
        print(s1)
        print(True)
        return True
    else:
        print(s1)
        print(difflib.SequenceMatcher(None, s1, s2).ratio() > threshold)
        return difflib.SequenceMatcher(None, s1, s2).ratio() > threshold


if __name__ == '__main__':
    args = parse_args()

    with open(os.path.join(args.batch_dir, args.input_file)) as fin:
        lines = fin.readlines()
        if args.num_instructions is not None:
            lines = lines[:args.num_instructions]
        tasks = []
        for line in lines:
            data = json.loads(line)
            if "metadata" in data:
                data["instruction_metadata"] = data["metadata"]
                del data["metadata"]
            tasks.append(data)

    task_clf_types = {}
    with open(os.path.join(args.batch_dir, 'is_clf_or_not_GLM-4-Plus.jsonl')) as fin:
        for line in fin:
            data = json.loads(line)
            task_clf_types[data["instruction"]] = data["is_classification"].strip() in ["Yes", "yes", "YES"]

    if args.classification_tasks_only:
        tasks = [task for task in tasks if task["instruction"] in task_clf_types.keys() and task_clf_types[task["instruction"]]]

    if args.generation_tasks_only:
        tasks = [task for task in tasks if not (task["instruction"] in task_clf_types.keys() and task_clf_types[task["instruction"]])]

    output_path = os.path.join(args.batch_dir, args.output_file)
    existing_requests = {}
    if os.path.exists(output_path):
        with open(output_path, encoding='utf-8') as fin:
            for line in tqdm.tqdm(fin):
                try:
                    data = json.loads(line)
                    existing_requests[data["instruction"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing requests")

    progress_bar = tqdm.tqdm(total=len(tasks))
    with open(output_path, "w", encoding='utf-8') as fout:
        for batch_idx in range(0, len(tasks), args.request_batch_size):
            batch = tasks[batch_idx: batch_idx + args.request_batch_size]
            if all(d["instruction"] in existing_requests for d in batch):
                for d in batch:
                    print(d['instruction'] + "*******************")
                    data = existing_requests[d["instruction"]]
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "raw_instances", "instance_metadata", "instruction_metadata", 
                            "most_similar", "avg_similarity_score"]
                        )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                prompts = []
                for task in batch:
                    print(task['instruction'] + "*******************NEW")
                    if task["instruction"] in task_clf_types.keys():
                        if task_clf_types[task["instruction"]]:
                            prompt = output_first_template_for_clf + " " + task["instruction"].strip()
                            prompts.append(prompt)
                        else:
                            prompt = input_first_template_for_gen + " " + task["instruction"].strip()
                            prompts.append(prompt)

                results = make_gpt3_requests(
                    engine=args.engine,
                    prompts=prompts,
                    max_tokens=2048,
                    temperature=0.1,
                    top_p=0.1,
                    system_ins='Now, as an expert in generating input-output instances, generate the corresponding '
                               'input and correct output based on the content of the instruction.')
                patternOut = r'^(?:###\s+)?output\d*[:.]?\s+.+'
                inAndOutList = []
                metadataList = []

                # 过滤大模型反馈中的'(For the remaining tasks'及后续内容
                content = results[0]['response'].content.split('(For the remaining tasks')[0]

                print("*********************************")
                print(content)
                print("*********************************")

                if content.startswith('Based on'):
                    responseList = content.split('\n\n')[1:]
                else:
                    responseList = content.split('\n\n')
                # 保证末尾是output的实例输出
                if not re.findall(patternOut, responseList[-1], re.MULTILINE | re.IGNORECASE):
                    responseList = responseList[:-1]

                pattern = r'Task.*'

                for index, task in enumerate(responseList):
                    if task.startswith('Output') or task.startswith('output'):
                        inAndOutList.append(responseList[index - 1] + responseList[index])

                    matches = re.findall(pattern, task)
                    if len(matches) == 1:
                        # 创建一个新的字典，而不是修改原始的temp引用
                        temp = results[0]['response'].copy()  # 使用.copy()来创建一个浅拷贝
                        strIns = matches[0].strip('**')[6:]
                        temp.content = strIns  # 更新值
                        print(strIns + "-----------------")
                        metadataList.append(temp)

                print(len(inAndOutList))
                print("****************************************")
                print(len(metadataList))

                if len(inAndOutList) == len(metadataList):
                    for i in range(len(inAndOutList)):
                        data = [task for task in batch if fuzzy_match(task['instruction'], metadataList[i].content)]
                        if len(data) == 0:
                            continue
                        print("**********************")
                        print(metadataList[i].content)
                        data[0]["instance_metadata"] = metadataList[i]
                        if results[0]["response"] is not None:
                            data[0]["raw_instances"] = inAndOutList[i]
                        else:
                            data[0]["raw_instances"] = ""
                        data = OrderedDict(
                            (k, str(data[0][k])) for k in \
                                ["instruction", "raw_instances", "instance_metadata", "instruction_metadata",
                                "most_similar", "avg_similarity_score"]
                            )
                        fout.write(json.dumps(data, ensure_ascii=False) + "\n")

            progress_bar.update(len(batch))
