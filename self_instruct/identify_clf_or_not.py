import os
import json
import random
import tqdm
import re
import argparse
import pandas as pd
from collections import OrderedDict
from gpt3_api import make_requests as make_gpt3_requests



random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--template", type=str, default="template_1", help="Which template to use.")
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--num_instructions",
        type=int,
        help="if specified, only generate instance input for this many instructions",
    )
    parser.add_argument(
        "--template", 
        type=str, 
        default="template_1", 
        help="Which template to use. Currently only `template_1` is supported.",
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
        default=5,
        help="The number of requests to send in a batch."
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    with open(os.path.join(args.batch_dir, "origin_combined_all_instructions.jsonl")) as fin:  # 读取模型生成的新指令所在的文件
        lines = fin.readlines()
        if args.num_instructions is not None:
            lines = lines[:args.num_instructions]
    # 在该文件中加载已有判断结果的指令
    output_path = os.path.join(args.batch_dir, f"is_clf_or_not_{args.engine}.jsonl")  # 生成文件路径
    existing_requests = {}
    if os.path.exists(output_path):
        with open(output_path) as fin:
            for line in tqdm.tqdm(fin):
                try:
                    data = json.loads(line)
                    existing_requests[data["instruction"]] = data
                except:
                    pass
        print(f"Loaded {len(existing_requests)} existing requests")

    progress_bar = tqdm.tqdm(total=len(lines))
    with open(output_path, "w") as fout:
        for batch_idx in range(0, len(lines), args.request_batch_size):
            batch = [json.loads(line) for line in lines[batch_idx: batch_idx + args.request_batch_size]]
            # 如果该batch中的全部指令都已经在上述文件中存在了已判断得出的结果，则直接生成字典格式数据，写入该上述文件
            if all(d["instruction"] in existing_requests for d in batch):
                for d in batch:
                    data = existing_requests[d["instruction"]]
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "is_classification"]
                        )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                prompts = [d["instruction"].strip() + "\n" + "Is it classification?" for d in batch]
                results = make_gpt3_requests(
                    engine=args.engine,
                    prompts=prompts,
                    max_tokens=1024,
                    temperature=0.1,
                    top_p=0.1,

                    system_ins='You are now acting as an expert in distinguishing between classification and '
                               'non-classification task instructions.You just need to respond to each instruction '
                               'with “yes” or “no”.')
                separators = re.compile(r',\s+|\n')
                responseList = re.split(separators, results[0]['response'].content)

                if responseList[0].startswith('Is it'):
                    del responseList[0]
                    if responseList[0] == '':
                        del responseList[0]

                pattern = r'\b(no|yes)\b'  # 过滤一下判定结果，只保留yes和no
                for i in range(len(batch)):
                    data = batch[i]
                    # 输入输出要对应，所以此循环以batch中的元素为基准进行迭代
                    if responseList[i] is not None:
                        matches = re.findall(pattern, responseList[i].strip('[]'), re.IGNORECASE)
                        data["is_classification"] = str(matches[0])
                    else:
                        data["is_classification"] = ""
                    data = {
                        "instruction": data["instruction"],
                        "is_classification": data["is_classification"]
                    }
                    data = OrderedDict(
                        (k, data[k]) for k in \
                            ["instruction", "is_classification"]
                        )
                    fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            progress_bar.update(len(batch))
