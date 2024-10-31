import os
import json
import argparse
import glob
import re
import random
import tqdm
import pandas as pd

random.seed(123)

if __name__ == "__main__":

    file_path = '../data/test/test_new_index7/2000_origin_all_instances.jsonl'
    pattern = r'output.*'

    instances = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            instances.append(json.loads(line))

    output_file_path = '../data/test/test_new_index7/2000-origin-fineTuning.jsonl'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # 为Llama-factory所需数据集做准备
    Lla_factory_data = []

    with open(output_file_path, 'w', encoding='utf-8') as fouW:
        for task in instances:
            outputTask = re.findall(pattern, task['raw_instances'], re.DOTALL | re.IGNORECASE)[0]
            inputTask = task['raw_instances'].replace(outputTask, '')

            ins = dict()
            ins['messages'] = []
            ins['messages'].append(
                {"role": "system", "content": "Now, as an expert in completing vulnerability descriptions."})
            ins['messages'].append(
                {"role": "user", "content": task['instruction'] + '\n' + inputTask})
            ins['messages'].append(
                {"role": "assistant", "content": task['instruction'] + '\n' + outputTask}
            )

            Lla_factory_data_entity = dict()
            Lla_factory_data_entity['instruction'] = task['instruction']
            Lla_factory_data_entity['input'] = inputTask
            Lla_factory_data_entity['output'] = outputTask
            Lla_factory_data.append(Lla_factory_data_entity)

            fouW.write(json.dumps(ins, ensure_ascii=False) + ",")

    with open('../data/test/test_new_index7/2000_origin_fineTuning.json', 'w', encoding='utf-8') as Lla:
        Lla.write("[" + "\n")
        for entity in Lla_factory_data:
            Lla.write(json.dumps(entity, ensure_ascii=False) + ",")
        Lla.write("]" + "\n")