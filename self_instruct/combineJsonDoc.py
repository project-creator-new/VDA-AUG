import json
import os

# 新文件的名称
new_combined_file = '../data/test/4k-combined_all_instructions.jsonl'

# 打开新文件用于写入
with open(new_combined_file, 'w') as combined_file:
    # 遍历文件编号1到8
    for i in range(1, 9):
        # 构造当前文件的名称
        file_name = f'split_file_{i}.json'
        output_path = os.path.join('../data/test/', file_name)
        # 打开当前文件
        with open(output_path, 'r') as file:
            # 逐行读取JSON对象
            for line in file:
                # 将JSON字符串转换为Python对象，然后再转换回字符串
                # 这样确保了即使原始文件中的JSON对象是经过压缩的，转换回字符串后也会是一行一个对象
                json_object = json.loads(line)
                json_string = json.dumps(json_object)
                # 将对象写入新文件，并确保每个对象后面都有换行符
                combined_file.write(json_string + '\n')

print(f"Combining files is complete. The combined file is '{new_combined_file}'.")
