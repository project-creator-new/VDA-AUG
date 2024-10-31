import json

# 定义每份文件包含的JSON对象数量
objects_per_file = 500

# 初始化计数器和文件索引
count = 0
file_index = 6

# 打开原始大文件
with open('../data/test/machine_generated_instructions_2.jsonl', 'r') as large_file:
    # 创建一个新的文件用于写入
    new_file = open(f'split_file_{file_index}.json', 'w')

    # 逐行读取原始文件
    for line in large_file:
        # 将JSON字符串转换为Python对象
        try:
            json_object = json.loads(line)
            # 将对象写入新文件，并确保每个对象后面都有换行符
            new_file.write(json.dumps(json_object) + '\n')
            # 增加计数器
            count += 1

            # 检查是否达到了指定的对象数量
            if count == objects_per_file:
                # 关闭当前文件，准备写入下一个文件
                new_file.close()
                # 重置计数器，并增加文件索引
                count = 0
                file_index += 1
                # 创建下一个文件
                new_file = open(f'split_file_{file_index}.json', 'w')

        except json.JSONDecodeError as e:
            # 如果遇到解析错误，打印错误并跳过该行
            print(f"Error decoding JSON line: {e}")

# 关闭最后一个文件
new_file.close()

print("File splitting is complete.")