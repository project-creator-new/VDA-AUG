

def count_json_objects(file_path):
    count = 0
    with open(file_path, 'r') as file:
        for line in file:
            count += 1
    return count


num_objects = count_json_objects('../data/test/GLM4-Air-all-instances.jsonl')
print(num_objects)