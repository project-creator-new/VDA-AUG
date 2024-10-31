import json


def get_and_write_lines_from_jsonl(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile):
            if i >= 2000:
                break
            json_object = json.loads(line)
            json.dump(json_object, outfile)
            outfile.write('\n')


get_and_write_lines_from_jsonl('origin_all_instances.jsonl',
                                         '2000_origin_all_instances.jsonl')

