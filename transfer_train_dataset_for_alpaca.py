import json
import re


def print_line(lines):
    for line in lines:
        for _ in line.items():
            print(_)
        print("-"*20)
def extract_field(raw_data):
    field_map = {"instruction": "instruction",
                 "input": "input",
                 "reference response": "output"}
    result = []
    for row in raw_data:
        r = {new_k: row[old_k] for old_k, new_k in field_map.items()}
        result.append(r)
    return result

def format_data(raw_data, cut_off=None):

    result = []
    for row in raw_data:
        result += [{"instruction": "Improve the following content to be more specific, detailed with more logical steps and grammarly corrected.",
             "input": f"###Instruction: {row['instruction']}\n"
                      f"###Input: {row['input']}\n"
                      f"###Response: {row['output']}\n",
             "output": ""}]
    if cut_off:
        result = result[:cut_off]
    return result


def pprint(_data):
    for row in _data:
        for k,v in row.items():
            print(k,v)

def save_file(_data, _path):
    with open(_path, "w", encoding='utf-8') as f:
        json.dump(_data, f, indent=4, ensure_ascii=False)
def build_dataset():
    with open("data/alpaca_data_en_52k.json") as f:
        original_data = json.load(f)
    formatted = format_data(original_data)
    # pprint(formatted[:3])
    print(len(formatted))
    save_file(formatted, "data/train_52k_alpaca_original.json")

if __name__ == '__main__':

    build_dataset()
