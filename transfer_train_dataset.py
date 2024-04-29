import json
import re


def print_line(lines):
    for line in lines:
        for _ in line.items():
            print(_)
        print("-"*20)
def extract_field(raw_data):
    field_map = {"Revised Instruction": "instruction",
                 "Revised Input": "input",
                 "Revised Response": "output"}
    result = []
    sort_data = sorted(raw_data, key= lambda r: r["Distance"])
    for row in sort_data:
        r = {new_k: row[old_k] for old_k, new_k in field_map.items()}
        result.append(r)
    return result

def format_data(raw_data, cut_off=None):
    result = []
    sort_data = sorted(raw_data, key= lambda r: r["Distance"])
    for row in sort_data:
        result += [{"instruction": "Improve the following content to be more specific, detailed with more logical steps and grammarly corrected.",
             "input": f"$$$ Instruction: {row['Raw Instruction']}\n"
                      f"$$$ Input: {row['Raw Input']}\n"
                      f"$$$ Response: {row['Raw Response']}\n",
             "output": f"$$$ Instruction: {row['Revised Instruction']}\n"
                       f"$$$ Input: {row['Revised Input']}\n"
                       f"$$$ Response: {row['Revised Response']}\n"}]
    if cut_off:
        result = result[:cut_off]
    return result


def build_dataset():
    with open("Expert Revision Dataset.json", 'r', encoding="utf-8") as f:
        raw_data = json.load(f)
    result = extract_field(raw_data)[:720]
    print(f"transformed data length:{len(result)}")
    with open("data/train_data_720.json", "w", encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

def pprint(_data):
    for row in _data:
        for k,v in row.items():
            print(k,v)

def save_file(_data, _path):
    with open(_path, "w", encoding='utf-8') as f:
        json.dump(_data, f, indent=4, ensure_ascii=False)
def build_dataset2():
    with open("Expert Revision Dataset.json", 'r', encoding='utf-8') as f:
        revised_data = json.load(f)
    with open("data/alpaca_data_en_52k.json") as f:
        original_data = json.load(f)
    formatted = format_data(revised_data, 720)
    save_file(formatted, "data/train_720_from_expert_revision_r3.json")
    print(len(formatted))


def check_bug(_row):
    return "Input:" in _row

def fix_bug(_data):
    _pattern = re.compile(r"(.*)Input:(.*)")
    result = []
    for _row in _data:
        if check_bug(_row["Revised Input"]):
            print(_row)
            _match = _pattern.findall(_row["Revised Input"])
            _row["Revised Instruction"] = f'{_row["Revised Instruction"]} input {_match[0][0]}'.strip()
            _row["Revised Input"] = f'{_match[0][1]}'.strip()
            print('fix revised: ', _row["Revised Instruction"], _row["Revised Input"])
        if check_bug(_row["Raw Input"]):
            print(_row)
            _match = _pattern.findall(_row["Raw Input"])
            _row["Raw Instruction"] = f'{_row["Raw Instruction"]} input {_match[0][0]}'.strip()
            _row["Raw Input"] = f'{_match[0][1]}'.strip()
            print('fix raw: ', _row["Raw Instruction"], _row["Raw Input"])

        result.append(_row)
    return result

def fix_expert_data():
    with open("Expert Revision Dataset.json", 'r', encoding="utf-8") as f:
        raw_data = json.load(f)
    revised_data = fix_bug(raw_data)
    with open("Expert Revision Dataset v2.json", "w", encoding="utf-8") as f:
        json.dump(revised_data, f, indent=4)
    # print(raw_data[284])
    # print(check_bug(raw_data[284]))

if __name__ == '__main__':
    # fix_expert_data()
    build_dataset2()
    # with open("data/train_data.json", "r", encoding="utf-8") as f:
    #     r = json.load(f)
    #     print(r[:3])