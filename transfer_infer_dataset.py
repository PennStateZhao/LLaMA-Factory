import json

def sort_data(raw_data, sort_field, cut_off=None):
    result = sorted(raw_data, key= lambda r: r[sort_field])
    if cut_off:
        result = result[:cut_off]
    return result

def extract_field(raw_data,
                  field_map=("Raw Instruction", "Raw Input", "Raw Response")):
    result = {}
    for idx, row in enumerate(raw_data):
        r = tuple(row[_] for _ in field_map)
        result[r] = idx
    return result

def collide_datasets(data1, data2):
    result = []
    for row in data1:
        try:
            result.append(data2[row])
        except Exception as e:
            print(f"Row is not matched: {row}")
            result.append(None)

    return result

def exclude_samples(_dataset, exclude_list):
    return [_ for idx,_ in enumerate(_dataset) if idx not in exclude_list]

def build_dataset(output_dir="./data", cut_length=720):
    with open("Expert Revision Dataset v2.json", 'r', encoding="utf-8") as f:
        raw_data = json.load(f)
    sorted_d = sort_data(raw_data, "Distance", cut_length)
    revised_data = extract_field(sorted_d)
    print(f"transformed data length:{len(revised_data)}")
    with open("data/alpaca_data_en_52k.json", 'r', encoding='utf-8') as f:
        alpaca52k = json.load(f)
    alpaca52k_data = extract_field(alpaca52k, ("instruction", "input", "output"))

    train_index = collide_datasets(revised_data, alpaca52k_data)
    sampled_data = exclude_samples(alpaca52k, train_index)
    format_d = format_data(sampled_data, cut_off=520)
    print(f"output dir:{output_dir}; for len:{len(format_d)}")
    save_file(format_d, f"{output_dir}/infer_520_from_train_720_r3.json")
    save_file(sampled_data, f"{output_dir}/infer_520_from_train_720_raw.json")

def format_data(raw_data, fields=("instruction", "input", "output"), cut_off=None):
    result = []
    for row in raw_data:
        result += [{"instruction": "Improve the following content to be more specific, detailed with more logical steps and grammarly corrected.",
             "input": f"$$$ Instruction: {row[fields[0]]}\n"
                      f"$$$ Input: {row[fields[1]]}\n"
                      f"$$$ Response: {row[fields[2]]}\n",
                   "output": ""}]
    if cut_off:
        result = result[:cut_off]
    return result

def save_file(_data, _path):
    with open(_path, "w", encoding='utf-8') as f:
        json.dump(_data, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    build_dataset(cut_length=720)