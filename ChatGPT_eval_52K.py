import re
import logging
import requests
import json
from time import sleep
import matplotlib.pyplot as plt
from collections import Counter

FORMAT =  '%(asctime)s - %(name)s - %(filename)s[line:%(lineno)d] - %(levelname)s - %(message)s'
logging.basicConfig(filename="log/run.log", level=logging.INFO, format=FORMAT)
logger = logging.getLogger()


def get_answer(prompt,sys_prompt='You are a friendly and helpful assistant.'):
    # URL="http://190.92.202.203:20006/chat"
    URL="http://190.92.202.203:22102/chat"
    headers={'Content-Type':'application/json'}
    msgs=[{'role':'system','content':sys_prompt}]
    payload={
        # "model":"gpt-4",
        "model":"gpt-3.5-turbo",
        "temperature":0.8,
        "top_p":1,
        "n":1,
        "stream":False,
        "stop":None,
        "presence_penalty":0,
        "frequency_penalty":0,
        "logit_bias":{}
            }
    msgs.append({'role':"user","content":prompt})
    payload["messages"]=msgs
    res=''
    while 1:
        sleep(0.1)
        try:
            response=requests.request("POST",URL,headers=headers,data=json.dumps(payload))
            res=response.json()
            logger.info(res)
            if 'choices' not in res:
                logger.info(f"chatgpt result failure:{res}")
            else:
                answer=res["choices"][0]["message"]["content"]
                break
        except Exception as e:
            logger.info(f"chatgpt exception:{e}")
    return answer

from tqdm import tqdm

def parse_coach_infer_data(_data):
    _pattern = re.compile(r"\$\$\$ Response: (.*)")
    _match = _pattern.findall(_data)
    return _match[0]

def load_data(predict_dir, raw_data_path, cut_len=None):

    predict_data = []
    with open(f"{predict_dir}/generated_predictions.jsonl", 'r', encoding='utf-8') as f:
        for _row in f.readlines():
            predict_data.append(parse_coach_infer_data(_row))
    with open(raw_data_path, 'r', encoding="utf-8") as f:
        alpaca52k = json.load(f)
    alpaca52k_revised = [{"instruction": a["instruction"], "input": a["input"], "output": b} for a, b in zip(alpaca52k, predict_data)]
    if not cut_len:
        cut_len = len(alpaca52k_revised)
    alpaca52k_revised = alpaca52k_revised[:cut_len]
    alpaca52k = alpaca52k[:cut_len]

    return alpaca52k_revised, alpaca52k

def eval_dataset(_dataset, save_dir):
    system_prompt = "We would like to request your feedback on the performance of AI assistant in response to " \
                    "the instruction and the given input displayed following."
    user_prompt = "Please rate according to the accuracy of the response to the instruction and the input. " \
                  "Each assistant receives a score on a scale of 0 to 5. The score should be an integer number, " \
                  "where a higher score indicates higher level of the accuracy. " \
                  "Please first output a single line containing only integer number that indicating the scores. " \
                  "In the subsequent line, please provide a comprehensive explanation of your evaluation, " \
                  "avoiding any potential bias. \n\n"
    logger.info(f"dataset length under evaluation: {len(_dataset)}")
    as_ls = []
    # 51326 total 48511 left 16170 each
    # for ele in tqdm(_dataset[6434:]):
    for ele in tqdm(_dataset):
        triplet = "###Instruction: {instruction}\n\n###Input: {input}\n\n###Response: {output}\n\n".format_map(ele)
        eval_prompt = triplet + user_prompt
        answer_f=get_answer(eval_prompt,system_prompt)
        as_ls.append(answer_f)
    json.dump(as_ls, open(save_dir, 'w'))
    return as_ls

pattern = re.compile(r"^([0-9]*\.?[0-9]*)$")
def match_number(_input):
    _match = pattern.match(_input)
    if not _match:
        return 0
    return float(_match.group())

def strip_score(_raw):
    _raw = _raw.strip()
    if _raw[-2:] == "/5":
        _raw = _raw[:-2]

    return match_number(_raw)
def score_bucket(_score):
    _score = float(_score)
    if _score>=0 and _score<3.5:
        return 1
    elif _score<4:
        return 2
    elif _score<4.5:
        return 3
    elif _score<=5:
        return 4
    else:
        return -1

from functools import reduce
def statistic_analysis(scores, save_file):
    if not scores:
        return
    total_cnt = len(scores)
    result = {}
    for _score in scores:
        result[score_bucket(_score)] = result.get(score_bucket(_score), 0) + 1
    statistics = []
    statistics.append(f"scores:{scores}\n")
    statistics.append(f"result:{result}\n")
    statistics.append(f"[0, 0.35):{result.get(1,0)*100.0/total_cnt}%\n")
    statistics.append(f"[0.35, 0.4):{result.get(2,0)*100.0/total_cnt}%\n")
    statistics.append(f"[0.4, 0.45):{result.get(3,0)*100.0/total_cnt}%\n")
    statistics.append(f"[0.45 ,5]:{result.get(4,0)*100.0/total_cnt}%\n")
    logger.info(statistics)
    with open(save_file, 'w+', encoding="utf-8") as f:
        f.writelines(statistics)

def parse_scores(eval_list):
    return [_.split("\n")[0].strip() for _ in eval_list]

def build_scores_histgram(scores):
    # fig, ax = plt.subplots()
    # points = [0, 0.35, 0.4, 0.45, 0.5]
    # ax.hist(points)
    hist_data, bin_edges, _ = plt.hist(scores, bins=[0, 3, 3.5, 4, 4.5, 5])
    plt.bar(range(len(hist_data)), list(hist_data))
    plt.xticks(list(range(len(hist_data))), ['[0,3)', '[3,3.5)', '[3.5-4)', '[4,4.5)', '[4.5,5]'])
    plt.show()

def load_data_yilun():
    with open("preprocess/chatGPT_eval_coach720_alpaca52k_0_2815.json", 'r', encoding="utf-8") as f:
        r= json.load(f)
    return r

def statistic_analysis2(scores, file_path):
    cnt = Counter(scores)
    with open(file_path, 'w' ,encoding="utf-8") as f:
        f.writelines([str(cnt)])

if __name__ == '__main__':
    infer_data_dir = "infer_result/infer_520_from_train_720_steps_157_r3"
    sample_alpaca52k = "data/infer_520_from_train_720_raw.json"
    _data, _data_raw = load_data(infer_data_dir, sample_alpaca52k)
    # _data, _data_raw = _data[:15], _data_raw[:15]
    logger.info(f"len(_data):{len(_data)}")
    eval_data = eval_dataset(_data, f"{infer_data_dir}/chatGPT_eval.json")
    scores = parse_scores(eval_data)
    statistic_analysis(scores, f"{infer_data_dir}/chatGPT_eval_statistic.txt")
    statistic_analysis2(scores, f"{infer_data_dir}/chatGPT_eval_statistic_cnt.txt")

    eval_data = eval_dataset(_data_raw, f"{infer_data_dir}/chatGPT_eval_original.json")
    scores = parse_scores(eval_data)
    statistic_analysis(scores, f"{infer_data_dir}/chatGPT_eval_statistic_original.txt")
    statistic_analysis2(scores, f"{infer_data_dir}/chatGPT_eval_statistic_original_cnt.txt")

    # with open("infer_result/infer_520_from_train_720_steps_157_r3_gpt4/chatGPT_eval.json", 'r') as f:
    #     eval_data = json.load(f)
    #
    # scores = parse_scores(eval_data)
    # statistic_analysis(scores, f"{infer_data_dir}/chatGPT_eval_statistic.txt")
    # statistic_analysis2(scores, f"{infer_data_dir}/chatGPT_eval_statistic_cnt.txt")
    # # histgrams = build_scores_histgram(scores)
