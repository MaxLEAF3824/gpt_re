import torch
from typing import Dict, List
import random
import hashlib
import requests
import json
import os
from treelib import Tree
import gpustat
import sys
import datetime
import openai
import concurrent.futures
from tqdm.auto import tqdm
from copy import deepcopy
import logging

def multithread_query_chatgpt(inputs: List[Dict], thread_num=1, max_round=10, temperature=1.0, **kwargs):
    all_answers = []
    round = 1
    while len(inputs) > 0 or round > max_round:
        def get_output(input):
            query_config = dict(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": input}],
                temperature=temperature,
                **kwargs
            )
            response = openai.ChatCompletion.create(**query_config)
            return response.choices[0].message['content']
        chatgpt_answers = []
        error_inputs = []
        with concurrent.futures.ThreadPoolExecutor(thread_num) as executor:
            future_to_input = {executor.submit(get_output, i['query_input']): i for i in inputs}
            for future in tqdm(concurrent.futures.as_completed(future_to_input),total=len(inputs)):
                input = future_to_input[future]
                try:
                    query_output = future.result()
                    new_input = deepcopy(input)
                    new_input['query_output'] = query_output
                    chatgpt_answers.append(new_input)
                except Exception as exc:
                    error_inputs.append(input)
                    logging.info(f'An exception occurred: {exc}')
        logging.info(f"round: {round} error_inputs: {len(error_inputs)}")
        all_answers.extend(chatgpt_answers)
        inputs = error_inputs
        round += 1
    return all_answers

def print_struct(data):
    def build_tree(data, tree, parent=None):
        for key, value in data.items() if isinstance(data, dict) else enumerate(data):
            if isinstance(value, list):
                node = tree.create_node(tag='list', parent=parent)
                build_tree(value, tree, parent=node.identifier)
            elif isinstance(value, tuple):
                node = tree.create_node(tag='tuple', parent=parent)
                build_tree(list(value), tree, parent=node.identifier)
            elif isinstance(value, dict):
                node = tree.create_node(tag='dict', parent=parent)
                build_tree(value, tree, parent=node.identifier)
            elif isinstance(value, torch.Tensor):
                node = tree.create_node(tag=f'torch.Tensor({list(value.shape)} device={value.device})', parent=parent)
            else:
                node = tree.create_node(tag=f'{type(value).__name__}', parent=parent)
            if isinstance(data, dict):
                node.tag = f'"{key}": {node.tag}'
        return tree

    tree = Tree()
    tree.create_node(tag='root', identifier=0)
    build_tree(data, tree, parent=0)
    tree.show()
    
def get_free_gpus():
    free_gpu_indexs = []
    gpus = gpustat.new_query()
    for gpu in gpus:
        if gpu.utilization <= 10 and gpu.memory_used < 5000:
            free_gpu_indexs.append(gpu.index)
    return free_gpu_indexs

class BaiduTrans:
    baidu_api = "https://fanyi-api.baidu.com/api/trans/vip/translate"
    
    def __init__(self):
        self.appid = "123"
        self.key = "123"
        sk_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "trans_sk.json")
        if os.path.exists(sk_path):
            sk = json.load(open(sk_path))
            self.appid = sk['app_id']
            self.key = sk['secret_key']


    def generateSignature(self, query):
        salt = str(random.randint(0, 999999))
        string1 = self.appid + query + salt + self.key
        sign = hashlib.md5(string1.encode(encoding='UTF-8')).hexdigest()
        return salt, sign

    def query(self, q, lang_to='zh', lang_from="auto"):
        # print(f"query:{q}")
        salt, sign = self.generateSignature(q)
        req_data = {
            "q": q,
            "from": lang_from,
            "to": lang_to,
            "appid": self.appid,
            "salt": salt,
            "sign": sign
        }
        response = requests.post(self.baidu_api, data=req_data)
        # print(f"response:{response.json()}")
        result = ""
        for res in response.json()["trans_result"]:
            result = result + res['dst'] + '\n'
        return result[:-1]

class Stdout2File:
    """
    Redirect stdout to file
    """
    def __init__(self, filename, mode='w'):
        self.filename = filename
        self.original_stdout = None
        self.mode = mode

    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = self
        self.file = open(self.filename, self.mode, buffering=1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        self.file.close()

    def write(self, text):
        now = datetime.datetime.now().strftime('[LINE][%H:%M:%S.%f]')
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if line.strip():
                if i == 0:
                    self.file.write(f'{now} {line}\n')
                else:
                    self.file.write(f'{" "*len(now)} {line}\n')

    def flush(self):
        self.file.flush()

class PaddingSide:
    """Context manager that change padding side to left or right."""

    def __init__(self, tokenizer, padding_side='left'):
        self.tokenizer = tokenizer
        self.new_padding_side = padding_side
        self.ori_padding_side = self.tokenizer.padding_side
        
    def __enter__(self, *args, **kwargs):
        self.tokenizer.padding_side = self.new_padding_side

    def __exit__(self, *args, **kwargs):
        self.tokenizer.padding_side = self.ori_padding_side

class LoadWoInit:
    """Context manager that disable parameter initialization."""

    def __init__(self):
        self.constant_ = torch.nn.init.constant_
        self.zeros_ = torch.nn.init.zeros_
        self.ones_ = torch.nn.init.ones_
        self.uniform_ = torch.nn.init.uniform_
        self.normal_ = torch.nn.init.normal_
        self.kaiming_uniform_ = torch.nn.init.kaiming_uniform_
        self.kaiming_normal_ = torch.nn.init.kaiming_normal_

    def __enter__(self, *args, **kwargs):
        torch.nn.init.constant_ = lambda *args, **kwargs: None
        torch.nn.init.zeros_ = lambda *args, **kwargs: None
        torch.nn.init.ones_ = lambda *args, **kwargs: None
        torch.nn.init.uniform_ = lambda *args, **kwargs: None
        torch.nn.init.normal_ = lambda *args, **kwargs: None
        torch.nn.init.kaiming_uniform_ = lambda *args, **kwargs: None
        torch.nn.init.kaiming_normal_ = lambda *args, **kwargs: None

    def __exit__(self, *args, **kwargs):
        torch.nn.init.constant_ = self.constant_
        torch.nn.init.zeros_ = self.zeros_
        torch.nn.init.ones_ = self.ones_
        torch.nn.init.uniform_ = self.uniform_
        torch.nn.init.normal_ = self.normal_
        torch.nn.init.kaiming_uniform_ = self.kaiming_uniform_
        torch.nn.init.kaiming_normal_ = self.kaiming_normal_