import random
import hashlib
import requests
import json
import os

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
