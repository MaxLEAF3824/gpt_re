import random
import hashlib
import requests
import json


class BaiduTrans:
    baidu_api = "https://fanyi-api.baidu.com/api/trans/vip/translate"
    
    
    def __init__(self):
        
        sk = json.load(open('/mnt/workspace/guoyiqiu/coding/gpt_re/utils/trans_sk.json'))
        self.appid = sk['appid']
        self.key = sk['key']

    def generateSignature(self, query):
        salt = str(random.randint(0, 999999))
        string1 = self.appid + query + salt + self.key
        sign = hashlib.md5(string1.encode(encoding='UTF-8')).hexdigest()
        return salt, sign

    def query(self, q, lang_to, lang_from="auto"):
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
        result = ""
        for res in response.json()["trans_result"]:
            result = result + res['dst'] + '\n'
        return result[:-1]
