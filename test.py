import json
import re
import os
import pylcs


default_tmp = ('性别', '年龄', '入院时间', '出院时间', '门诊诊断', '入院诊断', '出院诊断', '入院时主要症状及体征', '主要化验结果', '特殊检查及重要会诊', '病程与治疗结果', '合并症', '出院时情况', '出院后建议', '治疗结果', "主治医师")
tmp_orders = [(0,       1,      2,          3,      4,          5,          6,          7,                  8,              9,              10,             11,     12,         13,         14,         15),
              (0,       1,      2,          3,      4,          5,          6,          7,                  10,              8,              9,             11,     12,         13,         14,         15)]
    
specaial_variants = dict(
    入院时间=['入院日期',],
    出院时间=['出院日期','死亡时间'],
    门诊诊断=['门诊诊',],
    入院诊断=['入院',],
    出院诊断=['死亡诊断'],
    入院时主要症状及体征=['入院情况'],
    主要化验结果=['主要结果','检验结果'],
    特殊检查及重要会诊=['特殊检查结果','特殊检验及重要会诊','动态心电图结果','术中冰冻结果','术中冰冻','病理结果'],
    病程与治疗结果=['诊疗经过','经过','病程及治疗结果'],
    出院时情况=['院时情况',"死亡时情况"],
    出院后建议=['出院后用药及建议','出院后用药建议','其他'],
    治疗结果=['治果'],
)

variants = lambda key :[key] + specaial_variants.get(key, []) + [(k[:i] + "\n" + k[i:]).strip() for k in ([key] + specaial_variants.get(key, [])) for i in range(1,len(k))]
is_checkout = lambda text: all(any(kv in text for kv in variants(key)) for key in default_tmp)


data = json.load(open("/home/cs/yangyuchen/guoyiqiu/ninth/ninth_data.json"))
checkout_data = []
num_error = 0
not_checkout = 0
for d in data:
    for emr in d['emrs']:
        checkout_id = d['zid']+ "/" +emr['filename']
        try:
            if not emr['texts']:
                continue
            
            texts = [re.sub(r'第(\s)*\d+(\s)*页','',t) for t in emr['texts']]
            text = ' '.join([t.replace(os.path.commonprefix(texts),'') for t in texts]) if len(texts) > 1 else texts[0]
            
            if not is_checkout(text):
                if ("出院小结" in checkout_id and "小时入出院记录" not in text):
                    not_checkout += 1
                    # print(checkout_id)
                    # print(text)
                continue
            
            tmp = []
            for key in default_tmp:
                for kv in variants(key):
                    if kv in text:
                        tmp.append((kv,key))
                        break
            
            for order in tmp_orders:
                new_tmp = [tmp[i] for i in order]
                new_tmp_str = ''.join([t[0] for t in new_tmp])
                if pylcs.lcs_sequence_length(new_tmp_str,text) == len(new_tmp_str):
                    tmp = new_tmp
                    break
            
            info_dict = {}
            for (kv,k),(nkv,nk) in zip(tmp[:-1],tmp[1:]):
                value = re.search(f"{kv}(.*?){nkv}",text, re.DOTALL).group(1).replace("：","").strip()
                info_dict[k] = value
            
            full_checks = []
            year, month, day, hour, minute, second = 2023, 12, 31, 23, 59, 59
            time_span = []
            for string in [info_dict['入院时间'],info_dict['出院时间']]:
                year = re.findall(r"\d\d\d\d",string)[0]
                # for re.findall(r"[^0-9]\d\d[^0-9]",string)
            info_dict['文件ID'] = checkout_id
            
            checkout_data.append(info_dict)
        except Exception as e:
            print(f"ERROR{e}\n{checkout_id}")
            print(tmp)
            num_error += 1
        break
print(f"num error:{num_error}")
print(f"not_checkout:{not_checkout}")
print(f"success num :{len(checkout_data)}")
json.dump(checkout_data,open("/home/cs/yangyuchen/guoyiqiu/ninth/checkout_data.json",'w'),ensure_ascii=False,indent=4)