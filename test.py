import json
import re
import os
import pylcs
from datetime import datetime
from tqdm.auto import tqdm


default_tmp = ('性别', '年龄', '入院时间', '出院时间', '门诊诊断', '入院诊断', '出院诊断', '入院时主要症状及体征', '主要化验结果', '特殊检查及重要会诊', '病程与治疗结果', '合并症', '出院时情况', '出院后建议', '治疗结果', "主治医师")
tmp_orders = [(0,       1,        2,          3,        4,          5,          6,          7,                  8,              9,              10,             11,         12,          13,         14,         15),
              (0,       1,        2,          3,        4,          5,          6,          7,                  10,              8,              9,             11,         12,          13,         14,         15)]
    
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


data = json.load(open("../my_datasets/ninth/ninth_data.json"))
checkout_data = []
num_error = 0
not_checkout = 0
for d in tqdm(data):
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
            
            info_dict = dict(文件ID=checkout_id)
            for (kv,k),(nkv,nk) in zip(tmp[:-1],tmp[1:]):
                # print(f'kv:{kv} nkv:{nkv}')
                match = re.search(f"{kv}(.*?){nkv}", text, re.DOTALL)
                span = match.span()
                # print(f'span:{span}')
                text = text[span[0]:]
                info_dict[k] = match.group(1).replace("：","").strip()
            
            advice = info_dict['出院后建议']
            advice = re.sub("预约[\s\S]*","",advice)
            advice = re.sub("下次来院时间[\s\S]*","",advice)
            advice = re.sub("健康宣教[\s\S]*","",advice)
            advice = re.sub("请输入出院后建议[\s\S]*","",advice).strip()
            info_dict['出院后建议'] = advice
            
            related_checks = []
            time_span = []
            default_time_stamp = ['2023','12','31','23','59']
            for string in [info_dict['入院时间'],info_dict['出院时间']]:
                # print('string: ', string)
                time_stamp = re.findall("[\d]+",string)
                # print('time_stamp: ', time_stamp)
                time_stamp += default_time_stamp[len(time_stamp):]
                time_stamp = ' '.join(time_stamp)
                time_span.append(datetime.strptime(time_stamp, "%Y %m %d %H %M"))
            # print('time_span: ', time_span)
            
            for check in d['checks']:
                check_time = datetime.strptime(check['info']['报告时间'], "%Y/%m/%d %H:%M")
                if time_span[0] < check_time < time_span[1]:
                    check_dict = {}
                    check_dict['header'] = check['info']['标本名称']
                    check_dict['values'] = {}
                    for c in check['values']:
                        if c.get("异常标识"):
                            check_dict['values'][c['检验项']] = '[正常]' if c["异常标识"] == "N" else '[异常]'
                        elif c.get("结果标识"):
                            check_dict['values'][c['药品名称']] = '[正常]' if c["结果标识"] == "Z" else '[异常]'
                        else:
                            print(c)
                    related_checks.append(check_dict)
            info_dict['完整化验结果'] = related_checks
            checkout_data.append(info_dict)
        except Exception as e:
            print(f"ERROR{e}\n{checkout_id}")
            num_error += 1
        break
print(f"num error:{num_error}")
print(f"not_checkout:{not_checkout}")
print(f"success num :{len(checkout_data)}")
json.dump(checkout_data, open("../my_datasets/ninth/checkout_data.json",'w'), ensure_ascii=False)