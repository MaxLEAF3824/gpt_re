from model import *

if __name__ == "__main__":
    mt = LLM.from_pretrained(model_name="/home/cs/yangyuchen/guoyiqiu/my_models/gpt2").cuda()
    mt.vis_sentence("hello world this is a long sentence that is longer than 128 tokens. let's see what will happen. I hope it will be fine. good luck.", max_new_tokens=128)
    