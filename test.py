from model import *
from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("/home/cs/yangyuchen/yushengliao/Medical_LLM/llama-2-7b-chat-hugging").cuda()
    