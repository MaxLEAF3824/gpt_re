from transformers import AutoTokenizer, AutoModelForCausalLM
# from model import *

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("/home/cs/yangyuchen/guoyiqiu/my_models/gpt2").cuda()
    tok = AutoTokenizer.from_pretrained("/home/cs/yangyuchen/guoyiqiu/my_models/gpt2")
    output = model.generate(tok("hello world",return_tensors='pt')['input_ids'].cuda(),max_new_tokens=1)
    print('output: ', tok.batch_decode(output[0]))
    # mt = LLM.from_pretrained(model_name="/home/cs/yangyuchen/guoyiqiu/my_models/gpt2").cuda()
    # hook_configs = [LLMHookConfig(module_name='block', layer=i) for i in range(mt.n_layer)]
    # with LLMHooker(mt, hook_configs) as hooker:
    #     out_text = mt.generate("hello world", max_new_tokens=10)
    #     print('out_text: ', out_text)
    #     print(hooker.hooks[0].outputs[0].shape)
    # print(hooker.hooks[0].outputs[0].shape)