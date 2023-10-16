import os
import time
import torch

from .model_interface import LLM
from .llm_utils import BaiduTrans, get_free_gpus
from dataset import *
import ipywidgets as widgets

torch.set_float32_matmul_precision('medium')
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

VICUNA_TEMPLATE = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n##USER:\n{}\n\n##ASSISTANT:\n"

class LLMPanel(widgets.VBox):
    def __init__(self, model_list, chat_template=None):
        super(LLMPanel, self).__init__()
        self.translator = BaiduTrans()
        self.free_gpus = get_free_gpus()
        self.chat_template = chat_template if chat_template else VICUNA_TEMPLATE
        
        # model dropdown
        self.mt_dropdown = widgets.Dropdown(options=model_list, description='Model:', disabled=False,)
        self.mt = None
        
        # setup button
        self.setup_btn = widgets.Button(description="Setup everything", disabled=False,)
        self.setup_btn.on_click(self.setup_llm)

        # switch deivce
        self.device_tbtn = widgets.ToggleButtons(options=['cpu', f'cuda',], disabled=False,)
        self.device_tbtn.observe(self.switch_device, names='value')

        # free gpu list
        self.free_gpus_dropdown = widgets.Dropdown(options=self.free_gpus, description='Free GPUs:', disabled=False,)
        
        # switch precision
        self.precision_tbtn = widgets.ToggleButtons(options=['half', 'float'], disabled=False,)
        self.precision_tbtn.observe(self.switch_precision, names='value')

        # max new token slider
        self.mnt_slider = widgets.IntSlider(value=64,min=1,max=2048,step=1,description='new token:',disabled=False,)
        
        # temperature slider
        self.tem_slider = widgets.FloatSlider(value=1.0,min=0.0,max=10.0,step=0.1,description='temperature:',disabled=False,)
        
        # sample checkbox
        self.sample_checkbox = widgets.Checkbox(value=False,description='do sample',disabled=False,)
        
        # input and output textarea
        self.input_textarea = widgets.Textarea(value='',description='Input:',layout=widgets.Layout(width='30%', height='250px'),disabled=False)
        self.output_textarea = widgets.Textarea(value='',description='Output:',layout=widgets.Layout(width='30%', height='250px'),disabled=False)

        # submit button
        self.submit_btn = widgets.Button(description="generate",disabled=False,)
        self.submit_btn.on_click(self.generate)
        
        # translate button
        self.translate_btn = widgets.Button(description="translate",disabled=False,)
        self.translate_btn.on_click(self.translate)

        # chat mode checkbox
        self.chat_checkbox = widgets.Checkbox(value=False,description='chat mode',disabled=False,)
        
        # pannel layout
        self.control_panel = widgets.HBox([self.mt_dropdown, self.setup_btn, self.precision_tbtn, self.device_tbtn, self.free_gpus_dropdown])
        self.generate_panel = widgets.HBox([self.input_textarea, widgets.VBox([self.mnt_slider, self.tem_slider, self.sample_checkbox, self.chat_checkbox, self.submit_btn, self.translate_btn,]), self.output_textarea])
        self.children = [self.control_panel, self.generate_panel]
    
    def setup_llm(self, btn):
        time_st = time.time()
        btn.description = "Loading model..."
        try:
            self.mt = LLM.from_pretrained(
                model_name=self.mt_dropdown.value, 
                fp16=(self.precision_tbtn.value == "half"),
                )
            self.device_tbtn.value = 'cpu'
            print(f"Everything is ready. Time cost: {time.time() - time_st:.2f}s")
        except Exception as e:
            print(f"Loading model failed: {e}")
        finally:
            btn.description = "Setup everything"
    
    def switch_device(self, change):
        self.device_tbtn.disabled = True
        try:
            if change.new == 'cpu':
                self.mt.cpu()
                torch.cuda.empty_cache()
            else:
                self.mt.cuda(self.free_gpus_dropdown.value)
        except Exception as e:
            print(f"Switch device failed: {e}")
        finally:
            self.device_tbtn.disabled = False

    def switch_precision(self, change):
        self.precision_tbtn.disabled = True
        try:
            if self.mt is not None:
                self.mt.model.half() if change.new == 'half' else self.mt.model.float()
        except Exception as e:
            print(f"Switch failed: {e}")
        finally:
            self.precision_tbtn.disabled = False

    def generate(self, btn):
        btn.disabled = True
        self.submit_btn.description = "Generating..."
        input_text = self.chat_template.format(self.input_textarea.value) if self.chat_checkbox.value else self.input_textarea.value
        gen_kwargs = {
            "input_texts":input_text,
            "cut_input":True,
            "max_new_tokens":self.mnt_slider.value,
            "do_sample": self.sample_checkbox.value,
            "temperature": self.tem_slider.value,
        }
        try:
            self.output_textarea.value = self.mt.generate(**gen_kwargs)[0]
        except Exception as e:
            print(f"Generation failed: {e}") 
        finally:   
            btn.disabled = False
            self.submit_btn.description = "generate"


    def translate(self, btn):
        btn.disabled = True
        btn.description = "translating..."
        try:
            input_translated = self.translator.query(self.input_textarea.value.strip(), lang_to='zh') if self.input_textarea.value.strip() else ""
            output_translated = self.translator.query(self.output_textarea.value.strip(), lang_to='zh') if self.output_textarea.value.strip() else ""
            self.input_textarea.value += f"\n\n翻译:\n{input_translated}" if input_translated else ""
            self.output_textarea.value += f"\n\n翻译:\n{output_translated}" if output_translated else ""
        except Exception as e:
            print(f"Translation failed: {e}")
        finally:
            btn.description = "translate"
            btn.disabled = False