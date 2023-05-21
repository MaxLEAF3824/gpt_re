from model import LLM

class FlowVisualizer:
    def __init__(self, mt : LLM):
        self.mt = mt

    def init_hook(self):
        self.mt