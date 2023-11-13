import torch

x1 = torch.ones((3,4),dtype=torch.float16).cuda()
x2 = torch.ones((3,4),dtype=torch.bfloat16).cuda()
o = x1-x2
print(o,o.dtype)