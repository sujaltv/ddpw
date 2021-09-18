import torch

ckpt = torch.load('models/ckpt_99.pt')

print(ckpt['model'])