import torch
from torch import nn

torch.device('cpu')

x = torch.tensor([1,2,3])
x.device
# net.to(device='cuda:0')