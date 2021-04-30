#%%
import numpy as np
import torch 
from torch.utils import data
from d2l import torch as d2l

npx.np_set()
# %%
true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w,true_b,1000)
#%%
def load_array(data_array,batch_size,is_train = True):
    dataset = data.TensorDataset(*data_array)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

batch_size = 10
data_iter = load_array((features,labels),batch_size)
# %%
next(iter(data_iter))
# %%
nn.Squential()
net.add(nn.Dense(1))
#%%
from mxnet import init

net.initialize(init.Normal(sigma=0.01))
#%%
loss = gluon.loss.L2Loss()

#%%
torch.manual_seed(0)
a=torch.ones([6])/6
torch.
torch.multinomial(1,a).sample()
# %%
torch.manual_seed(0)
torch.multinomial(a,1)
# %%
torch.manual_seed(0)
torch.randn ((5,)) 
# %%
