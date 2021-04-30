#%%
%matplotlib inline
import torch
import matplotlib.pyplot as plt

# from d2l import torch as d2l

#%%
"""'relu(x)'"""
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
plot1 = plt.plot(x.detach(), y.detach(),'-g',label = 'v',lw =10,ms =15)

# %%
"""grad of relu"""
y.backward(torch.ones_like(x), retain_graph=True)
plt.plot(x.detach(), x.grad)
# %%
"""'sigmoid(x)"""
y = torch.sigmoid(x)
plt.plot(x.detach(), y.detach())
#%%
# Clear out previous gradients
"""'grad of sigmoid'"""
x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
plt.plot(x.detach(), x.grad)

#%%
""" tanh(x)"""
y = torch.tanh(x)
plt.plot(x.detach(), y.detach())

##############################################################################
#################### 4.2 Implementation of MPLP  #############################
##############################################################################

#%%
import torch
from torch import nn
# from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# %%
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(
    torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(
    torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]
# %%
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
# %%
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)  # Here '@' stands for matrix multiplication
    return (H @ W2 + b2)
# %%
loss = nn.CrossEntropyLoss()
# %%
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
# %%
d2l.predict_ch3(net, test_iter)
# %%

# %%

# %%

# %%
