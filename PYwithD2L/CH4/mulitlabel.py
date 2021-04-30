#%%
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random

batch_size = 128
train_dataset = datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ]))
test_dataset =  datasets.MNIST('./data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ]))
for i in range(len(train_dataset.targets)):
    if(train_dataset.targets[i]>0):
        train_dataset.targets[i]=1
    elif(train_dataset.targets[i]==0):
        train_dataset.targets[i]=0

for i in range(len(test_dataset.targets)):
    if(test_dataset.targets[i]>0):
        test_dataset.targets[i]=1
    elif(test_dataset.targets[i]==0):
        test_dataset.targets[i]=0
        

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)  

next(iter(train_loader))[0].shape
#%%
model = nn.Sequential(
   nn.Linear(784,2)
)

import torch.optim as optim
from sklearn.metrics import accuracy_score

lr = 0.005 
optimizer = optim.SGD(model.parameters(), lr=lr) 

cls_loss = nn.CrossEntropyLoss()

list_epoch = [] 
list_train_loss = []
list_acc = []
list_acc_epoch = []

epoch = 30
for i in range(epoch):
    
    # ====== Train ====== #
    train_loss = 0
    model.train() 
    
    for input_X, true_y in train_loader:
        optimizer.zero_grad() # [21.01.05 오류 수정] 매 Epoch 마다 .zero_grad()가 실행되는 것을 매 iteration 마다 실행되도록 수정했습니다. 

        input_X = input_X.squeeze()
        input_X = input_X.view(-1, 784)
        pred_y = model(input_X)

        loss = cls_loss(pred_y.squeeze(), true_y)
        loss.backward() 
        optimizer.step() 
        train_loss += loss.detach().numpy()

    train_loss = train_loss / len(train_loader)
    list_train_loss.append(train_loss)
    list_epoch.append(i)
    
    #test set
    correct = 0
    model.eval()
    
    with torch.no_grad():
        for input_X, true_y in test_loader:
            input_X = input_X.squeeze()
            input_X = input_X.view(-1, 784)
            pred_y = model(input_X).max(1, keepdim=True)[1].squeeze()
            correct += pred_y.eq(true_y).sum()

        acc = correct.numpy() / len(test_loader.dataset)
        list_acc.append(acc)
        list_acc_epoch.append(i)
    
    print('Epoch: {}, Train Loss: {}, Test Acc: {}%'.format(i, train_loss, acc*100))
# %%
