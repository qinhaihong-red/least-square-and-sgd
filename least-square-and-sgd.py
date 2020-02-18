import torch 
import torch.nn as nn
import torch.optim as optim

X=torch.arange(1,7,dtype=torch.float).reshape(3,2)
t=torch.tensor([7,8,9],dtype=torch.float).reshape(3,-1)
fc=nn.Linear(2,1,bias=False)
E=nn.MSELoss()
opt=optim.SGD(fc.parameters(),lr=0.04,momentum=0.9)
sheduler=optim.lr_scheduler.MultiStepLR(opt,[10,30,60],0.5)


def train():
    for i in range(1000):
        opt.zero_grad()
        loss=E(fc(X),t)
        
        loss.backward()
        opt.step()
        sheduler.step()

        if i%100==0:
            print('round {} ,loss is:\n{}'.format(i,(fc(X)-t).sum()))
            print()
  
train()
print('*'*10)
print('sgd final parameters are:\n',fc.weight.data)

w=torch.matmul(torch.matmul(X.t(),X).inverse(),torch.matmul(X.t(),t))
print('lesat square w is:\n',w)
