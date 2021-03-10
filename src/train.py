import dataModel 
from loadDataset import train_loader,valid_loader
import torch.nn as nn 
import torch.optim as optim 
from config import n_epochs
import numpy as np
import torch

model = dataModel.load_model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr = 0.01)

train_loader= train_loader
valid_loader = valid_loader

n_epochs = n_epochs

if __name__=="__main__":
    valid_loss_min = np.Inf
    for epoch in range(1,n_epochs+1):
        train_loss=0.0
        valid_loss = 0.0
        model.train()
        for data,target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
        model.eval()
            with torch.no_grad():
            for data,target in valid_loader:
                output = model(data)
                loss = criterion(output,target)
                valid_loss += loss.item()*data.size(0)       
        train_loss = train_loss/len(train_loader.sampler)

        valid_loss = valid_loss/len(valid_loader.sampler)
        print('Epoch : {}  Train Loss : {:.6f} Valid Loss : {:.6f}'.format(epoch,train_loss,valid_loss))
        if valid_loss<=valid_loss_min:
            print('Validation Loss Decreased : {:6f} ---> {:.6f} . Saving model')
            torch.save(model.state_dict(),'../models/cifar_10_pretrained.pt')
            valid_loss_min=valid_loss

