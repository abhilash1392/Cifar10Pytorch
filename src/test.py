import dataModel 
from loadDataset import test_loader,y_test
import torch.nn as nn 
import torch.optim as optim 
from config import n_epochs
import numpy as np
import torch
from sklearn.metrics import accuracy_score



model = dataModel.load_model()
criterion = nn.CrossEntropyLoss()
model.load_state_dict(torch.load('../models/cifar_10_pretrained.pt'))

if __name__=="__main__":
    model.eval()
    y_pred = []
    test_loss = 0.0
    for data,target in test_loader:
        output = model(data)
        loss = criterion(output,target)
        _,pred = torch.max(output,1)
        y_pred.append(pred)
        test_loss += loss.item()*data.size(0)

    test_loss = test_loss/len(test_loader.sampler)
    print('Test Loss : {:.6f}'.format(test_loss))

    accuracy = accuracy_score(y_test,y_pred)
    print('The Accuracy of the model is : {:.6f}'.format(accuracy))
        



