import numpy as np 
import pickle 
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader 
from PIL import Image 
from sklearn.model_selection import train_test_split
import os
import torch

classes=['airplane',
  'automobile',
  'bird',
  'cat',
  'deer',
  'dog',
  'frog',
  'horse',
  'ship',
  'truck']

def batchToNumpy(file):
    with open(file,'rb') as f:
        d = pickle.load(f,encoding='latin1')
    x = d['data']
    y = d['labels']
    x = np.array(x)
    x = x.reshape(10000,3,32,32)
    y = np.array(y)
    return x,y 

filename = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
X = []
Y = []
for f in filename:
    x,y = batchToNumpy(os.path.join('../input/cifar-10-batches-py',f))
    X.append(x)
    Y.append(y)
X = np.concatenate(X)
Y = np.concatenate(Y)

x_test,y_test = batchToNumpy('../input/cifar-10-batches-py/test_batch')
x_train,x_valid,y_train,y_valid = train_test_split(X,Y,test_size=0.2)

class ClassificationDataset(Dataset):
    def __init__(self,data,label,transform=None):
        self.data = data 
        self.label = label 
        self.transform = transform 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image = self.data[item]
        image = Image.fromarray(np.transpose(image,(1,2,0)))
        if self.transform:
            image = self.transform(image)
        image = np.array(image,dtype='float32')
        label = self.label[item]
        image = torch.tensor(image,dtype=torch.float)
        label = torch.tensor(label,dtype=torch.long)
        return image,label 

transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_data = ClassificationDataset(data=x_train,label=y_train,transform=transform)
valid_data = ClassificationDataset(data=x_valid,label=y_valid,transform=transform)
test_data = ClassificationDataset(data = x_test,label=y_test,transform=transform)

train_loader = DataLoader(train_data,batch_size=64,shuffle=True)

valid_loader = DataLoader(valid_data,batch_size=64,shuffle=False)

test_loader = DataLoader(test_data,batch_size=1,shuffle=False)









