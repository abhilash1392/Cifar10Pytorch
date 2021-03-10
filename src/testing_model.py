import numpy as np
from numpy.lib.npyio import load 
from loadDataset import classes
from PIL import Image 
from dataModel import load_model 
import torch 
import matplotlib.pyplot as plt
from torchvision import transforms

model = load_model()
model.load_state_dict(torch.load('../input/cifar_10_pretrained.pt',map_location=torch.device('cpu')))
transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])

def predict(file):
    img = Image.open(file,'r')
    img = transform(img)
    img = torch.unsqueeze(img,0)
    output = model(img)
    _,pred = torch.max(output,1)
    predicted_class = pred
    print(predicted_class)

filename = '../input/airplane-flight.jpg'
predict(filename)