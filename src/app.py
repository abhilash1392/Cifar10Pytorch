# Importing the libraries 
import uvicorn ## For ASGI 
from fastapi import FastAPI , File, UploadFile
import numpy as np
from numpy.lib.npyio import load 
from loadDataset import classes
from PIL import Image 
from dataModel import load_model 
import torch 
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable

model = load_model()
model.load_state_dict(torch.load('../models/cifar_10_pretrained.pt',map_location=torch.device('cpu')))
transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])

app = FastAPI()

@app.get('/')
def index():
    return {"message":"This is a cifar 10 recognizer"}

@app.post("/predict")
def predict_file(file:UploadFile = File(...)):
    img = Image.open(file.filename)
    img = transform(img)
    img = torch.unsqueeze(img,0)
    img = Variable(img)
    output = model(img)
    pred = output.data.numpy().argmax()
    predicted_class = classes[pred]

    return {"label":'{}'.format(predicted_class)}

if __name__=="__main__":
    uvicorn.run(app)
