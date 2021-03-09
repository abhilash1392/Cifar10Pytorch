import torchvision.models as models 
import torch.nn as nn

resnet = models.resnet18(pretrained=True)


def load_model():
    for param in resnet.parameters():
        param.requires_grad = False
    num_features = resnet.fc.in_features

    resnet.fc = nn.Linear(num_features,10)

    return resnet 


