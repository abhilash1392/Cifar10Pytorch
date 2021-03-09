# Cifar 10 Classification Model using Pytorch 

In this repository we look at how to first build a image classifier using Convolutional neural network and also, we dip our toes in the module of transfer learning.




## To download data
```
mkdir input 
cd input 
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvfz cifar-10-python.tar.gz
```

## Download pretrained model 

```
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True) # This for getting weights 

```