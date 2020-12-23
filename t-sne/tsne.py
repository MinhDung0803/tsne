import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from model import ft_net
from datetime import datetime
import matplotlib.pyplot as plt
import random

###load config###
#config variable values
data_dir = './data/'
ms = 1 #scale number
batchsize = 1
which_epoch = 49
name = 'ft_ResNet50' #name of network
gpu_ids = 0
stride = 2
nclasses = 751

print('Extracting features...')

# set gpu ids
torch.cuda.set_device(gpu_ids)
cudnn.benchmark = True

# Load Data
# We will use torchvision and torch.utils.data packages for loading the data.
data_transforms = transforms.Compose([
        transforms.Resize((256,128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

#image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
#dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,shuffle=False, num_workers=16) for x in ['gallery','query']}

image_datasets = datasets.ImageFolder( os.path.join(data_dir) ,data_transforms) 
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batchsize,shuffle=False, num_workers=16) 
use_gpu = torch.cuda.is_available()

# Load model
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network

# Extract feature
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        print(img)
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n,512).zero_().cuda()
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img) 
            ff += outputs
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
        #print(features)
    return features

data_path = image_datasets.imgs

# Load Collected data Trained model
model_structure = ft_net(nclasses, stride = stride)
model = load_network(model_structure)
# Remove the final fc layer and classifier layer
model.classifier.classifier = nn.Sequential()
# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()
# Measure performance - start time
a = datetime.now()
# Extract feature
with torch.no_grad():
    data_feature = extract_feature(model,dataloaders)
#print the data feature shape
print("Shape of data feature: ",data_feature.shape)
# Measure performance - end time
b = datetime.now()
c = b-a
print("Performance of extracting vectors :",c.total_seconds())
