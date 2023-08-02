import torch
import cv2
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np


VggNet = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


vgg16 = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(6, 64, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),  # relu1_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2,padding=0, dilation=1, ceil_mode=False),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),  # relu2_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128,128, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, kernel_size=3,stride=1),
    nn.ReLU(inplace=True),  # relu3_1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, kernel_size=3, stride=1),
    nn.ReLU(inplace=True),  # relu4_1
)

vgg_conv_list = [1,4,8,11,15,18,21,25]
vgg_model_conv_list = [0, 2, 5, 7, 10, 12, 14, 17]

class VGG(nn.Module):
    def __init__(self, options):
        super(VGG, self).__init__()
        # vgg_pad
        vgg_model = models.vgg16(pretrained=False)
        vgg_model.load_state_dict(torch.load(options.path))
        vgg_model = vgg_model.features
        vgg = vgg16
        weight0 = vgg_model[vgg_model_conv_list[0]].weight.repeat(1, 2, 1, 1)
        # bias0 = vgg_model[vgg_model_conv_list[0]].bias.repeat(1, 2)
        # print('weight0', weight0.shape)
        vgg[vgg_conv_list[0]].weight = nn.Parameter(weight0)
        vgg[vgg_conv_list[0]].bias = vgg_model[vgg_model_conv_list[0]].bias

        for i in range(1, 8):
            vgg[vgg_conv_list[i]].weight = vgg_model[vgg_model_conv_list[i]].weight
            vgg[vgg_conv_list[i]].bias = vgg_model[vgg_model_conv_list[i]].bias
        self.test = vgg[vgg_conv_list[7]].weight

        for p in self.parameters():
            p.requires_grad = True
        self.slice1 = vgg[:3]  # relu1_1
        self.slice2 = vgg[3:10]  # relu2_1
        self.slice3 = vgg[10:17]  # relu3_1
        self.slice4 = vgg[17:27]  # relu4_1
        self.classifier1 = nn.Linear(1024, 512)
        self.classifier2 = nn.Linear(512, 2)
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x):
        out = []
        x = self.slice1(x)
        out.append(x)
        x = self.slice2(x)
        out.append(x)
        x = self.slice3(x)
        out.append(x)
        # x = self.slice4(x)
        # out.append(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.classifier1(x)
        x = self.classifier2(x)
        return out, x






