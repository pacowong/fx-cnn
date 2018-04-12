from algorithm.parameters import params
from utilities.stats.logger import Logger
from utilities.stats.network_visualizer import Visualizer
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, fcn_layers, conv_layers):
        super(Model, self).__init__()
        self._fcn_layers = fcn_layers
        self._conv_layers = conv_layers
        self.conv_layers = self.set_conv(conv_layers)
        self.fcn_layers = self.set_fcn(fcn_layers)

        if params['CUDA_ENABLED']:
            if self.conv_layers:
                self.conv_layers = nn.DataParallel(self.conv_layers).cuda()
            if self.fcn_layers:
                self.fcn_layers = nn.DataParallel(self.fcn_layers).cuda()

    def show_parameters(self):
        for q_name, q_params in self.conv_layers.named_parameters():
            print(q_name)
            print(q_params.data)
        for q_name, q_params in self.fcn_layers.named_parameters():
            print(q_name)
            print(q_params.data)
            
    def set_conv(self, conv_layers):
        # (output, kernel, padding, stride, pool?)
        conv = collections.OrderedDict()
        for idx, l in enumerate(conv_layers):
            input_channel = conv_layers[idx-1][0] if idx > 0 else params['INPUT_CHANNEL']
            output_channel, kernel, stride, pool = l[0], l[1], l[3], l[4]
            padding = l[2] if l[2] != None else kernel // 2

            conv['conv'+str(idx)] = nn.Conv2d(input_channel, output_channel, kernel_size=kernel, stride=stride, padding=padding)
            nn.init.xavier_uniform(conv['conv'+str(idx)].weight, gain=np.sqrt(2))
            conv['relu'+str(idx)] = nn.ReLU(inplace=True)
            if pool:
                conv['pool'+str(idx)] = nn.MaxPool2d(kernel_size=2, stride=2)
        return nn.Sequential(conv)

    def set_fcn(self, layers):
        fcn = collections.OrderedDict()
        for l in range(1, len(layers)):
            linear = nn.Linear(layers[l-1], layers[l])
            nn.init.xavier_uniform(linear.weight, gain=np.sqrt(2))
            fcn["fcn"+str(l)] = linear
            if l != len(layers) - 1:
                fcn["dp"+str(l)] = nn.Dropout(p=0.7)
                fcn["reluf"+str(l)] = nn.Tanh() if params['ACTIVATION'] == 'Tanh' else nn.ReLU(inplace=True)
        return nn.Sequential(fcn)

    def reinitialize_params(self):
        for l in self.conv_layers.module if params['CUDA_ENABLED'] else self.conv_layers:
            if hasattr(l, 'weight'):
                nn.init.xavier_uniform(l.weight, gain=np.sqrt(2))
        for l in self.fcn_layers.module if params['CUDA_ENABLED'] else self.fcn_layers:
            if hasattr(l, 'weight'):
                nn.init.xavier_uniform(l.weight, gain=np.sqrt(2))
        if params['DEBUG_NET']:
            for l in self.conv_layers.module if params['CUDA_ENABLED'] else self.conv_layers:
                if hasattr(l, 'weight'):
                    print("Layer {}: \t\t{}".format(str(l), l.weight.mean().data.cpu().numpy()))
            for l in self.fcn_layers.module if params['CUDA_ENABLED'] else self.fcn_layers:
                if hasattr(l, 'weight'):
                    print("Layer {}: \t\t{}".format(str(l), l.weight.mean().data.cpu().numpy()))

    def get_module(self):
        return self.model.modules()

class FCNModel(Model):
    def __init__(self, fcn_layers):
        super(FCNModel, self).__init__(fcn_layers, [])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fcn_layers(x)
        return x

class ConvModel(Model):
    def __init__(self, fcn_layers=[256, 10], conv_layers=[]):
        super(ConvModel, self).__init__(fcn_layers, conv_layers)

    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fcn_layers(x)
        return x

class Conv2Model(ConvModel):
    def __init__(self, fcn_layers=[256, 10], conv_layers=[]):
        conv_layers = [(64, 3, None, 1, True), (256, 3, None, 1, True)]
        super(Conv2Model, self).__init__(fcn_layers, conv_layers)

class AlexNetModel(ConvModel):
    def __init__(self, fcn_layers=[256, 10], conv_layers=[]):
        conv_layers = [
            (64, 11, None, 4, True),
            (192, 5, None, 1, True),
            (384, 3, None, 1, False),
            (256, 3, None, 1, False),
            (256, 3, None, 1, True)
        ]
        super(AlexNetModel, self).__init__(fcn_layers, conv_layers)

class AlexNetModel2(ConvModel):
    def __init__(self, fcn_layers=[320, 10], conv_layers=[]):
        conv_layers = [
            (80, 11, None, 4, True),
            (240, 5, None, 1, True),
            (480, 3, None, 1, False),
            (320, 3, None, 1, False),
            (320, 3, None, 1, True)
        ]
        super(AlexNetModel2, self).__init__(fcn_layers, conv_layers)
