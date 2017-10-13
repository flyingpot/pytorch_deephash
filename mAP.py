import os
import argparse

import numpy as np

import torch
import torch.nn as nn

from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler

parser = argparse.ArgumentParser(description='Deep Hashing evaluate mAP')
parser.add_argument('--pretrained', type=int, default=0, metavar='pretrained_model',
                    help='loading pretrained model(default = None)')
args = parser.parse_args()

net = models.alexnet()
net.classifier._modules['6'] = nn.Linear(4096, 48)
net.classifier._modules['7'] = nn.Sigmoid()
net.classifier._modules['8'] = nn.Linear(48, 10)
net.load_state_dict(torch.load('./model/%d' %args.pretrained))
new_classifier = nn.Sequential(*list(net.classifier.children())[:-1])
net.classifier = new_classifier
best_acc = 0
start_epoch = 0
transform_train = transforms.Compose(
    [transforms.Scale(227),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test = transforms.Compose(
    [transforms.Scale(227),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
trainset = datasets.CIFAR10(root='./data', train=True, download=True,
                            transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=False, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False, download=True,
                           transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

use_cuda = torch.cuda.is_available()
if use_cuda:
    net.cuda()

test_output = torch.cuda.FloatTensor()
def test():
    global best_acc
    global test_output
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        # test_output[100*batch_idx:100*(batch_idx+1), :] = torch.Tensor(outputs)
        test_output = torch.cat((test_output, outputs.data), 0)
        # print(outputs.data)
    print(torch.round(test_output))

test()