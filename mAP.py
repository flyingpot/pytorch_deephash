import os
import argparse

import numpy as np
from scipy.spatial.distance import hamming, cdist

from timeit import time

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

def load_data():
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
    return trainloader, testloader


def binary_output(dataloader):
    net = models.alexnet()
    net.classifier._modules['6'] = nn.Linear(4096, 48)
    net.classifier._modules['7'] = nn.Sigmoid()
    net.classifier._modules['8'] = nn.Linear(48, 10)
    net.load_state_dict(torch.load('./model/%d' %args.pretrained))
    new_classifier = nn.Sequential(*list(net.classifier.children())[:-1])
    net.classifier = new_classifier
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
    full_batch_output = torch.cuda.FloatTensor()
    full_batch_label = torch.cuda.LongTensor()
    net.eval()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        full_batch_output = torch.cat((full_batch_output, outputs.data), 0)
        full_batch_label = torch.cat((full_batch_label, targets.data), 0)
    return torch.round(full_batch_output), full_batch_label

def precision(trn_binary, trn_label, tst_binary, tst_label):
    trn_binary = trn_binary.cpu().numpy()
    trn_binary = np.asarray(trn_binary, np.int32)
    trn_label = trn_label.cpu().numpy()
    tst_binary = tst_binary.cpu().numpy()
    tst_binary = np.asarray(tst_binary, np.int32)
    tst_label = tst_label.cpu().numpy()
    # binary_len = tst_binary.shape[1]
    query_times = tst_binary.shape[0]
    trainset_len = train_binary.shape[0]
    AP = np.zeros(query_times)
    Ns = np.arange(1, trainset_len + 1)
    total_time_start = time.time()
    for i in range(query_times):
        print('Query ', i+1)
        # query_starttime = time.time()
        query_label = tst_label[i]
        query_binary = tst_binary[i,:]
        query_result = np.count_nonzero(query_binary != trn_binary, axis=1)    #don't need to divide binary length
        sort_indices = np.argsort(query_result)
        buffer_yes= np.equal(query_label, trn_label[sort_indices]).astype(int)
        P = np.cumsum(buffer_yes) / Ns
        AP[i] = np.sum(P * buffer_yes) /sum(buffer_yes)
        # print('query time ', time.time() - query_starttime)
    map = np.mean(AP)
    print(map)
    print('total query time = ', time.time() - total_time_start)



if os.path.exists('./result/train_binary') and os.path.exists('./result/train_label') and \
   os.path.exists('./result/test_binary') and os.path.exists('./result/test_label') and args.pretrained == 0:
    train_binary = torch.load('./result/train_binary')
    train_label = torch.load('./result/train_label')
    test_binary = torch.load('./result/test_binary')
    test_label = torch.load('./result/test_label')

else:
    trainloader, testloader = load_data()
    train_binary, train_label = binary_output(trainloader)
    test_binary, test_label = binary_output(testloader)
    if not os.path.isdir('result'):
        os.mkdir('result')
    torch.save(train_binary, './result/train_binary')
    torch.save(train_label, './result/train_label')
    torch.save(test_binary, './result/test_binary')
    torch.save(test_label, './result/test_label')


precision(train_binary, train_label, test_binary, test_label)