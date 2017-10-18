import os
import shutil
import argparse

import torch
import torch.nn as nn

from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
#transform_train = transforms.Compose([
#    transforms.RandomCrop(32, padding=4),
#    transforms.RandomHorizontalFlip(),
#    transforms.ToTensor(),
#    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#])
#transform_test = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#])
parser = argparse.ArgumentParser(description='Deep Hashing')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
#lr参数，默认为0.01
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
#momentum参数，默认为0.5
parser.add_argument('--epoch', type=int, default=32, metavar='epoch',
                    help='epoch')
parser.add_argument('--pretrained', type=int, default=0, metavar='pretrained_model',
                    help='loading pretrained model(default = None)')
parser.add_argument('--bits', type=int, default=48, metavar='bts',
                    help='binary bits')
args = parser.parse_args()

best_acc = 0
start_epoch = 0
transform_train = transforms.Compose(
    [transforms.Scale(256),
     transforms.RandomCrop(227),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor()])
     # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test = transforms.Compose(
    [transforms.Scale(227),
     transforms.ToTensor()])
     # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
trainset = datasets.CIFAR10(root='./data', train=True, download=True,
                            transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False, download=True,
                           transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=True, num_workers=2)

alexnet_model = models.alexnet(pretrained=True)

#for param in alexnet_model.parameters():
#    param.requires_grad = False

alexnet_model.classifier._modules['6'] = nn.Linear(4096, args.bits)
alexnet_model.classifier._modules['7'] = nn.Sigmoid()
alexnet_model.classifier._modules['8'] = nn.Linear(args.bits, 10)

net = alexnet_model

use_cuda = torch.cuda.is_available()
if use_cuda:
    net.cuda()
#    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
#    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

ignored_params = list(net.classifier._modules['6'].parameters()) + list(net.classifier._modules['7'].parameters()) + list(net.classifier._modules['8'].parameters())
base_params = list(net.features.parameters()) + list(net.classifier._modules['0'].parameters()) + list(net.classifier._modules['1'].parameters()) + list(net.classifier._modules['2'].parameters()) + list(net.classifier._modules['3'].parameters()) + list(net.classifier._modules['4'].parameters()) + list(net.classifier._modules['5'].parameters())
#optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.5)
#optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
optimizer = torch.optim.SGD([{'params': ignored_params}, {'params': base_params, 'lr': 0.1*args.lr}], lr=args.lr, momentum=args.momentum, weight_decay=0.0005)
#optimizer = torch.optim.SGD(net.classifier._modules['6'].parameters(), lr=args.lr, momentum=args.momentum)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[32, 50], gamma=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

def test():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct / total
    if acc > best_acc and not args.pretrained:
        print('Saving')
        if not os.path.isdir('model'):
            os.mkdir('model')
        torch.save(net.state_dict(), './model/%d' %acc)
        best_acc = acc
#     # Save checkpoint.
#     acc = 100.*correct/total
#     if acc > best_acc:
#         print('Saving..')
#         state = {
# #            'net': net.parameters() if use_cuda else net,
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, './checkpoint/ckpt.t7')
#         best_acc = acc

if args.pretrained:
    net.load_state_dict(torch.load('./model/%d' %args.pretrained))
    test()
else:
    if os.path.isdir('model'):
        shutil.rmtree('model')
    for epoch in range(start_epoch, start_epoch+args.epoch):
        val_loss = train(epoch)
        test()
        scheduler.step(val_loss)
