import argparse
import math
import os
import shutil

import torch
import torch.nn as nn
import torch.optim.lr_scheduler
from torchvision import datasets, transforms
from tqdm import tqdm

from net import AlexNetPlusLatent

parser = argparse.ArgumentParser(description='Deep Hashing')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--epoch', type=int, default=128, metavar='epoch',
                    help='epoch')
parser.add_argument('--pretrained', type=int, default=0, metavar='pretrained_model',
                    help='loading pretrained model(default = None)')
parser.add_argument('--bits', type=int, default=48, metavar='bts',
                    help='binary bits')
parser.add_argument('--path', type=str, default='model', metavar='P',
                    help='path directory')
args = parser.parse_args()


def init_dataset():
    transform_train = transforms.Compose(
        [transforms.Resize(256),
         transforms.RandomCrop(227),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Compose(
        [transforms.Resize(227),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True,
                                transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=0)

    testset = datasets.CIFAR10(root='./data', train=False, download=True,
                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=True, num_workers=0)
    return trainloader, testloader


def train(epoch_num):
    print('\nEpoch: %d' % epoch_num)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    with tqdm(total=math.ceil(len(trainloader)), desc="Training") as pbar:
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            _, outputs = net(inputs)
            loss = softmaxloss(outputs, targets)
            optimizer4nn.zero_grad()
            loss.backward()
            optimizer4nn.step()
            train_loss += softmaxloss(outputs, targets).item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).sum()
            pbar.set_postfix({'loss': '{0:1.5f}'.format(loss), 'accurate': '{:.2%}'.format(correct.item() / total)})
            pbar.update(1)
    pbar.close()
    return train_loss / (batch_idx + 1)


def test():
    with torch.no_grad():
        test_loss = 0
        correct = 0
        total = 0
        with tqdm(total=math.ceil(len(testloader)), desc="Testing") as pbar:
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                _, outputs = net(inputs)
                loss = softmaxloss(outputs, targets)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).sum()
                pbar.set_postfix({'loss': '{0:1.5f}'.format(loss), 'accurate': '{:.2%}'.format(correct.item() / total)})
                pbar.update(1)
        pbar.close()
        acc = 100 * int(correct) / int(total)
        if epoch == args.epoch:
            print('Saving')
            if not os.path.isdir('{}'.format(args.path)):
                os.mkdir('{}'.format(args.path))
            torch.save(net.state_dict(), './{}/{}'.format(args.path, acc))


if __name__ == '__main__':
    torch.cuda.empty_cache()  # When using windows, this line is needed
    trainloader, testloader = init_dataset()
    net = AlexNetPlusLatent(args.bits)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Use device: " + str(device))
    net.to(device)
    softmaxloss = nn.CrossEntropyLoss().cuda()
    optimizer4nn = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer4nn, milestones=[args.epoch], gamma=0.1)
    best_acc = 0
    start_epoch = 1
    if args.pretrained:
        net.load_state_dict(torch.load('./{}/{}'.format(args.path, args.pretrained)))
        test()
    else:
        if os.path.isdir('{}'.format(args.path)):
            shutil.rmtree('{}'.format(args.path))
        for epoch in range(start_epoch, start_epoch + args.epoch):
            train(epoch)
            test()
            scheduler.step(epoch)

