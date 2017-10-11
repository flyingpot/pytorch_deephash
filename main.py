import torch
import torch.nn as nn

from torchvision import datasets, models

trainset = datasets.CIFAR10(root='./data', train=True, download=True)
trainloader = torch.utils.data.Dataloader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=True, download=True)
testloader = torch.utils.data.Dataloader(testset, batch_size=100,
                                         shuffle=True, num_workers=2)

alexnet_model = models.alexnet(pretrained=True)

for param in alexnet_model.parameters():
    param.requires_grad = False

alexnet_model.classifier._modules['6'] = nn.Linear(4096, 48)
alexnet_model.classifier._modules['7'] = nn.Sigmoid()
alexnet_model.classifier._modules['8'] = nn.Linear(48, 1000)

model = alexnet_model
model.cuda()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.
                  format(epoch, batch_idx * len(data), len(
                  trainloader.dataset), 100. * batch_idx / len(
                  trainloader), loss.data[0]))

for epoch in range(1, 11):
    train(epoch)