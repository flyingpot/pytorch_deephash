import torch
import torch.nn as nn

from torchvision import datasets, models, transforms
from torch.autograd import Variable

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
transform = transforms.Compose(
    [transforms.CenterCrop(227),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = datasets.CIFAR10(root='./data', train=True, download=True,
                            transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=True, download=True,
                           transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
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

params = list(model.classifier._modules['6'].parameters()) + list(model.classifier._modules['7'].parameters()) + list(model.classifier._modules['8'].parameters())  
optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.5)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.
                  format(epoch, batch_idx * len(data), len(
                  trainloader.dataset), 100. * batch_idx / len(
                  trainloader), loss.data[0]))

for epoch in range(1, 11):
    train(epoch)
