import argparse
import os
from timeit import time

import numpy as np
import torch
import torch.optim.lr_scheduler
from torchvision import datasets, transforms
from tqdm import tqdm

from net import AlexNetPlusLatent

parser = argparse.ArgumentParser(description='Deep Hashing evaluate mAP')
parser.add_argument('--pretrained', type=float, default=0, metavar='pretrained_model',
                    help='loading pretrained model(default = None)')
parser.add_argument('--bits', type=int, default=48, metavar='bts',
                    help='binary bits')
args = parser.parse_args()


def load_data():
    transform_train = transforms.Compose(
        [transforms.Resize(227),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    transform_test = transforms.Compose(
        [transforms.Resize(227),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True,
                                transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=False, num_workers=0)

    testset = datasets.CIFAR10(root='./data', train=False, download=True,
                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=0)
    return trainloader, testloader


def binary_output(dataloader):
    net = AlexNetPlusLatent(args.bits)
    net.load_state_dict(torch.load('./model/{}'.format(args.pretrained)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Use device: " + str(device))
    net.to(device)
    full_batch_output = torch.cuda.FloatTensor()
    full_batch_label = torch.cuda.LongTensor()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs)
            full_batch_output = torch.cat((full_batch_output, outputs.data), 0)
            full_batch_label = torch.cat((full_batch_label, targets.data), 0)
        return torch.round(full_batch_output), full_batch_label


def evaluate(trn_binary, trn_label, tst_binary, tst_label):
    classes = np.max(tst_label) + 1
    for i in range(classes):
        if i == 0:
            tst_sample_binary = tst_binary[np.random.RandomState(seed=i).permutation(np.where(tst_label == i)[0])[:100]]
            tst_sample_label = np.array([i]).repeat(100)
            continue
        else:
            tst_sample_binary = np.concatenate([tst_sample_binary, tst_binary[np.random.RandomState(seed=i).permutation(np.where(tst_label==i)[0])[:100]]])
            tst_sample_label = np.concatenate([tst_sample_label, np.array([i]).repeat(100)])
    query_times = tst_sample_binary.shape[0]
    trainset_len = trn_binary.shape[0]
    AP = np.zeros(query_times)
    precision_radius = np.zeros(query_times)
    Ns = np.arange(1, trainset_len + 1)
    sum_tp = np.zeros(trainset_len)
    total_time_start = time.time()
    with tqdm(total=query_times, desc="Query") as pbar:
        for i in range(query_times):
            query_label = tst_sample_label[i]
            query_binary = tst_sample_binary[i, :]
            query_result = np.count_nonzero(query_binary != trn_binary, axis=1)    # don't need to divide binary length
            sort_indices = np.argsort(query_result)
            buffer_yes = np.equal(query_label, trn_label[sort_indices]).astype(int)
            P = np.cumsum(buffer_yes) / Ns
            precision_radius[i] = P[np.where(np.sort(query_result) > 2)[0][0]-1]
            AP[i] = np.sum(P * buffer_yes) / sum(buffer_yes)
            sum_tp = sum_tp + np.cumsum(buffer_yes)
            pbar.set_postfix({'Average Precision': '{0:1.5f}'.format(AP[i])})
            pbar.update(1)
    pbar.close()
    mAP = np.mean(AP)
    precision_at_k = sum_tp / Ns / query_times
    index = [100, 200, 400, 600, 800, 1000]
    index = [i - 1 for i in index]
    print('precision at k:', precision_at_k[index])
    print('precision within Hamming radius 2:', np.mean(precision_radius))
    map = np.mean(AP)
    print('mAP:', map)
    print('Total query time:', time.time() - total_time_start)


if __name__ == "__main__":
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

    train_binary = train_binary.cpu().numpy()
    train_binary = np.asarray(train_binary, np.int32)
    train_label = train_label.cpu().numpy()
    test_binary = test_binary.cpu().numpy()
    test_binary = np.asarray(test_binary, np.int32)
    test_label = test_label.cpu().numpy()

    evaluate(train_binary, train_label, test_binary, test_label)


