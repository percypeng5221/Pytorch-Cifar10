'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import sys 
import time

from models import *
from utils import progress_bar

stdoutOrigin=sys.stdout 
sys.stdout = open("log.txt", "w")


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
global best_acc 
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=32, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')




# Training
def train(epoch, net):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    acc = 100.*correct/total
    print("Epoch: {epoch}  Train accuracy: {acc:.6f}".format(epoch = epoch, acc = acc))


def test(epoch, net):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    
    acc = 100.*correct/total
    print("Epoch: {epoch}  Test accuracy: {acc:.6f}".format(epoch = epoch, acc = acc))
    time_now = time.localtime()
    print("Date:{y:4d}/{m:2d}/{d:2d}  Time:{h:2d}:{min:2d}:{s:2d}".format(y = time_now.tm_year, m = time_now.tm_mon,\
        d = time_now.tm_mday, h = time_now.tm_hour, min = time_now.tm_min, s = time_now.tm_sec))
    # time_now = time.localtime
    if acc > best_acc:
    #     # print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

def check_net(net):
    timecheck1 = time.time()
    
    for epoch in range(start_epoch, start_epoch+100):
        train(epoch, net)
        test(epoch, net)
        scheduler.step()
        
    timecheck2 = time.time()
    print("Total time: {time:.2f}s".format(time = timecheck2 - timecheck1))
    print("Best accuracy: {acc:.2f}%".format(acc = best_acc))
    
# Model
print('==> Building model..')

best_acc = 0
print("ResNet18")        
net = ResNet18()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)    
check_net(net)

best_acc = 0
print("ResNet50")        
net = ResNet50()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)    
check_net(net)

best_acc = 0
print("MobileNet")        
net = MobileNet()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)    
check_net(net)

best_acc = 0
print("ResNet34")        
net = ResNet34()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)    
check_net(net)

best_acc = 0
print("ResNet101")        
net = ResNet101()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)    
check_net(net)

best_acc = 0
print("ResNet152")        
net = ResNet152()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)    
check_net(net)

best_acc = 0
print("MobileNetV2")        
net = MobileNetV2()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)    
check_net(net)
