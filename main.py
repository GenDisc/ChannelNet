import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from channelnet import ChannelNet
from dataloader import train_loader, test_loader


parser = argparse.ArgumentParser()
# Directory
parser.add_argument('--data_path', default='./data')
parser.add_argument('--save_path', default='./save')
# Hyperparameters
parser.add_argument('--max_epoch', default=200)
parser.add_argument('--batchsize', default=128)
parser.add_argument('--lr', default=0.1)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--weight_decay', default=0.001)
parser.add_argument('--lr_decay_interval', default=20)
# Utility parameters
parser.add_argument('--log_interval', default=50)
parser.add_argument('--num_workers', default=16)
parser.add_argument('--gpu', default=1)
parser.add_argument('--version', default=3)



def main():
    opt = parser.parse_args()
    model = ChannelNet(v=opt.version, num_class=100).cuda(opt.gpu)
    print(model)

    # if opt.gpu is not None:
        # model = nn.parallel.DataParallel(model, device_ids=opt.gpu)

    criterion = nn.CrossEntropyLoss().cuda(opt.gpu)        
    optimizer = optim.SGD(model.parameters(), 
                            lr=opt.lr, 
                            momentum=opt.momentum, 
                            weight_decay=opt.weight_decay)

    trainloader = train_loader(opt.data_path, opt.batchsize, opt.num_workers)
    testloader = test_loader(opt.data_path, opt.batchsize, opt.num_workers)

    best_acc = 0.0
    for epoch in range(opt.max_epoch):
        if epoch > 30:
            adjust_learning_rate(optimizer, epoch, opt)
        train(trainloader, model, optimizer, criterion, epoch, opt)
        acc = test(testloader, model, criterion, opt)
        if acc > best_acc:
            best_acc = acc
            state = {
                'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict()
            }
            torch.save(state, os.path.join(opt.save_path, '%d_checkpoint.ckpt'%epoch))
        print(' Best accuracy so far : %.4f%%'%best_acc)


def train(train_loader, model, optimizer, criterion, epoch, opt):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    model.train()
    for i, (x, target) in enumerate(tqdm(train_loader, desc='##Training CIFAR100')):
        x = x.cuda(opt.gpu)
        target = target.cuda(opt.gpu)

        out = model(x)
        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = accuracy(out, target)
        acc_meter.update(acc, out.size(0))
        loss_meter.update(loss, out.size(0))

        if i%opt.log_interval==0:
            string = '[%d/%d][%d/%d] Loss : %.4f / %.4f, Acc : %.4f%% / %.4f%%'% \
                (epoch, opt.max_epoch, i, len(train_loader), loss_meter.val, loss_meter.avg, 
                acc_meter.val*100.0, acc_meter.avg*100.0)
            tqdm.write(string)

def test(test_loader, model, criterion, opt):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (x, target) in enumerate(tqdm(test_loader, desc='##Evaluating CIFAR100')):
            x = x.cuda(opt.gpu)
            target = target.cuda(opt.gpu)

            out = model(x)
            loss = criterion(out, target)

            acc = accuracy(out, target)
            acc_meter.update(acc, out.size(0))
            loss_meter.update(loss, out.size(0))
    print('# Evaluation || Loss : %.4f, Acc : %.4f%%'% (loss_meter.avg, acc_meter.avg*100.0))
    return acc_meter.avg*100.0

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, opt):
    lr = opt.lr * (0.3 ** ((epoch - 30) // opt.lr_decay_interval))
    print('Learning rate : %f'%lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target):
    correct = 0
    _, pred = output.max(1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    acc = correct/target.size(0)
    return acc

if __name__=='__main__':
    main()