import os
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def train_loader(path, batchsize, numworkers=4):
    trainset = datasets.CIFAR100(path, 
                                train=True, 
                                transform=transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                ]),
                                download=True)
    return data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=numworkers)

def test_loader(path, batchsize, numworkers=4):
    testset = datasets.CIFAR100(path, 
                                train=False, 
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                ]),
                                download=True)
    return data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=numworkers)