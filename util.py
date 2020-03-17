import os
import numpy as np
import time
import argparse

from mpi4py import MPI
from math import ceil
from random import Random
import networkx as nx

import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.multiprocessing import Process
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import torchvision.models as models

from models import *

import GraphPreprocess 

class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234, isNonIID=False):
        self.data = data 
        self.partitions = [] 
        rng = Random() 
        rng.seed(seed) 
        data_len = len(data) 
        indexes = [x for x in range(0, data_len)] 
        rng.shuffle(indexes) 
         
 
        for frac in sizes: 
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

        if isNonIID:
            self.partitions = __getNonIIDdata__(self, data, sizes, seed)

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

    def __getNonIIDdata__(self, data, sizes, seed):
        labelList = data.train_labels
        rng = Random()
        rng.seed(seed)
        a = [(label, idx) for idx, label in enumerate(labelList)]
        # Same Part
        labelIdxDict = dict()
        for label, idx in a:
                labelIdxDict.setdefault(label,[])
                labelIdxDict[label].append(idx)
        labelNum = len(labelIdxDict)
        labelNameList = [key for key in labelIdxDict]
        labelIdxPointer = [0] * labelNum
        partitions = [list() for i  in range(len(sizes))]
        eachPartitionLen= int(len(labelList)/len(sizes))
        majorLabelNumPerPartition = ceil(labelNum/len(partitions))
        basicLabelRatio = 0.4

        interval = 1
        labelPointer = 0

        #basic part
        for partPointer in range(len(partitions)):
            requiredLabelList = list()
            for _ in range(majorLabelNumPerPartition):
                requiredLabelList.append(labelPointer)
                labelPointer += interval
                if labelPointer > labelNum - 1:
                    labelPointer = interval
                    interval += 1
            for labelIdx in requiredLabelList:
                start = labelIdxPointer[labelIdx]
                idxIncrement = int(basicLabelRatio*len(labelIdxDict[labelNameList[labelIdx]]))
                partitions[partPointer].extend(labelIdxDict[labelNameList[labelIdx]][start:start+ idxIncrement])
                labelIdxPointer[labelIdx] += idxIncrement

        #random part
        remainLabels = list()
        for labelIdx in range(labelNum):
            remainLabels.extend(labelIdxDict[labelNameList[labelIdx]][labelIdxPointer[labelIdx]:])
        rng.shuffle(remainLabels)
        for partPointer in range(len(partitions)):
            idxIncrement = eachPartitionLen - len(partitions[partPointer])
            partitions[partPointer].extend(remainLabels[:idxIncrement])
            rng.shuffle(partitions[partPointer])
            remainLabels = remainLabels[idxIncrement:]
        return partitions

def partition_dataset(rank, size, args):
    print('==> load train data')
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root=args.datasetRoot, 
                                                train=True, 
                                                download=True, 
                                                transform=transform_train)
 
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(trainset, partition_sizes, isNonIID=False)
        partition = partition.use(rank)
        train_loader = torch.utils.data.DataLoader(partition, 
                                                batch_size=args.bs, 
                                                shuffle=True, 
                                                pin_memory=True)
 
        print('==> load test data')
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root=args.datasetRoot, 
                                            train=False, 
                                            download=True, 
                                            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, 
                                                batch_size=64, 
                                                shuffle=False, 
                                                num_workers=size)

    if args.dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        trainset = torchvision.datasets.CIFAR100(root=args.datasetRoot, 
                                                train=True, 
                                                download=True, 
                                                transform=transform_train)
 
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(trainset, partition_sizes, isNonIID=False)
        partition = partition.use(rank)
        train_loader = torch.utils.data.DataLoader(partition, 
                                                batch_size=args.bs, 
                                                shuffle=True, 
                                                pin_memory=True)
 
        print('==> load test data')
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        testset = torchvision.datasets.CIFAR100(root=args.datasetRoot, 
                                            train=False, 
                                            download=True, 
                                            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, 
                                                batch_size=64, 
                                                shuffle=False, 
                                                num_workers=size)
 

    elif args.dataset == 'imagenet':
        datadir = args.datasetRoot
        traindir = os.path.join(datadir, 'CLS-LOC/train/')
        #valdir = os.path.join(datadir, 'CLS-LOC/')
        #testdir = os.path.join(datadir, 'CLS-LOC/')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
 
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(train_dataset, partition_sizes, isNonIID=False)
        partition = partition.use(rank)
 
        train_loader = torch.utils.data.DataLoader(
            partition, batch_size=args.bs, shuffle=True,
             pin_memory=True)
        '''
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.bs, shuffle=False,
            pin_memory=True)
        val_loader = None
        '''
        test_loader = None

    if args.dataset == 'emnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.EMNIST(root=args.datasetRoot,
                                              split = 'balanced',
                                              train=True,
                                              download=True,
                                              transform=transform_train) 
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(train_dataset, partition_sizes, isNonIID=False)
        partition = partition.use(rank)
 
        train_loader = torch.utils.data.DataLoader(
            partition, batch_size=args.bs, shuffle=True,
             pin_memory=True)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        testset = torchvision.datasets.EMNIST(root=args.datasetRoot,
                                             split = 'balanced',
                                             train=False,
                                             download=True,
                                             transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, 
                                                batch_size=64, 
                                                shuffle=False, 
                                                num_workers=size)
 

    return train_loader, test_loader

def select_model(num_class, args):
    if args.model == 'VGG':
        model = vggnet.VGG(16, num_class)
    elif args.model == 'res':
        if args.dataset == 'cifar10':
            # model = large_resnet.ResNet18()
            model = resnet.ResNet(50, num_class)
        elif args.dataset == 'imagenet':
            model = models.resnet18()
    elif args.model == 'wrn':
        model = wrn.Wide_ResNet(28,10,0,num_class)
    elif args.model == 'mlp':
        if args.dataset == 'emnist':
            model = MLP.MNIST_MLP(47)
    return model

def select_graph(graphid):
    # pre-defined base network topologies
    # you can add more by extending the list
    Graphs =[ 
             # graph 0: 
             # 8-node erdos-renyi graph as shown in Fig. 1(a) in the main paper
             [[(1, 5), (6, 7), (0, 4), (2, 3)], 
              [(1, 7), (3, 6)], 
              [(1, 0), (3, 7), (5, 6)], 
              [(1, 2), (7, 0)], 
              [(3, 1)]],

             # graph 1:
             # 16-node gemetric graph as shown in Fig. A.3(a) in Appendix
             [[(4, 8), (6, 11), (7, 13), (0, 12), (5, 14), (10, 15), (2, 3), (1, 9)], 
              [(11, 13), (14, 2), (5, 6), (15, 3), (10, 9)], 
              [(11, 8), (2, 5), (13, 4), (14, 3), (0, 10)], 
              [(11, 5), (15, 14), (13, 8)], 
              [(2, 11)]],

             # graph 2:
             # 16-node gemetric graph as shown in Fig. A.3(b) in Appendix
             [[(2, 7), (12, 15), (3, 13), (5, 6), (8, 0), (9, 4), (11, 14), (1, 10)], 
              [(8, 6), (0, 11), (3, 2), (5, 4), (15, 14), (1, 9)], 
              [(8, 3), (0, 6), (11, 2), (4, 1), (12, 14)], 
              [(8, 11), (6, 3), (0, 5)], 
              [(8, 2), (0, 3), (6, 7), (11, 12)], 
              [(8, 5), (6, 4), (0, 2), (11, 7)], 
              [(8, 15), (3, 7), (0, 4), (6, 2)], 
              [(8, 14), (5, 3), (11, 6), (0, 9)], 
              [(8, 7), (15, 11), (2, 5), (4, 3), (1, 0), (13, 6)], 
              [(12, 8)]],

             # graph 3:
             # 16-node gemetric graph as shown in Fig. A.3(c) in Appendix
             [[(3, 12), (4, 8), (1, 13), (5, 7), (9, 10), (11, 14), (6, 15), (0, 2)], 
              [(7, 14), (2, 6), (5, 13), (8, 10), (1, 15), (0, 11), (3, 9), (4, 12)], 
              [(2, 7), (3, 15), (9, 13), (6, 11), (4, 14), (10, 12), (1, 8), (0, 5)], 
              [(5, 14), (1, 12), (13, 8), (9, 4), (2, 11), (7, 0)], 
              [(5, 1), (14, 8), (13, 12), (10, 4), (6, 7)], 
              [(5, 9), (14, 1), (13, 3), (8, 2), (11, 7)], 
              [(5, 12), (14, 13), (1, 9), (8, 0)], 
              [(5, 2), (14, 10), (1, 3), (9, 8), (13, 15)], 
              [(5, 8), (14, 12), (1, 4), (13, 10)], 
              [(5, 3), (14, 2), (9, 12), (1, 10), (13, 4)], 
              [(5, 6), (14, 0), (8, 12), (1, 2)], 
              [(5, 15), (9, 14)], 
              [(11, 5)]],

             # graph 4:
             # 16-node erdos-renyi graph as shown in Fig 3.(b) in the main paper
             [[(2, 7), (3, 15), (13, 14), (8, 9), (1, 5), (0, 10), (6, 12), (4, 11)], 
             [(12, 11), (5, 6), (14, 1), (9, 10), (15, 2), (8, 13)], 
             [(12, 5), (11, 6), (1, 8), (9, 3), (2, 10)], 
             [(12, 14), (11, 9), (5, 15), (0, 6), (1, 7)], 
             [(12, 8), (5, 2), (11, 14), (1, 6)], 
             [(12, 15), (13, 11), (10, 5), (3, 14)], 
             [(12, 9)], 
             [(0, 12)]], 

             # graph 5, 8-node ring
             [[(0, 1), (2, 3), (4, 5), (6, 7)], 
              [(0, 7), (2, 1), (4, 3), (6, 5)]]

            ]
            
    return Graphs[graphid] 

def comp_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res 

class AverageMeter(object):
    """Computes and stores the average and current value"""
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

class Recorder(object):
    def __init__(self, args, rank):
        self.record_accuracy = list()
        self.record_timing = list()
        self.record_comp_timing = list()
        self.record_comm_timing = list()
        self.record_losses = list()
        self.record_trainacc = list()
        self.total_record_timing = list()
        self.args = args
        self.rank = rank
        self.saveFolderName = args.savePath + args.name + '_' + args.model
        if rank == 0 and os.path.isdir(self.saveFolderName)==False and self.args.save:
	        os.mkdir(self.saveFolderName)

    def add_new(self,record_time,comp_time,comm_time,epoch_time,top1,losses,test_acc):
        self.total_record_timing.append(record_time)
        self.record_timing.append(epoch_time)
        self.record_comp_timing.append(comp_time)
        self.record_comm_timing.append(comm_time)
        self.record_trainacc.append(top1)
        self.record_losses.append(losses)
        self.record_accuracy.append(test_acc)

    def save_to_file(self):
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-recordtime.log', self.total_record_timing, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-time.log',  self.record_timing, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-comptime.log',  self.record_comp_timing, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-commtime.log',  self.record_comm_timing, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-acc.log',  self.record_accuracy, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-losses.log',  self.record_losses, delimiter=',')
        np.savetxt(self.saveFolderName+'/dsgd-lr'+str(self.args.lr)+'-budget'+str(self.args.budget)+'-r'+str(self.rank)+'-tacc.log',  self.record_trainacc, delimiter=',')
        with open(self.saveFolderName+'/ExpDescription', 'w') as f:
            f.write(str(self.args)+ '\n')
            f.write(self.args.description + '\n')


def test(model, test_loader):
    model.eval()
    top1 = AverageMeter()
    # correct = 0
    # total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        outputs = model(inputs)
        acc1 = comp_accuracy(outputs, targets)
        top1.update(acc1[0], inputs.size(0))
    return top1.avg
