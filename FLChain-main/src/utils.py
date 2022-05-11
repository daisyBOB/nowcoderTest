#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# utils.py包存放一些常用的公共方法,提供一些实用的方法和数据结构
import copy    # 拷贝文件模块
import torch
from torchvision import datasets, transforms     # 视觉数据集
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal


def get_dataset(args):
    """
    Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    data_dir = '../data/mnist/'

    apply_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)

    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

    if args.iid:
        user_groups = mnist_iid(train_dataset, args.num_users)
    else:
        if args.unequal:
            user_groups = mnist_noniid(train_dataset, args.num_users)
        else:
            user_groups = mnist_noniid_unequal(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def get_global_weight():
    """
        Get the previous node model weight or global model weight
    """
    # get weight from swarm and save to local path
    w_file_path = None
    w = torch.load(w_file_path)
    return w


def get_local_weight(swarmID):
    """
        Get the local weight from swarm
    """
    w_file_path = None
    w = torch.load(w_file_path)
    return w


def save_weight(model_dict):
    """
        Save model weight to swarm
        Return swarm id
    """
    path = "../weight/local_weight.tar"
    torch.save(model_dict, path)
    # upload local_weight.tar to swarm and get the swarm id
    swarmID = ""
    return swarmID


def output(args, accuracy_list, loss_list):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    plt.figure()
    plt.title('{}_Client'.format(args.idx))
    plt.plot(range(len(loss_list)), loss_list, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('fed_{}_idx:{}_loss.png'.format(args.model, args.idx))

    plt.figure()
    plt.title('{}_Client'.format(args.idx))
    plt.plot(range(len(accuracy_list)), accuracy_list, color='r')
    plt.ylabel('Training accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('fed_{}_idx:{}_acc.png'.format(args.model, args.idx))


if __name__ == '__main__':
    from options import args_parser

    args = args_parser()
    train_dataset, test_dataset, user_groups = get_dataset(args)
