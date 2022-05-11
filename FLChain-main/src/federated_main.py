#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch.utils.data import DataLoader
import torch
from utils import get_dataset, get_global_weight, get_local_weight, save_weight, output
from update import LocalUpdate
from options import args_parser
from models import MLP, CNNMnist


def update_local(args, model, local_weight, global_weight):
    """
    Test Federated Learning
    Return new_weight, accuracy, loss
    """
    model = model.load_state_dict(local_weight)
    train_dataset, test_dataset, user_groups = get_dataset(args)
    local_update = LocalUpdate(dataset=train_dataset, idxs=user_groups[args.idx])
    weight, loss = local_update.update_weights(
        model=model, epochs=args.local_ep)
    new_weight = torch.div((weight + global_weight), 2)
    accuracy = test_model(model)
    return new_weight, accuracy, loss


def test_model(model, test_dataset):
    model.eval()
    testloader = DataLoader(test_dataset, batch_size=64,
                            shuffle=False)
    correct, total = 0, 0
    for batch_idx, (images, labels) in enumerate(testloader):
        outputs = model(images)
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
    accuracy = correct / total
    return accuracy


if __name__ == '__main__':
    args = args_parser()
    train_dataset, test_dataset, user_groups = get_dataset(args)
    if args.model == 'cnn':
        local_model = CNNMnist()
    elif args.model == 'mlp':
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        local_model = MLP(dim_in=len_in, dim_hidden=64,
                          dim_out=10)
    else:
        exit('Error: unrecognized model')

    local_weight_swarmID = save_weight(local_model.state_dict())
    accuracy_list, loss_list = [], []
    for round in range(args.global_round):
        global_weight = get_global_weight()
        local_weight = get_local_weight(local_weight_swarmID)
        new_weight, accuracy, loss = update_local(args, local_model, local_weight, global_weight)
        local_weight_swarmID = save_weight(new_weight)
        accuracy_list.append(accuracy)
        loss_list.append(loss)
    output(accuracy_list, loss_list)
