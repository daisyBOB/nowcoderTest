#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from utils import get_dataset, save_weight
from options import args_parser
from models import MLP, CNNMnist

if __name__ == '__main__':
    args = args_parser()
    train_dataset, test_dataset, user_groups = get_dataset(args)
    if args.model == 'cnn':
        global_model = CNNMnist()
    elif args.model == 'mlp':
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64,
                           dim_out=10)
    else:
        exit('Error: unrecognized model')
    print(global_model)
    global_model.train()
    global_weights = global_model.state_dict()
    swarmID = save_weight(global_weights)
    print("Global Swarm ID : {}".format(swarmID))
