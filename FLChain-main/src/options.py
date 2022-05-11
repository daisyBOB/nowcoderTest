#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--global_round', type=int, default=10,
                        help="number of global round")
    parser.add_argument('--idx', type=int, default=1,
                        help="idx of users: idx")
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")

    parser.add_argument('--model', type=str, default='mlp', help='model name')

    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')

    args = parser.parse_args()
    return args
