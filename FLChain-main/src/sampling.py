#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# sampling 抽样
import numpy as np

# 独立同分布（数据集，用户数）
def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    '''-----初始化：每个客户端分配的数据量、用户字典、索引列表-----'''
    # （每个客户端分配的）物品数 = 数据集大小 / 用户数
    num_items = int(len(dataset) / num_users)
    # 初始化：用户字典、所有的索引 [0,1,2,...]
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    '''-----为每个客户端分配数据集-----'''
    # numpy.random.choice(a, size=None, replace=True, p=None):
    # 从a(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
    # replace:True表示可以取相同数字，False表示不可以取相同数字
    # 数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
    for i in range(num_users):
        # set() 函数创建一个无序不重复元素集
        # 客户端字典[客户端号] = {0,5,6,8  集合形式的数据下标}
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        # 更新索引集合 = 原有索引的集合 - 已选索引的集合
        all_idxs = list(set(all_idxs) - dict_users[i])
    # 返回客户端字典{客户端号 ：{0，5，6，8}数据下标}
    return dict_users

# 非独立同分布（数据集，客户端数）
def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards

    # 60000张图像，200个切片，每个切片300张图像
    num_shards, num_imgs = 200, 300
    # 切片序号列表[0,1,2,...,199]
    idx_shard = [i for i in range(num_shards)]
    # 客户端字典{客户端号 ：一个数组}  ，np.array()创建一个数组
    dict_users = {i: np.array([]) for i in range(num_users)}

    # np.arange()函数：
    # 函数返回一个有终点和起点的固定步长的排列，如[1,2,3,4,5]，起点是1，终点是6，步长为1。
    # 参数个数情况： np.arange()函数分为一个参数，两个参数，三个参数三种情况
    # 1）一个参数时，参数值为终点，起点取默认值0，步长取默认值1。
    # 2）两个参数时，第一个参数为起点，第二个参数为终点，步长取默认值1。
    # 3）三个参数时，第一个参数为起点，第二个参数为终点，第三个参数为步长。其中步长支持小数
    idxs = np.arange(num_shards * num_imgs)  # [0,1,2,...,5999]
    # 获取训练集标签
    labels = dataset.train_labels.numpy()

    # sort labels
    # np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组
    # [[0,1,2,...,5999],[l0,l1,l2,...,l5999]]
    # 拼接数据 sample 索引和标签 variable :  <class 'numpy.ndarray'> :  (2, 60000)
    # dim0: 数据 sample 的索引, dim1: 相应的 label
    idxs_labels = np.vstack((idxs, labels))

    # argsort()：将X中的元素从小到大排序后，提取对应的索引index，然后输出
    # 按照标签排序得到对应索引 variable :  <class 'numpy.ndarray'> :  (2, 60000)
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # 按照标签从 0-9 的数据 sample 索引
    idxs = idxs_labels[0, :]

    # 划分和分配 2 shards/client，每个客户端2个切片
    for i in range(num_users):
        # 随机选择 0-199 中的两组
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        # 删除对应的rand_set
        idx_shard = list(set(idx_shard) - rand_set)
        # 连接
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard + 1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size - 1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]),
                    axis=0)

    return dict_users
