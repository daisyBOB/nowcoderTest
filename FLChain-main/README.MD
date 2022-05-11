# Fedetaed Learning on BlockChain

## Options选项

The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:解析到实验的各种参数的默认值以"options.py"给出。给出了其中一些参数的详细信息：

* ```--model:```    Default: 'mlp'. Options: 'mlp', 'cnn'

  模型：默认："mlp"。选项："mlp"、"cnn"

* ```--global_round:```   Number of rounds of training. 

  训练回合数

* ```--idx``` The ID for training client. 

  训练客户端的id

* ```--iid:```      Distribution of data amongst users. Default set to IID. Set to 0 for non-IID.

  在用户之间分发数据。默认设置为 IID。对于非 IID，设置为 0。

* ```--num_users:```Number of users. Default is 100.

  用户数。默认值为 100。

* ```--local_ep:``` Number of local training epochs in each user. Default is 10.

  每个用户的本地训练轮数。默认值为 10。epochs：训练过程中数据将被“轮”多少次。

* ```--local_bs:``` Batch size of local updates in each user. Default is 10.

  每个用户中本地更新的批大小。默认值为 10。

* ```--unequal:```  Used in non-iid setting. Option to split the data amongst users equally or unequally. Default set to 0 for equal splits. Set to 1 for unequal splits.
  
  用于非 iid 设置。在用户之间平均或不平等地拆分数据的选项。对于相等的拆分，默认设置为 0。对于不相等的拆分，设置为 1。
## Usage用法//
* To init global model on CNN :

  在CNN上引入全局模型
```
python src/init_global.py --model=cnn 

```

* To run the federated experiment on CNN for i-th client

  在 CNN 上为第 i 个客户端运行联邦实验
```
python src/init_global.py --model=cnn --global_round=10 --iid=1 --num_users=5 --idx=1
```