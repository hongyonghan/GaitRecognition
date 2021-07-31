conf = {
    "WORK_PATH": "./work",
    "CUDA_VISIBLE_DEVICES": "0",
    "data": {
        'dataset_path': "../dataset/train_pre",
        'resolution': '64',
        'dataset': 'CASIA-B',
        'pid_num': 74, #102  # LT划分方式，74用于训练，其余用于测试
        'pid_shuffle': False,
        # 是否随机进行划分数据集，如果为false,那么直接选取1-74为训练集，剩余的为测试集
    },
    "probe": {
        'dataset_path': "../dataset/test_probe_pre",## 'dataset_path': "../dataset/test_probe_pre",
        'resolution': '64',
        'dataset': 'CASIA-B',
        'pid_num': 1,
        'pid_shuffle': False,
    },
    "gallery": {
        'dataset_path': "../dataset/test_gallery_pre",
        'resolution': '64',
        'dataset': 'CASIA-B',
        'pid_num': 11,
        'pid_shuffle': False,
    },
    "model": {
        'hidden_dim': 256,
        'lr': 1e-4,
        'hard_or_full_trip': 'full',   #设置三元损失的模式为full,hard模式下容易梯度爆炸
        #这个修改batchsize的大小。原来是(8.16)  修改为(8.4)
        'batch_size': (8, 16),
        #这里是接着训练的轮数 原来是0，修改为7690
        'restore_iter': 0,
        #修改为8000，原来为200000
        'total_iter': 80000,
        'margin': 0.2,  #三元损失时控制正负样本之间的距离。具体看论文：https://blog.csdn.net/xuluohongshang/article/details/78965580
        #可以尝试num_workers修改为8
        'num_workers': 0,
        #原来是30,修改为20
        # 随机取30帧来进行训练或者测试
        'frame_num': 45,
        'model_name': 'GaitSet',
    },
}
