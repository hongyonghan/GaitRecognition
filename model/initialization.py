# -*- coding: utf-8 -*-
# @Author  : admin
# @Time    : 2018/11/15
import os
from copy import deepcopy

import numpy as np

from .utils import load_data
from .model import Model


def initialize_data(config, train=False, test=False):
    #这里的train和test代表是否使用cache来缓存数据
    print("Initializing data source...")
    #得到Dateset对象
    train_source, test_source = load_data(**config['data'], cache=(train or test))

    probe_source, _ = load_data(**config['probe'], cache=False)
    gallery_source, _ = load_data(**config['gallery'], cache=False)

    if train:
        print("Loading training data...")
        train_source.load_all_data()
    if test:
        print("Loading test data...")
        test_source.load_all_data()
    print("Data initialization complete.")
    return train_source, test_source, probe_source, gallery_source


def initialize_model(config, train_source, test_source, probe_source, gallery_source):
    print("Initializing model...")
    data_config = config['data']
    # print("data_config",data_config)   #测试使用，检查data_config
    model_config = config['model']
    # print("model_config",model_config)
    model_param = deepcopy(model_config)
    model_param['train_source'] = train_source
    model_param['test_source'] = test_source

    model_param['probe_source'] = probe_source
    model_param['gallery_source'] = gallery_source

    model_param['train_pid_num'] = data_config['pid_num']
    batch_size = int(np.prod(model_config['batch_size']))  # np.prod 计算所有元素的乘积。
    model_param['save_name'] = '_'.join(map(str,[
        model_config['model_name'],
        data_config['dataset'],
        data_config['pid_num'],
        data_config['pid_shuffle'],
        model_config['hidden_dim'],
        model_config['margin'],
        batch_size,
        model_config['hard_or_full_trip'],
        model_config['frame_num'],
    ]))

    m = Model(**model_param)
    print("Model initialization complete.")
    return m, model_param['save_name']


def initialization(config, train=False, test=False):
    print("Initialzing...")
    WORK_PATH = config['WORK_PATH']
    os.chdir(WORK_PATH)   #改变当前工作目录到指定的路径
    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]
    train_source, test_source, probe_source, gallery_source = initialize_data(config, train, test)
    return initialize_model(config, train_source, test_source, probe_source, gallery_source)