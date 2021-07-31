from datetime import datetime
import numpy as np
import argparse

from model.initialization import initialization
from model.utils import evaluation
from config import conf


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'

def modeltest():
    #上面函数为测试flask程序使用。
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--iter', default='-1', type=int,
                            help='iter: iteration of the checkpoint to load. Default: 80000')
    parser.add_argument('--batch_size', default='1', type=int,
                            help='batch_size: batch size for parallel test. Default: 1')
    parser.add_argument('--cache', default=False, type=boolean_string,
                            help='cache: if set as TRUE all the test data will be loaded at once'
                                 ' before the transforming start. Default: FALSE')
    opt = parser.parse_args()
    m = initialization(conf, test=opt.cache)[0]

        # load model checkpoint of iteration opt.iter
    print('Loading the model...')
    m.load()
    print('Transforming...')
    time = datetime.now()

    probe = m.transform('probe', opt.batch_size)
    gallery = m.transform('gallery', opt.batch_size)

    return evaluation(probe, gallery)
if __name__ == '__main__':
    modeltest()

