import math
import os
import os.path as osp
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as tordata

from .network import TripletLoss, SetNet
from .utils import TripletSampler, evaluation


class Model:
    def __init__(self,
                 hidden_dim,
                 lr,
                 hard_or_full_trip,
                 margin,
                 num_workers,
                 batch_size,
                 restore_iter,
                 total_iter,
                 save_name,
                 train_pid_num,
                 frame_num,
                 model_name,
                 train_source,
                 test_source,
                 probe_source,
                 gallery_source,
                 img_size=64):

        self.save_name = save_name
        self.train_pid_num = train_pid_num
        self.train_source = train_source
        self.test_source = test_source

        self.probe_source = probe_source
        self.gallery_source = gallery_source

        self.hidden_dim = hidden_dim
        self.lr = lr
        self.hard_or_full_trip = hard_or_full_trip
        self.margin = margin
        self.frame_num = frame_num
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.model_name = model_name
        self.P, self.M = batch_size

        self.restore_iter = restore_iter
        self.total_iter = total_iter

        self.img_size = img_size

        self.encoder = SetNet(self.hidden_dim).float()
        self.encoder = nn.DataParallel(self.encoder)
        self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin).float()
        self.triplet_loss = nn.DataParallel(self.triplet_loss)
        self.encoder.cuda()
        self.triplet_loss.cuda()

        self.optimizer = optim.Adam([
            {'params': self.encoder.parameters()},
        ], lr=self.lr)

        self.hard_loss_metric = []
        self.full_loss_metric = []
        self.full_loss_num = []
        self.dist_list = []
        self.mean_dist = 0.01

        self.sample_type = 'all'

    def collate_fn(self, batch):
        batch_size = len(batch)   # batch的大小
        """
        data = [self.__loader__(_path) for _path in self.seq_dir[index]]
        feature_num代表的是data数据所包含的集合的个数,这里一直为1，因为读取的是
          _seq_dir = osp.join(seq_type_path, _view)
                seqs = os.listdir(_seq_dir)  # 遍历出所有的轮廓剪影
        """
        feature_num = len(batch[0][0])
        seqs = [batch[i][0] for i in range(batch_size)]
        frame_sets = [batch[i][1] for i in range(batch_size)]
        view = [batch[i][2] for i in range(batch_size)]
        seq_type = [batch[i][3] for i in range(batch_size)]
        label = [batch[i][4] for i in range(batch_size)]
        batch = [seqs, view, seq_type, label, None]
        '''
        这里的一个样本由data,frame_set,self.view[index],self.seq_type[index],self.label[index]
        组成
        '''

        def select_frame(index):
            sample = seqs[index]
            frame_set = frame_sets[index]
            if self.sample_type == 'random':
                # 这里的random.choices是有放回的抽取样本
                frame_id_list = random.choices(frame_set, k=self.frame_num)
                _ = [feature.loc[frame_id_list].values for feature in sample]
                #feature.loc[] 传入list会取出一组的数据
            else:
                _ = [feature.values for feature in sample]
            return _

        # 提取出每个样本的帧，组成list,存的是array组成的list
        seqs = list(map(select_frame, range(len(seqs))))
        # print(self.sample_type,"采样版本")


        if self.sample_type == 'random':
            seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]

            if random.random() < 0.5:
                seqs[0] = seqs[0][:, :, :, ::-1].copy()

            # for ii in range(seqs[0].shape[0]):
            #     for jj in range(seqs[0].shape[1]):
            #         if random.random() < 0.3:
            #             line = random.randint(5, 59)
            #             seqs[0][ii, jj, line-5:line+5, :] = 0
        else:
            gpu_num = min(torch.cuda.device_count(), batch_size)
            batch_per_gpu = math.ceil(batch_size / gpu_num)
            batch_frames = [[
                                len(frame_sets[i])
                                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                if i < batch_size
                                ] for _ in range(gpu_num)]
            if len(batch_frames[-1]) != batch_per_gpu:
                for _ in range(batch_per_gpu - len(batch_frames[-1])):
                    batch_frames[-1].append(0)
            max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
            # max_sum_frame = 100
            seqs = [[
                        np.concatenate([
                                           seqs[i][j]
                                           for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                                           if i < batch_size
                                           ], 0) for _ in range(gpu_num)]
                    for j in range(feature_num)]
            seqs = [np.asarray([
                                   np.pad(seqs[j][_],
                                          ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                                          'constant',
                                          constant_values=0)
                                   for _ in range(gpu_num)])
                    for j in range(feature_num)]
            batch[4] = np.asarray(batch_frames)

        batch[0] = seqs
        return batch

    def fit(self):
        if self.restore_iter != 0:
            self.load(self.restore_iter)

        self.encoder.train()
        self.sample_type = 'random'
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        triplet_sampler = TripletSampler(self.train_source, self.batch_size)
        # print("triplet_sampler", triplet_sampler)
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_sampler=triplet_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)


        train_label_set = list(self.train_source.label_set)
        train_label_set.sort()

        _time1 = datetime.now()
        pre_valid_acc = -100
        for seq, view, seq_type, label, batch_frame in train_loader:
            self.restore_iter += 1
            self.optimizer.zero_grad()

            for i in range(len(seq)):
                seq[i] = self.np2var(seq[i]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            feature, label_prob = self.encoder(*seq, batch_frame)

            target_label = [train_label_set.index(l) for l in label]
            target_label = self.np2var(np.array(target_label)).long()
            # print("target_label",target_label)

            triplet_feature = feature.permute(1, 0, 2).contiguous()
            triplet_label = target_label.unsqueeze(0).repeat(triplet_feature.size(0), 1)
            (full_loss_metric, hard_loss_metric, mean_dist, full_loss_num
             ) = self.triplet_loss(triplet_feature, triplet_label)
            if self.hard_or_full_trip == 'hard':
                loss = hard_loss_metric.mean()
            elif self.hard_or_full_trip == 'full':
                loss = full_loss_metric.mean()

            self.hard_loss_metric.append(hard_loss_metric.mean().data.cpu().numpy())
            self.full_loss_metric.append(full_loss_metric.mean().data.cpu().numpy())
            self.full_loss_num.append(full_loss_num.mean().data.cpu().numpy())
            self.dist_list.append(mean_dist.mean().data.cpu().numpy())

            if loss > 1e-9:
                loss.backward()
                self.optimizer.step()

            if self.restore_iter % 100 == 0:
                print('iter {}:'.format(self.restore_iter), end='')
                print(', hard_loss_metric={0:.8f}'.format(np.mean(self.hard_loss_metric)), end='')
                print(', full_loss_metric={0:.8f}'.format(np.mean(self.full_loss_metric)), end='')
                print(', full_loss_num={0:.8f}'.format(np.mean(self.full_loss_num)), end='')
                self.mean_dist = np.mean(self.dist_list)
                print(', mean_dist={0:.8f}'.format(self.mean_dist), end='')
                print(', lr=%f' % self.optimizer.param_groups[0]['lr'], end='')
                print(', hard or full=%r' % self.hard_or_full_trip)

                # with open('train_log.txt', 'a') as f:
                #     f.write('iter {}: hard_loss_metric={:.4f} full_loss_metric={:.4f} full_loss_num={:.4f} mean_dist={:.4f} \n'.format(
                #         self.restore_iter, np.mean(self.hard_loss_metric), np.mean(self.full_loss_metric), np.mean(self.full_loss_num), self.mean_dist
                #     ))

                sys.stdout.flush()
                self.hard_loss_metric = []
                self.full_loss_metric = []
                self.full_loss_num = []
                self.dist_list = []

            # Visualization using t-SNE
            # if self.restore_iter % 500 == 0:
            #     pca = TSNE(2)
            #     pca_feature = pca.fit_transform(feature.view(feature.size(0), -1).data.cpu().numpy())
            #     for i in range(self.P):
            #         plt.scatter(pca_feature[self.M * i:self.M * (i + 1), 0],
            #                     pca_feature[self.M * i:self.M * (i + 1), 1], label=label[self.M * i])
            #
            #     plt.show()
            
            if self.restore_iter % 1000 == 0:
                valid_probe = self.transform('probe')
                valid_gallery = self.transform('gallery')
                valid_acc = evaluation(valid_probe, valid_gallery, valid=True)
                print('iter {}:'.format(self.restore_iter), end='')
                print('valid_acc={:.4f} \n'.format(valid_acc))

                if valid_acc > pre_valid_acc:
                    pre_valid_acc = valid_acc
                    self.save()
                    print('save success!')

                # with open('train_log.txt', 'a') as f:
                #     f.write('iter {}: valid_acc={:.4f} \n'.format(self.restore_iter, valid_acc))
                    
                self.encoder.train()
                self.sample_type = 'random'

            if self.restore_iter == self.total_iter:
                break

    def ts2var(self, x):
        return autograd.Variable(x).cuda()

    def np2var(self, x):
        return self.ts2var(torch.from_numpy(x))

    def transform(self, flag, batch_size=1):
        self.encoder.eval()
        source = self.test_source if flag == 'test' else self.train_source

        if flag == 'probe':
            source = self.probe_source
        if flag == 'gallery':
            source = self.gallery_source

        self.sample_type = 'all'
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            collate_fn=self.collate_fn,
            num_workers=self.num_workers)

        feature_list = list()
        view_list = list()
        seq_type_list = list()
        label_list = list()

        for i, x in enumerate(data_loader):
            # print(i, len(data_loader))
            seq, view, seq_type, label, batch_frame = x

            for j in range(len(seq)):
                seq[j] = self.np2var(seq[j]).float()
            if batch_frame is not None:
                batch_frame = self.np2var(batch_frame).int()

            temp = seq[0]
            frame_size = temp.size(1)
            idx = list(range(frame_size))
            feature = list()

            for j in range(30):
                random.shuffle(idx)
                choice_idx = sorted(idx[:self.frame_num])

                current_temp = temp[:, choice_idx, :, :]
                if len(choice_idx) < self.frame_num:
                    current_temp = torch.cat([current_temp, torch.zeros((temp.size(0), self.frame_num-len(choice_idx), temp.size(2), temp.size(3))).cuda()], 1)

                f, _ = self.encoder(current_temp, batch_frame)
                feature.append(f)

            feature = torch.stack(feature)
            feature = torch.mean(feature, 0)
            
            n, num_bin, _ = feature.size()
            feature_list.append(feature.view(n, -1).data.cpu().numpy())
            view_list += view
            seq_type_list += seq_type
            label_list += label

        return np.concatenate(feature_list, 0), view_list, seq_type_list, label_list

    def save(self):
        os.makedirs(osp.join('checkpoint', self.model_name), exist_ok=True)
        torch.save(self.encoder.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-encoder.ptm'.format(
                                self.save_name, 100)))
        torch.save(self.optimizer.state_dict(),
                   osp.join('checkpoint', self.model_name,
                            '{}-{:0>5}-optimizer.ptm'.format(
                                self.save_name, 100)))

    # restore_iter: iteration index of the checkpoint to load
    def load(self, restore_iter=-1):
        model_path = osp.join(
            'checkpoint', self.model_name,
            '{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter))
        if osp.exists(model_path):
            self.encoder.load_state_dict(torch.load(model_path))
            self.optimizer.load_state_dict(torch.load(osp.join(
                'checkpoint', self.model_name,
                '{}-{:0>5}-optimizer.ptm'.format(self.save_name, restore_iter))))
        else:
            self.encoder.load_state_dict(torch.load('weights/best-model.ptm'))
