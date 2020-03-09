#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: reconstruction.py
@Time: 2020/1/2 10:26 AM
"""

import os
import sys
import time
import shutil
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics

from tensorboardX import SummaryWriter

from model import SegmentationNet
from dataset import Dataset
from utils import Logger

torch.multiprocessing.set_sharing_strategy('file_system')

seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]


def calculate_shape_IoU(pred_np, seg_np, label, class_choice):
    label = label.squeeze()
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious


class Segmentation(object):
    def __init__(self, args):
        self.dataset_name = args.dataset
        if args.epochs != None:
            self.epochs = args.epochs
        else:
            self.epochs = 250
        self.batch_size = args.batch_size
        self.snapshot_interval = args.snapshot_interval
        self.no_cuda = args.no_cuda
        self.model_path = args.model_path
        self.class_choice = args.class_choice
        self.no_scheduler = args.no_scheduler
        self.loss = args.loss

        # create exp directory
        file = [f for f in args.model_path.split('/')]
        if args.exp_name != None:
            self.experiment_id = "Segment_" + args.exp_name
        elif file[-2] == 'models':
            self.experiment_id = file[-3]
        else:
            self.experiment_id = "Segment" + time.strftime('%m%d%H%M%S')
        snapshot_root = 'snapshot/%s' % self.experiment_id
        tensorboard_root = 'tensorboard/%s' % self.experiment_id
        self.save_dir = os.path.join(snapshot_root, 'models/')
        self.tboard_dir = tensorboard_root

        # check arguments
        if self.model_path == '':
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            else:
                choose = input("Remove " + self.save_dir + " ? (y/n)")
                if choose == "y":
                    shutil.rmtree(self.save_dir)
                    os.makedirs(self.save_dir)
                else:
                    sys.exit(0)
            if not os.path.exists(self.tboard_dir):
                os.makedirs(self.tboard_dir)
            else:
                shutil.rmtree(self.tboard_dir)
                os.makedirs(self.tboard_dir)
        sys.stdout = Logger(os.path.join(snapshot_root, 'log.txt'))
        self.writer = SummaryWriter(log_dir=self.tboard_dir)

        # print args
        print(str(args))

        # get gpu id
        gids = ''.join(args.gpu.split())
        self.gpu_ids = [int(gid) for gid in gids.split(',')]
        self.first_gpu = self.gpu_ids[1]
        self.loss_gpu = self.gpu_ids[0]
        self.gpu_ids = self.gpu_ids[1:]
        # self.device = torch.device("cpu" if args.no_cuda else "cuda")

        # generate dataset
        self.train_dataset = Dataset(
                root=args.dataset_root,
                dataset_name=args.dataset,
                split='all',
                num_points=args.num_points,
                segmentation=True,
                random_translate=True,
                random_rotate=args.use_rotate,
                random_jitter=args.use_jitter
            )
        self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers
            )
        print("Training set size:", self.train_loader.dataset.__len__())

        # initialize model
        self.seg_num_all = self.train_dataset.seg_num_all
        self.seg_start_index = self.train_dataset.seg_start_index
        self.model = SegmentationNet(args, self.seg_num_all)
        if self.model_path != '':
            self._load_pretrain(args.model_path)

        # load model to gpu
        if not self.no_cuda:
            if len(self.gpu_ids) != 1:  # multiple gpus
                self.model = torch.nn.DataParallel(self.model.cuda(self.first_gpu), self.gpu_ids)
            else:
                self.model = self.model.cuda(self.gpu_ids[0])
        # self.model = self.model.to(self.device)
        
        
        # initialize optimizer
        self.parameter = self.model.parameters()
        if self.no_scheduler == False:
            self.optimizer = optim.SGD(self.parameter, lr=0.1, weight_decay=1e-4)
            self.scheduler = CosineAnnealingLR(self.optimizer, self.epochs, eta_min=1e-3)
        else:
            self.optimizer = optim.SGD(self.parameter, lr=0.01, weight_decay=1e-4)


    def run(self):
        self.train_hist = {
            'loss': [],
            'per_epoch_time': [],
            'total_time': []
        }
        best_loss = 1000000000
        print('Training start!!')
        start_time = time.time()
        self.model.train()
        if self.model_path != '':
            start_epoch = self.model_path[-7:-4]
            if start_epoch[0] == '_':
                start_epoch = start_epoch[1:]
            start_epoch = int(start_epoch)
        else:
            start_epoch = 0
        for epoch in range(start_epoch, self.epochs):
            loss = self.train_epoch(epoch)
            
            # save snapeshot
            if (epoch + 1) % self.snapshot_interval == 0:
                self._snapshot(epoch + 1)
                if loss < best_loss:
                    best_loss = loss
                    self._snapshot('best')
            
            # save tensorboard
            if self.writer:
                self.writer.add_scalar('Train Loss', self.train_hist['loss'][-1], epoch)
                self.writer.add_scalar('Learning Rate', self._get_lr(), epoch)
        
        # finish all epoch
        self._snapshot(epoch + 1)
        if loss < best_loss:
            best_loss = loss
            self._snapshot('best')
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epochs, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")


    def train_epoch(self, epoch):
        epoch_start_time = time.time()
        loss_buf = []
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        dist_ap_buf = []
        dist_an_buf = []
        num_batch = int(len(self.train_loader.dataset) / self.batch_size)
        for iter, (pts, label, seg) in enumerate(self.train_loader):
            num_seg = seg.max(1)[0] - seg.min(1)[0]
            if 0 in num_seg:
                pts = pts[num_seg != 0]
                label = label[num_seg != 0]
                seg = seg[num_seg != 0]

            if pts.size(0) == 1:
                continue
            seg = seg - self.seg_start_index
            label_one_hot = torch.zeros((label.size(0), 16))
            for idx in range(label.size(0)):
                label_one_hot[idx, label[idx]] = 1

            if not self.no_cuda:
                pts = pts.cuda(self.first_gpu)
                label_one_hot = label_one_hot.cuda(self.first_gpu)
                seg = seg.cuda(self.first_gpu)
            
            # forward
            self.optimizer.zero_grad()
            output, _ = self.model(pts, label_one_hot)
            output = output.permute(0, 2, 1).contiguous()

            # loss
            if self.loss == 'softmax':
                if len(self.gpu_ids) != 1:  # multiple gpus
                    loss = self.model.module.get_loss(output.view(-1, self.seg_num_all), seg.view(-1))
                else:
                    loss = self.model.get_loss(output.view(-1, self.seg_num_all), seg.view(-1))

                # backward
                loss.backward()
                self.optimizer.step()
                loss_buf.append(loss.detach().cpu().numpy())

                pred = output.max(dim=2)[1]
                seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
                pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
                train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
                train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
                train_true_seg.append(seg_np)
                train_pred_seg.append(pred_np)
                train_label_seg.append(label.reshape(-1))

            elif self.loss == 'triplet':
                if len(self.gpu_ids) != 1:  # multiple gpus
                    loss, dist_ap, dist_an = self.model.module.get_loss(output, seg, new_device=self.loss_gpu)
                else:
                    loss, dist_ap, dist_an = self.model.get_loss(output, seg, new_device=self.loss_gpu)

                # backward
                loss.backward()
                self.optimizer.step()
                loss_buf.append(loss.detach().cpu().numpy())
                dist_ap_buf.append(np.mean(dist_ap.detach().cpu().numpy()))
                dist_an_buf.append(np.mean(dist_an.detach().cpu().numpy()))

            # print(iter, loss, time.time() - epoch_start_time, np.mean(dist_an.detach().cpu().numpy()), np.mean(dist_ap.detach().cpu().numpy()))

        # finish one epoch
        if self.no_scheduler == False:
            self.scheduler.step()
        epoch_time = time.time() - epoch_start_time
        self.train_hist['per_epoch_time'].append(epoch_time)
        self.train_hist['loss'].append(np.mean(loss_buf))
        if self.loss == 'softmax':
            train_true_cls = np.concatenate(train_true_cls)
            train_pred_cls = np.concatenate(train_pred_cls)
            train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
            avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
            train_true_seg = np.concatenate(train_true_seg, axis=0)
            train_pred_seg = np.concatenate(train_pred_seg, axis=0)
            train_label_seg = np.concatenate(train_label_seg)
            train_ious = calculate_shape_IoU(train_pred_seg, train_true_seg, train_label_seg, self.class_choice)
            print("Epoch %d: Loss %.6f, train acc %.6f, train avg acc %.6f, train iou: %.6f, time %.4fs" % (epoch+1,
                                                                                                              np.mean(loss_buf),
                                                                                                              train_acc,
                                                                                                              avg_per_class_acc,
                                                                                                              np.mean(train_ious),
                                                                                                              epoch_time))
        elif self.loss == 'triplet':
            print("Epoch %d: Loss %.6f, dist an %.6f, dist ap %.6f, time %.4fs" % (epoch+1,
                                                                                   np.mean(loss_buf),
                                                                                   np.mean(dist_an_buf),
                                                                                   np.mean(dist_ap_buf),
                                                                                   epoch_time))
        return np.mean(loss_buf)


    def _snapshot(self, epoch):
        state_dict = self.model.state_dict()
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key[:6] == 'module':
                name = key[7:]  # remove 'module.'
            else:
                name = key
            new_state_dict[name] = val
        save_dir = os.path.join(self.save_dir, self.dataset_name)
        torch.save(new_state_dict, save_dir + "_" + str(epoch) + '.pkl')
        print(f"Save model to {save_dir}_{str(epoch)}.pkl")


    def _load_pretrain(self, pretrain):
        state_dict = torch.load(pretrain, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key[:6] == 'module':
                name = key[7:]  # remove 'module.'
            else:
                name = key
            new_state_dict[name] = val
        self.model.load_state_dict(new_state_dict)
        print(f"Load model from {pretrain}")


    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']
