#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: main.py
@Time: 2020/1/2 10:26 AM
"""

import argparse

from reconstruction import Reconstruction
from classification import Classification
from segmentation import Segmentation
from inference import Inference
from svm import SVM


def get_parser():
    parser = argparse.ArgumentParser(description='Unsupervised Point Cloud Feature Learning')
    parser.add_argument('--exp_name', type=str, default=None, metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--task', type=str, default='reconstruct', metavar='N',
                        choices=['reconstruct', 'classify', 'segment'],
                        help='Experiment task, [reconstruct, classify, segment]')
    parser.add_argument('--seg_no_class_label', action='store_true',
                        help='Do not use class labels in segmentation')
    parser.add_argument('--loss', type=str, default='softmax', metavar='N',
                        choices=['softmax', 'triplet'],
                        help='Loss to use, [softmax, triplet]')
    parser.add_argument('--margin', type=float, default=None,
                        help='Margin for triplet loss')
    parser.add_argument('--encoder', type=str, default='foldingnet', metavar='N',
                        choices=['foldnet', 'dgcnn_cls', 'dgcnn_seg'],
                        help='Encoder to use, [foldingnet, dgcnn_cls, dgcnn_seg]')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--feat_dims', type=int, default=512, metavar='N',
                        help='Number of dims for feature ')
    parser.add_argument('--k', type=int, default=None, metavar='N',
                        help='Num of nearest neighbors to use for KNN')
    parser.add_argument('--shape', type=str, default='sphere', metavar='N',
                        choices=['plane', 'sphere', 'gaussian'],
                        help='Shape of points to input decoder, [plane, sphere, gaussian]')
    parser.add_argument('--dataset', type=str, default='shapenetcorev2', metavar='N',
                        choices=['shapenetpart', 'modelnet40', 'modelnet10', 'shapenetpartpart'],
                        help='Dataset to use, [shapenetpart, modelnet40, modelnet10, shapenetpartpart]')
    parser.add_argument('--class_choice', type=str, default=None, metavar='N',
                        choices=['airplane', 'bag', 'cap', 'car', 'chair',
                                 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 
                                 'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table'])
    parser.add_argument('--no_scheduler', action='store_true',
                        help='Do not use scheduler in training')
    parser.add_argument('--use_rotate', action='store_true',
                        help='Rotate the pointcloud before training')
    parser.add_argument('--use_translate', action='store_true',
                        help='Translate the pointcloud before training')
    parser.add_argument('--use_jitter', action='store_true',
                        help='Jitter the pointcloud before training')
    parser.add_argument('--dataset_root', type=str, default='../dataset', help="Dataset root path")
    parser.add_argument('--gpu', type=str, help='Id of gpu device to be used', default='0')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--workers', type=int, help='Number of data loading workers', default=16)
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='Number of episode to train ')
    parser.add_argument('--snapshot_interval', type=int, default=10, metavar='N',
                        help='Save snapshot interval ')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Enables CUDA training')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='Num of points to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Path to load model')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    if args.eval == False:
        if args.task == 'reconstruct':
            reconstruction = Reconstruction(args)
            reconstruction.run()
        elif args.task == 'classify':
            classification = Classification(args)
            classification.run()
        elif args.task == 'segment':
            segmentation = Segmentation(args)
            segmentation.run()
    else:
        inference = Inference(args)
        feature_dir = inference.run()
        svm = SVM(feature_dir)
        svm.run()
