#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: loss.py
@Time: 2020/1/2 10:26 AM
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def batch_pairwise_dist(x, y):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind_x = torch.arange(0, num_points_x)
    diag_ind_y = torch.arange(0, num_points_y)
    if x.get_device() != -1:
        diag_ind_x = diag_ind_x.cuda(x.get_device())
        diag_ind_y = diag_ind_y.cuda(x.get_device())
    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        P = batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)
        return loss_1 + loss_2


class CrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=True):
        super(CrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, preds, gts):
        gts = gts.contiguous().view(-1)

        if self.smoothing:
            eps = 0.2
            n_class = preds.size(1)

            one_hot = torch.zeros_like(preds).scatter(1, gts.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(preds, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(preds, gts, reduction='mean')

        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=None, normalize_feature=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        if self.margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def normalize(self, x, axis=-1):
        """Normalizing to unit length along the specified dimension.
        Args:
            x: pytorch Variable
        Returns:
            x: pytorch Variable, same shape as input      
        """
        x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        return x

    def euclidean_dist(self, x, y):
        """
        Args:
            x: pytorch Variable, with shape [m, d]
            y: pytorch Variable, with shape [n, d]
        Returns:
            dist: pytorch Variable, with shape [m, n]
        """
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

    def hard_example_mining(self, dist_mat, labels, return_inds=False):
        """For each anchor, find the hardest positive and negative sample.
        Args:
            dist_mat: pytorch Variable, pair wise distance between samples, shape [B, N, N]
            labels: pytorch LongTensor, with shape [B, N]
            return_inds: whether to return the indices. Save time if `False`(?)
        Returns:
            dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
            dist_an: pytorch Variable, distance(anchor, negative); shape [N]
            p_inds: pytorch LongTensor, with shape [N]; 
                indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
            n_inds: pytorch LongTensor, with shape [N];
                indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
        NOTE: Only consider the case in which all labels have same num of samples, 
            thus we can cope with all anchors in parallel.
        """

        assert len(dist_mat.size()) == 3
        assert dist_mat.size(1) == dist_mat.size(2)
        B = dist_mat.size(0)
        N = dist_mat.size(1)

        # shape [B, N, N]
        is_pos = labels.unsqueeze(2).expand(B, N, N).eq(labels.unsqueeze(2).expand(B, N, N).transpose(2,1))
        is_neg = labels.unsqueeze(2).expand(B, N, N).ne(labels.unsqueeze(2).expand(B, N, N).transpose(2,1))

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [B, N, 1]
        dist_mat_pos = torch.zeros(dist_mat.size(), dtype=torch.float32, device=dist_mat.device)
        dist_mat_pos[is_pos] = dist_mat[is_pos]
        dist_ap, relative_p_inds = torch.max(dist_mat_pos, 2, keepdim=True)
        # `dist_an` means distance(anchor, negative)
        # both `dist_an` and `relative_n_inds` with shape [B, N, 1]
        dist_mat_neg = torch.zeros(dist_mat.size(), dtype=torch.float32, device=dist_mat.device).fill_(1000)
        dist_mat_neg[is_neg] = dist_mat[is_neg]
        dist_an, relative_n_inds = torch.min(dist_mat_neg, 2, keepdim=True)
        
        # shape [B, N]
        dist_ap = dist_ap.squeeze(2)
        dist_an = dist_an.squeeze(2)

        return dist_ap, dist_an

    def forward(self, preds, gts, new_device):
        """
        Args:
            preds: pytorch Variable, shape [B, N, C]
            gts: pytorch LongTensor, with shape [B, N]
        Returns:
            loss: pytorch Variable, with shape [1]
            p_inds: pytorch LongTensor, with shape [N]; 
                indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
            n_inds: pytorch LongTensor, with shape [N];
                indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
            ==================
            For Debugging, etc
            ==================
            dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
            dist_an: pytorch Variable, distance(anchor, negative); shape [N]
            dist_mat: pytorch Variable, pairwise euclidean distance; shape [N, N]
        """
        if self.normalize_feature:
            preds = self.normalize(preds, axis=-1)
        preds = preds.cuda(new_device)
        gts = gts.cuda(new_device)
        # shape [B, N, N]
        dist_mat = batch_pairwise_dist(preds, preds)
        dist_mat = dist_mat.clamp(min=1e-12).sqrt()  # for numerical stability
        dist_ap, dist_an = self.hard_example_mining(dist_mat, gts)
        y = Variable(dist_an.data.new().resize_as_(dist_an.data).fill_(1))
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an

