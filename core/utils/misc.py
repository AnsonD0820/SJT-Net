# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import errno
import os
import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.init as initer


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = 255
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()

def get_color_pallete(npimg, dataset='voc'):
    out_img = Image.fromarray(npimg.astype('uint8')).convert('P')
    if dataset == 'vaihingen':
        vaihingen_pallete = [
            255, 0, 0,
            255, 255, 255,
            255, 255, 0,
            0, 255, 0,
            0, 255, 255,
            0, 0, 255,
        ]
        out_img.putpalette(vaihingen_pallete)
    else:
        vocpallete = _getvocpallete(256)
        out_img.putpalette(vocpallete)
    return out_img

def _getvocpallete(num_cls):
    n = num_cls
    pallete = [0]*(n*3)
    for j in range(0, n):
        lab = j
        pallete[j*3+0] = 0
        pallete[j*3+1] = 0
        pallete[j*3+2] = 0
        i = 0
        while (lab > 0):
            pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
            pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
            pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3
    return pallete


def tp(output, target, n_classes=6):
    res = []
    for cls in range(n_classes):
        pred_inds = output == cls
        target_inds = target == cls
        res.append(float(pred_inds[target_inds].sum()))
    return np.array(res).astype(float)


def fp(output, target, n_classes=6):
    res = []
    for cls in range(n_classes):
        pred_inds = output == cls
        target_inds = target != cls
        res.append(float(pred_inds[target_inds].sum()))
    return np.array(res).astype(float)


def fn(output, target, n_classes=6):
    res = []
    for cls in range(n_classes):
        pred_inds = output != cls
        target_inds = target == cls
        res.append(float(pred_inds[target_inds].sum()))
    return np.array(res).astype(float)


def tn(output, target, n_classes=6):
    res = []
    for cls in range(n_classes):
        pred_inds = output != cls
        target_inds = target != cls
        res.append(float(pred_inds[target_inds].sum()))
    return np.array(res).astype(float)


def iou(output, target, n_classes=6):
    smooth = 1e-5
    ious = []
    for cls in range(n_classes):
        pred_inds = output == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        ious.append((float(intersection)+smooth)/ (float(union) + smooth))
    return np.array(ious)*100


def f1(output, target, n_classes=6):
    smooth = 1e-5
    f1 = (2*tp(output, target, n_classes) + smooth)/(2*tp(output, target, n_classes)+fp(output, target, n_classes)+fn(output, target, n_classes) + smooth)
    return f1*100


def OA(output, target, n_classes=6):
    smooth = 1e-5
    acc = (tp(output, target, n_classes) + tn(output, target, n_classes) + smooth)/(tp(output, target, n_classes)+fp(output, target, n_classes) + tn(output, target, n_classes) + fn(output, target, n_classes) + smooth)
    return acc*100