#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2018-01-24

import json
import pickle
from math import ceil

import click
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm

from libs.datasets import VOCSegmentation
from libs.models import PSPNet
from libs.utils import scores


def pad_image(image, crop_size):
    new_h, new_w = image.shape[2:]
    pad_h = max(crop_size - new_h, 0)
    pad_w = max(crop_size - new_w, 0)
    padded_image = torch.FloatTensor(1, 3, new_h + pad_h, new_w + pad_w).zero_()
    for i in range(3):  # RGB
        padded_image[:, [i], ...] = F.pad(
            image[:, [i], ...],
            pad=(0, pad_w, 0, pad_h),  # Pad right and bottom
            mode="constant",
            value=0,
        ).data
    return padded_image


def to_cuda(tensors, cuda):
    return tensors.cuda() if cuda else tensors


def to_var(tensors, cuda):
    tensors = to_cuda(tensors, cuda)
    variables = Variable(tensors, volatile=True)
    return variables


def flip(x, dim=3):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[
        :,
        getattr(
            torch.arange(x.size(1) - 1, -1, -1), ("cpu", "cuda")[x.is_cuda]
        )().long(),
        :,
    ]
    return x.view(xsize)


def tile_predict(image, model, crop_size, cuda, n_classes):
    # Original MATLAB script
    # https://github.com/hszhao/PSPNet/blob/master/evaluation/scale_process.m
    pad_h, pad_w = image.shape[2:]
    stride_rate = 2 / 3.0
    stride = int(ceil(crop_size * stride_rate))
    h_grid = int(ceil((pad_h - crop_size) / float(stride)) + 1)
    w_grid = int(ceil((pad_w - crop_size) / float(stride)) + 1)
    count = to_cuda(torch.FloatTensor(1, 1, pad_h, pad_w).zero_(), cuda)
    prediction = to_cuda(torch.FloatTensor(1, n_classes, pad_h, pad_w).zero_(), cuda)
    for ih in range(h_grid):
        for iw in range(w_grid):
            sh, sw = ih * stride, iw * stride
            eh, ew = min(sh + crop_size, pad_h), min(sw + crop_size, pad_w)
            sh, sw = eh - crop_size, ew - crop_size  # Stay within image size
            image_sub = image[..., sh:eh, sw:ew]
            image_sub = pad_image(image_sub, crop_size)
            image_sub = to_var(image_sub, cuda)
            output = model(image_sub)
            output = F.upsample(output, size=(crop_size,) * 2, mode="bilinear")
            count[..., sh:eh, sw:ew] += 1
            prediction[..., sh:eh, sw:ew] += output.data
    prediction /= count  # Normalize overlayed parts
    return prediction


@click.command()
@click.option("--config", "-c", required=True)
@click.option("--cuda/--no-cuda", default=True)
@click.option("--show", is_flag=True)
def main(config, cuda, show):
    CONFIG = Dict(yaml.load(open(config)))

    cuda = cuda and torch.cuda.is_available()

    dataset = VOCSegmentation(
        root=CONFIG.DATASET_ROOT, image_set="val", dataset_name="VOC2012"
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,  #! DO NOT CHANGE
        num_workers=CONFIG.NUM_WORKERS,
        pin_memory=False,
        shuffle=False,
    )

    # Load a model
    state_dict = torch.load(CONFIG.PYTORCH_MODEL)

    # Model
    model = PSPNet(
        n_classes=CONFIG.N_CLASSES, n_blocks=CONFIG.N_BLOCKS, pyramids=CONFIG.PYRAMIDS
    )
    model.load_state_dict(state_dict)
    model = nn.DataParallel(model)
    model.eval()
    if cuda:
        model.cuda()

    crop_size = CONFIG.IMAGE.SIZE.TEST
    targets, outputs = [], []

    for image, target in tqdm(
        dataloader, total=len(dataloader), leave=False, dynamic_ncols=True
    ):

        h, w = image.size()[2:]
        outputs_ = []

        for scale in CONFIG.SCALES:

            # Resize
            long_side = int(scale * CONFIG.IMAGE.SIZE.BASE)
            new_h = long_side
            new_w = long_side
            if h > w:
                new_w = int(long_side * w / h)
            else:
                new_h = int(long_side * h / w)
            image_ = F.upsample(image, size=(new_h, new_w), mode="bilinear").data

            # Predict (w/ flipping)
            if long_side <= crop_size:
                # Padding evaluation
                image_ = pad_image(image_, crop_size)
                image_ = to_var(image_, cuda)
                output = torch.cat(
                    (model(image_), flip(model(flip(image_))))  # C, H, W  # C, H, W
                )
                output = F.upsample(output, size=(crop_size,) * 2, mode="bilinear")
                # Revert to original size
                output = output[..., 0:new_h, 0:new_w]
                output = F.upsample(output, size=(h, w), mode="bilinear")
                outputs_ += [o for o in output.data]  # 2 x [C, H, W]
            else:
                # Sliced evaluation
                image_ = pad_image(image_, crop_size)
                output = torch.cat(
                    (
                        tile_predict(image_, model, crop_size, cuda, CONFIG.N_CLASSES),
                        flip(
                            tile_predict(
                                flip(image_), model, crop_size, cuda, CONFIG.N_CLASSES
                            )
                        ),
                    )
                )
                # Revert to original size
                output = output[..., 0:new_h, 0:new_w]
                output = F.upsample(output, size=(h, w), mode="bilinear")
                outputs_ += [o for o in output.data]  # 2 x [C, H, W]

        # Average
        output = torch.stack(outputs_, dim=0)  # 2x#scales, C, H, W
        output = torch.mean(output, dim=0)  # C, H, W
        output = torch.max(output, dim=0)[1]  # H, W
        output = output.cpu().numpy()
        target = target.squeeze(0).numpy()

        if show:
            res_gt = np.concatenate((output, target), 1)
            mask = (res_gt >= 0)[..., None]
            res_gt[res_gt < 0] = 0
            res_gt = np.uint8(res_gt / float(CONFIG.N_CLASSES) * 255)
            res_gt = cv2.applyColorMap(res_gt, cv2.COLORMAP_JET)
            res_gt = np.uint8(res_gt * mask)
            img = np.uint8(image.numpy()[0].transpose(1, 2, 0) + dataset.mean_rgb)[
                ..., ::-1
            ]
            img_res_gt = np.concatenate((img, res_gt), 1)
            cv2.imshow("result", img_res_gt)
            cv2.waitKey(10)

        outputs.append(output)
        targets.append(target)

    score, class_iou = scores(targets, outputs, n_class=CONFIG.N_CLASSES)

    for k, v in score.items():
        print(k, v)

    score["Class IoU"] = {}
    for i in range(CONFIG.N_CLASSES):
        score["Class IoU"][i] = class_iou[i]

    with open("results.json", "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
