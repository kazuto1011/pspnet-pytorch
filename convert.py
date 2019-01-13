#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-15

from __future__ import print_function

import re
from collections import OrderedDict

import click
import numpy as np
import torch
import yaml
from addict import Dict

from libs import caffe_pb2
from libs.models import PSPNet


def parse_caffemodel(model_path):
    caffemodel = caffe_pb2.NetParameter()
    with open(model_path, "rb") as f:
        caffemodel.MergeFromString(f.read())

    # Check trainable layers
    print(set([(layer.type, len(layer.blobs)) for layer in caffemodel.layer]))

    params = OrderedDict()
    for layer in caffemodel.layer:
        print("{} ({}): {}".format(layer.name, layer.type, len(layer.blobs)))

        # Convolution or Dilated Convolution
        if "Convolution" in layer.type:
            params[layer.name] = {}
            params[layer.name]["kernel_size"] = layer.convolution_param.kernel_size[0]
            params[layer.name]["stride"] = layer.convolution_param.stride[0]
            params[layer.name]["weight"] = list(layer.blobs[0].data)
            if len(layer.blobs) == 2:
                params[layer.name]["bias"] = list(layer.blobs[1].data)
            if len(layer.convolution_param.pad) == 1:  # or []
                params[layer.name]["padding"] = layer.convolution_param.pad[0]
            else:
                params[layer.name]["padding"] = 0
            if isinstance(layer.convolution_param.dilation, int):  # or []
                params[layer.name]["dilation"] = layer.convolution_param.dilation
            else:
                params[layer.name]["dilation"] = 1

        # Batch Normalization
        elif "BN" in layer.type:
            params[layer.name] = {}
            params[layer.name]["weight"] = list(layer.blobs[0].data)
            params[layer.name]["bias"] = list(layer.blobs[1].data)
            params[layer.name]["running_mean"] = list(layer.blobs[2].data)
            params[layer.name]["running_var"] = list(layer.blobs[3].data)
            params[layer.name]["eps"] = layer.bn_param.eps
            params[layer.name]["momentum"] = layer.bn_param.momentum

    return params


# Hard coded translater
def translate_layer_name(source):
    def conv_or_bn(source):
        if "bn" in source:
            return ".bn"
        else:
            return ".conv"

    source = re.split("[_/]", source)
    layer = int(source[0][4])  # Remove "conv"
    target = ""

    if layer == 1:
        target += "fcn.layer{}.conv{}".format(layer, source[1])
        target += conv_or_bn(source)
    elif layer in range(2, 6):
        block = int(source[1])
        # Auxirally layer
        if layer == 4 and len(source) == 3 and source[2] == "bn":
            target += "aux.conv4_aux.bn"
        elif layer == 4 and len(source) == 2:
            target += "aux.conv4_aux.conv"
        # Pyramid pooling modules
        elif layer == 5 and block == 3 and "pool" in source[2]:
            pyramid = {1: 3, 2: 2, 3: 1, 6: 0}[int(source[2][4])]
            target += "ppm.stages.s{}.conv".format(pyramid)
            target += conv_or_bn(source)
        # Last convolutions
        elif layer == 5 and block == 4:
            target += "final.conv5_4"
            target += conv_or_bn(source)
        else:
            target += "fcn.layer{}".format(layer)
            target += ".block{}".format(block)
            if source[2] == "3x3":
                target += ".conv3x3"
            else:
                target += ".{}".format(source[3])
            target += conv_or_bn(source)
    elif layer == 6:
        if len(source) == 1:
            target += "final.conv6"
        else:
            target += "aux.conv6_1"

    return target


@click.command()
@click.option("--config", "-c", required=True)
def main(config):
    WHITELIST = ["kernel_size", "stride", "padding", "dilation", "eps", "momentum"]
    CONFIG = Dict(yaml.load(open(config)))

    params = parse_caffemodel(CONFIG.CAFFE_MODEL)

    model = PSPNet(
        n_classes=CONFIG.N_CLASSES, n_blocks=CONFIG.N_BLOCKS, pyramids=CONFIG.PYRAMIDS
    )
    model.eval()
    own_state = model.state_dict()

    report = []
    state_dict = OrderedDict()
    for layer_name, layer_dict in params.items():
        for param_name, values in layer_dict.items():
            if param_name in WHITELIST:
                attribute = translate_layer_name(layer_name)
                attribute = eval("model." + attribute + "." + param_name)
                message = " ".join(
                    [
                        layer_name.ljust(25),
                        "->",
                        param_name,
                        "pytorch: " + str(attribute),
                        "caffe: " + str(values),
                    ]
                )
                print(message, end="")
                if isinstance(attribute, tuple):
                    if attribute[0] != values:
                        report.append(message)
                else:
                    if abs(attribute - values) > 1e-4:
                        report.append(message)
                print(": Checked!")
                continue
            param_name = translate_layer_name(layer_name) + "." + param_name
            if param_name in own_state:
                print(layer_name.ljust(25), "->", param_name, end="")
                values = torch.FloatTensor(values)
                values = values.view_as(own_state[param_name])
                state_dict[param_name] = values
                print(": Copied!")

    print("Inconsistent parameters (*_3x3 dilation and momentum can be ignored):")
    print(*report, sep="\n")

    # Check
    model.load_state_dict(state_dict)
    torch.save(state_dict, CONFIG.PYTORCH_MODEL)


if __name__ == "__main__":
    main()
