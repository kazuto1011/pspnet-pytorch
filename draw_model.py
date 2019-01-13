#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-06

import torch
from graphviz import Digraph
from torch.autograd import Variable

from libs.models import *


def make_dot(var, params):

    param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(
        style="filled",
        shape="box",
        align="left",
        fontsize="12",
        ranksep="0.1",
        height="0.2",
    )
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return "(" + (", ").join(["%d" % v for v in size]) + ")"

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor="orange")
            elif hasattr(var, "variable"):
                u = var.variable
                dot.node(str(id(var)), size_to_str(u.size()), fillcolor="lightblue")
            else:
                dot.node(str(id(var)), str(type(var).__name__.replace("Backward", "")))
            seen.add(var)
            if hasattr(var, "next_functions"):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, "saved_tensors"):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)

    return dot


if __name__ == "__main__":
    # Define a model
    model = PSPNet(n_classes=6, n_blocks=[3, 4, 6, 3], pyramids=[6, 3, 2, 1])

    # Build a computational graph from x to y
    x = torch.randn(2, 3, 512, 512)
    y1, y2 = model(Variable(x))
    g = make_dot(y1 + y2, model.state_dict())
    g.view(filename="model", cleanup=True)
