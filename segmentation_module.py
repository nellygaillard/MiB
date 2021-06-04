import torch
import torch.nn as nn
from torch import distributed
import torch.nn.functional as functional

from functools import partial, reduce

from model.build_BiSeNet import IncrementalBiSeNet
import os

def make_model(opts, classes=None):

    # --- inplace ABN eventually ---

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    assert classes is not None, "Error: classes is None"
    assert isinstance(classes, list), \
        "Classes must be a list where to every index correspond the num of classes for that task"

    model = IncrementalBiSeNet(classes, opts.backbone)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    return model