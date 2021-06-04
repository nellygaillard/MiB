import torch
import torch.nn as nn
from torch import distributed
import torch.nn.functional as functional

from functools import partial, reduce

import models
from model.build_contextpath import build_contextpath
import os

def make_model(opts, classes=None):

    # --- inplace ABN eventually ---

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = BiSeNet(32, 'resnet50')