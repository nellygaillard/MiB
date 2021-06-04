import torch
import torch.nn as nn
from torch import distributed
import torch.nn.functional as functional

#import inplace_abn
#from inplace_abn import InPlaceABNSync, InPlaceABN, ABN

from functools import partial, reduce

import models
from model.build_BiSeNet import BiSeNet
import os


def make_model(opts, classes=None):
    
    #if opts.norm_act == 'iabn_sync':
    #    norm = partial(InPlaceABNSync, activation="leaky_relu", activation_param=.01)
    #elif opts.norm_act == 'iabn':
    #    norm = partial(InPlaceABN, activation="leaky_relu", activation_param=.01)
    #elif opts.norm_act == 'abn':
    #    norm = partial(ABN, activation="leaky_relu", activation_param=.01)
    #else:
    #    norm = nn.BatchNorm2d  # not synchronized, can be enabled with apex
    
    
    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    num_classes = reduce(lambda a,b: a+b, self.classes)
    
    
    if not opts.no_pretrained:      # if we have a pretrained model
        pretrained_path = f'pretrained/{opts.backbone}_{opts.norm_act}.pth.tar'
        pre_dict = torch.load(pretrained_path, map_location='cpu')
        del pre_dict['state_dict']['classifier.fc.weight']
        del pre_dict['state_dict']['classifier.fc.bias']

        body.load_state_dict(pre_dict['state_dict'])
        del pre_dict  # free memory

    head_channels = 256

    if classes is not None:
        model = BiSeNet(num_classes, opts.backbone)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
        model2 = IncrementalSegmentationModule(model, 256, classes=classes, fusion_mode=opts.fusion_mode)
    else:
        model = BiSeNet(num_classes, opts.backbone)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
        model2 = SegmentationModule(model, 256, opts.num_classes, opts.fusion_mode)

    return model


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class IncrementalSegmentationModule(nn.Module):

    def __init__(self, model, head_channels, classes, ncm=False, fusion_mode="mean"):
        super(IncrementalSegmentationModule, self).__init__()
        self.model = model
        
        # classes must be a list where [n_class_task[i] for i in tasks]
        assert isinstance(classes, list), \
            "Classes must be a list where to every index correspond the num of classes for that task"

        # list of classifiers
        self.cls = nn.ModuleList(
            [nn.Conv2d(head_channels, c, 1) for c in classes]
        )
        self.classes = classes
        self.head_channels = head_channels
        self.tot_classes = reduce(lambda a, b: a + b, self.classes)
        self.means = None

    def _network(self, x, ret_intermediate=False):

        x_net = self.model(x)
        out = []
        for mod in self.cls:
            out.append(mod(x_net))
        x_o = torch.cat(out, dim=1)

        if ret_intermediate:
            return x_o, x_net
        return x_o

    def init_new_classifier(self, device):
        cls = self.cls[-1]
        imprinting_w = self.cls[0].weight[0]
        bkg_bias = self.cls[0].bias[0]

        bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).to(device)

        new_bias = (bkg_bias - bias_diff)

        cls.weight.data.copy_(imprinting_w)
        cls.bias.data.copy_(new_bias)

        self.cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    def forward(self, x, scales=None, do_flip=False, ret_intermediate=False):
        out_size = x.shape[-2:]

        out = self._network(x, ret_intermediate)

        sem_logits = out[0] if ret_intermediate else out

        sem_logits = functional.interpolate(sem_logits, size=out_size, mode="bilinear", align_corners=False)

        if ret_intermediate:
            return sem_logits, {"body": out[1], "pre_logits": out[2]}

        return sem_logits, {}

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, inplace_abn.ABN):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
