import torch
from torch import nn
from model.build_contextpath import build_contextpath
import warnings
from torch.cuda.amp import autocast

warnings.filterwarnings(action='ignore')

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

class Spatial_path(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(in_channels=3, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock3 = ConvBlock(in_channels=128, out_channels=256)

    @autocast()
    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x

class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    @autocast()
    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        x = self.sigmoid(self.bn(x))
        #x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x


class FeatureFusionModule(torch.nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels
        # resnet101 3328 = 256(from spatial path) + 1024(from context path) + 2048(from context path)
        # resnet18  1024 = 256(from spatial path) + 256(from context path) + 512(from context path)
        # resnet50 3328 = 256(from spatial path) + 1024(from context path) + 2048(from context path)
        self.in_channels = in_channels

        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    @autocast()
    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)

        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x

class IncrementalBiSeNet(torch.nn.Module):
    def __init__(self, classes, context_path):
        super().__init__()
        self.classes=classes
        # build spatial path
        self.saptial_path = Spatial_path()

        # build context path
        self.context_path = build_contextpath(name=context_path)

        # build attention refinement module  for resnet 101
        if context_path == 'resnet101':
            self.attention_refinement_module1 = AttentionRefinementModule(1024, 1024)
            self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048)
            # supervision block
            self.supervision1 = nn.ModuleList(
                [nn.Conv2d(in_channels=1024, out_channels=c, kernel_size=1) for c in classes])
            self.supervision2 = nn.ModuleList(
                [nn.Conv2d(in_channels=2048, out_channels=c, kernel_size=1) for c in classes])
            # build feature fusion module
            #self.FFM = nn.ModuleList(
            #    [FeatureFusionModule(c, 3328) for c in classes]
            #)
            self.feature_fusion_module = FeatureFusionModule(32, 3328)

        # build attention refinement module  for resnet 50
        elif context_path == 'resnet50':
            self.attention_refinement_module1 = AttentionRefinementModule(1024, 1024)
            self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048)
            # supervision block
            self.supervision1 = nn.ModuleList(
                [nn.Conv2d(in_channels=1024, out_channels=c, kernel_size=1) for c in classes])
            self.supervision2 = nn.ModuleList(
                [nn.Conv2d(in_channels=2048, out_channels=c, kernel_size=1) for c in classes])
            # build feature fusion module
            #self.FFM = nn.ModuleList(
            #    [FeatureFusionModule(c, 3328) for c in classes]
            #)
            self.feature_fusion_module = FeatureFusionModule(32, 3328)

        elif context_path == 'resnet18':
            # build attention refinement module  for resnet 18
            self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
            self.attention_refinement_module2 = AttentionRefinementModule(512, 512)
            # supervision block
            self.supervision1 = nn.ModuleList(
                [nn.Conv2d(in_channels=256, out_channels=c, kernel_size=1) for c in classes])
            self.supervision2 = nn.ModuleList(
                [nn.Conv2d(in_channels=512, out_channels=c, kernel_size=1) for c in classes])

            # build feature fusion module
            self.feature_fusion_module = FeatureFusionModule(32, 1024)
            #self.FFM = nn.ModuleList(
            #    [FeatureFusionModule(c, 1024) for c in classes]
            #)
        else:
            print('Error: unspport context_path network \n')


        # build final list of classifiers (convolutions)
        self.cls = nn.ModuleList(
            [nn.Conv2d(in_channels=32, out_channels=c, kernel_size=1) for c in classes]
        )

        self.init_weight()

        self.mul_lr = []
        self.mul_lr.append(self.saptial_path)
        self.mul_lr.append(self.attention_refinement_module1)
        self.mul_lr.append(self.attention_refinement_module2)
        self.mul_lr.append(self.supervision1)
        self.mul_lr.append(self.supervision2)
        self.mul_lr.append(self.feature_fusion_module)
        self.mul_lr.append(self.cls)

    def init_weight(self):
        for name, m in self.named_modules():
            if 'context_path' not in name:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-5
                    m.momentum = 0.1
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                    
    def init_new_classifier(self):
        cls = self.cls[-1]
        imprinting_w = self.cls[0].weight[0]
        bkg_bias = self.cls[0].bias[0]

        bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).cuda()

        new_bias = (bkg_bias - bias_diff)

        cls.weight.data.copy_(imprinting_w)
        cls.bias.data.copy_(new_bias)

        self.cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    def init_new_classifier(self):
        cls = self.cls[-1]
        imprinting_w = self.cls[0].weight[0]
        bkg_bias = self.cls[0].bias[0]

        bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).cuda()

        new_bias = (bkg_bias - bias_diff)

        cls.weight.data.copy_(imprinting_w)
        cls.bias.data.copy_(new_bias)

        self.cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    @autocast()
    def forward(self, input):

        out_size = input.shape[-2:]     # aggiunto in prova per interpolazione finale

        # output of spatial path
        sx = self.saptial_path(input)

        # output of context path
        cx1, cx2, tail = self.context_path(input)
        cx1 = self.attention_refinement_module1(cx1)
        cx2 = self.attention_refinement_module2(cx2)
        cx2 = torch.mul(cx2, tail)
        # upsampling
        cx1 = torch.nn.functional.interpolate(cx1, size=sx.size()[-2:], mode='bilinear')
        cx2 = torch.nn.functional.interpolate(cx2, size=sx.size()[-2:], mode='bilinear')
        cx = torch.cat((cx1, cx2), dim=1)

        if self.training == True:
            cx1_sup_l = []
            for mod in self.supervision1:
                cx1_sup_l.append(mod(cx1))
            cx1_sup = torch.cat(cx1_sup_l, dim=1)
            #cx1_sup = self.supervision1(cx1)
            cx2_sup_list = []
            for mod in self.supervision2:
                cx2_sup_list.append(mod(cx2))
            cx2_sup = torch.cat(cx2_sup_list, dim=1)

            #cx2_sup = self.supervision2(cx2)
            cx1_sup = torch.nn.functional.interpolate(cx1_sup, size=input.size()[-2:], mode='bilinear')
            cx2_sup = torch.nn.functional.interpolate(cx2_sup, size=input.size()[-2:], mode='bilinear')

        # output of feature fusion module
        #result_list = []
        #for mod in self.FFM:
        #    result_list.append(mod(sx, cx))
        #result = torch.cat(result_list, dim=1)      # dimensions: (batch_size, classes, height, width)
        result = self.feature_fusion_module(sx, cx)

        # upsampling
        result = torch.nn.functional.interpolate(result, scale_factor=8, mode='bilinear')

        out = []
        for mod in self.cls:
            out.append(mod(result))
        out_result = torch.cat(out, dim=1)

        # result = self.conv(result)          # final convolution == actual classifier

        out_result = torch.nn.functional.interpolate(out_result, size=out_size, mode="bilinear", align_corners=False)

        if self.training == True:
            return out_result, cx1_sup, cx2_sup

        return out_result, {}


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = BiSeNet(32, 'resnet18')
    # model = nn.DataParallel(model)

    model = model.cuda()
    x = torch.rand(2, 3, 256, 256)
    record = model.parameters()
    # for key, params in model.named_parameters():
    #     if 'bn' in key:
    #         params.requires_grad = False
    from utils import group_weight
    # params_list = []
    # for module in model.mul_lr:
    #     params_list = group_weight(params_list, module, nn.BatchNorm2d, 10)
    # params_list = group_weight(params_list, model.context_path, torch.nn.BatchNorm2d, 1)

    print(model.parameters())
