import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import ops
from torchvision.models.resnet import ResNet, BasicBlock
import functools
from collections import OrderedDict
from Models.DCNV1 import Deform_Conv_V1
from Models.ops.deformable_conv import ModulatedDeformConvPack

ENCODER_RESNET = [
    'resnet18', 'resnet34', 'resnet50', 'resnet50_deform_v1', 'resnet50_deform_v2', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d'
]
ENCODER_DENSENET = [
    'densenet121', 'densenet169', 'densenet161', 'densenet201'
]


def lr_pad(x, padding=1):
    ''' Pad left/right-most to each other instead of zero padding '''
    return torch.cat([x[..., -padding:], x, x[..., :padding]], dim=3)


class LR_PAD(nn.Module):
    ''' Pad left/right-most to each other instead of zero padding '''
    def __init__(self, padding=1):
        super(LR_PAD, self).__init__()
        self.padding = padding

    def forward(self, x):
        return lr_pad(x, self.padding)


def wrap_lr_pad(net):
    for name, m in net.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if m.padding[1] == 0:
            continue
        w_pad = int(m.padding[1])
        m.padding = (m.padding[0], 0)
        names = name.split('.')
        root = functools.reduce(lambda o, i: getattr(o, i), [net] + names[:-1])
        setattr(
            root, names[-1],
            nn.Sequential(LR_PAD(w_pad), m)
        )


'''
Encoder
'''
class Resnet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(Resnet, self).__init__()
        assert backbone in ENCODER_RESNET
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        del self.encoder.fc, self.encoder.avgpool

    def list_blocks(self):
        lst = [m for m in self.encoder.children()]
        block0 = lst[:4]
        block1 = lst[4:5]
        block2 = lst[5:6]
        block3 = lst[6:7]
        block4 = lst[7:8]
        return block0, block1, block2, block3, block4

class Resnet_Deform_V1(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(Resnet_Deform_V1, self).__init__()
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        self.encoder.conv1 = Deform_Conv_V1(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder.layer1[0].conv1 = Deform_Conv_V1(64, 64, kernel_size=1, stride=1, bias=False)
        self.encoder.layer1[0].conv2 = Deform_Conv_V1(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.layer1[0].conv3 = Deform_Conv_V1(64, 256, kernel_size=1, stride=1, bias=False)
        self.encoder.layer1[0].downsample[0] = Deform_Conv_V1(64, 256, kernel_size=1, stride=1, bias=False)
        self.encoder.layer1[1].conv1 = Deform_Conv_V1(256, 64, kernel_size=1, stride=1, bias=False)
        self.encoder.layer1[1].conv2 = Deform_Conv_V1(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.layer1[1].conv3 = Deform_Conv_V1(64, 256, kernel_size=1, stride=1, bias=False)
        self.encoder.layer1[2].conv1 = Deform_Conv_V1(256, 64, kernel_size=1, stride=1, bias=False)
        self.encoder.layer1[2].conv2 = Deform_Conv_V1(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.layer1[2].conv3 = Deform_Conv_V1(64, 256, kernel_size=1, stride=1, bias=False)

        self.encoder.layer2[0].conv1 = Deform_Conv_V1(256, 128, kernel_size=1, stride=1, bias=False)
        self.encoder.layer2[0].conv2 = Deform_Conv_V1(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.encoder.layer2[0].conv3 = Deform_Conv_V1(128, 512, kernel_size=1, stride=1, bias=False)
        self.encoder.layer2[0].downsample[0] = Deform_Conv_V1(256, 512, kernel_size=1, stride=2, bias=False)
        self.encoder.layer2[1].conv1 = Deform_Conv_V1(512, 128, kernel_size=1, stride=1, bias=False)
        self.encoder.layer2[1].conv2 = Deform_Conv_V1(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.layer2[1].conv3 = Deform_Conv_V1(128, 512, kernel_size=1, stride=1, bias=False)
        self.encoder.layer2[2].conv1 = Deform_Conv_V1(512, 128, kernel_size=1, stride=1, bias=False)
        self.encoder.layer2[2].conv2 = Deform_Conv_V1(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.layer2[2].conv3 = Deform_Conv_V1(128, 512, kernel_size=1, stride=1, bias=False)
        self.encoder.layer2[3].conv1 = Deform_Conv_V1(512, 128, kernel_size=1, stride=1, bias=False)
        self.encoder.layer2[3].conv2 = Deform_Conv_V1(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.layer2[3].conv3 = Deform_Conv_V1(128, 512, kernel_size=1, stride=1, bias=False)

        self.encoder.layer3[0].conv1 = Deform_Conv_V1(512, 256, kernel_size=1, stride=1, bias=False)
        self.encoder.layer3[0].conv2 = Deform_Conv_V1(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.encoder.layer3[0].conv3 = Deform_Conv_V1(256, 1024, kernel_size=1, stride=1, bias=False)
        self.encoder.layer3[0].downsample[0] = Deform_Conv_V1(512, 1024, kernel_size=1, stride=2, bias=False)
        self.encoder.layer3[1].conv1 = Deform_Conv_V1(1024, 256, kernel_size=1, stride=1, bias=False)
        self.encoder.layer3[1].conv2 = Deform_Conv_V1(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.layer3[1].conv3 = Deform_Conv_V1(256, 1024, kernel_size=1, stride=1, bias=False)
        self.encoder.layer3[2].conv1 = Deform_Conv_V1(1024, 256, kernel_size=1, stride=1, bias=False)
        self.encoder.layer3[2].conv2 = Deform_Conv_V1(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.layer3[2].conv3 = Deform_Conv_V1(256, 1024, kernel_size=1, stride=1, bias=False)
        self.encoder.layer3[3].conv1 = Deform_Conv_V1(1024, 256, kernel_size=1, stride=1, bias=False)
        self.encoder.layer3[3].conv2 = Deform_Conv_V1(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.layer3[3].conv3 = Deform_Conv_V1(256, 1024, kernel_size=1, stride=1, bias=False)
        self.encoder.layer3[4].conv1 = Deform_Conv_V1(1024, 256, kernel_size=1, stride=1, bias=False)
        self.encoder.layer3[4].conv2 = Deform_Conv_V1(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.layer3[4].conv3 = Deform_Conv_V1(256, 1024, kernel_size=1, stride=1, bias=False)
        self.encoder.layer3[5].conv1 = Deform_Conv_V1(1024, 256, kernel_size=1, stride=1, bias=False)
        self.encoder.layer3[5].conv2 = Deform_Conv_V1(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.layer3[5].conv3 = Deform_Conv_V1(256, 1024, kernel_size=1, stride=1, bias=False)

        # self.encoder.layer4[0].conv1 = Deform_Conv_V1(1024, 512, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer4[0].conv2 = Deform_Conv_V1(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        # self.encoder.layer4[0].conv3 = Deform_Conv_V1(512, 2048, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer4[0].downsample[0] = Deform_Conv_V1(1024, 2048, kernel_size=1, stride=2, bias=False)
        # self.encoder.layer4[1].conv1 = Deform_Conv_V1(2048, 512, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer4[1].conv2 = Deform_Conv_V1(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        # self.encoder.layer4[1].conv3 = Deform_Conv_V1(512, 2048, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer4[2].conv1 = Deform_Conv_V1(2048, 512, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer4[2].conv2 = Deform_Conv_V1(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        # self.encoder.layer4[2].conv3 = Deform_Conv_V1(512, 2048, kernel_size=1, stride=1, bias=False)

        del self.encoder.fc, self.encoder.avgpool

    def forward(self, x):
        features = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x);  features.append(x)  # 1/4
        x = self.encoder.layer2(x);  features.append(x)  # 1/8
        x = self.encoder.layer3(x);  features.append(x)  # 1/16
        x = self.encoder.layer4(x);  features.append(x)  # 1/32
        return features

    def list_blocks(self):
        lst = [m for m in self.encoder.children()]
        block0 = lst[:4]
        block1 = lst[4:5]
        block2 = lst[5:6]
        block3 = lst[6:7]
        block4 = lst[7:8]
        return block0, block1, block2, block3, block4

class Resnet_Deform_V2(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(Resnet_Deform_V2, self).__init__()
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        # self.encoder.conv1 = ModulatedDeformConvPack(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.encoder.layer1[0].conv1 = ModulatedDeformConvPack(64, 64, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer1[0].conv2 = ModulatedDeformConvPack(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.encoder.layer1[0].conv3 = ModulatedDeformConvPack(64, 256, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer1[0].downsample[0] = ModulatedDeformConvPack(64, 256, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer1[1].conv1 = ModulatedDeformConvPack(256, 64, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer1[1].conv2 = ModulatedDeformConvPack(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.encoder.layer1[1].conv3 = ModulatedDeformConvPack(64, 256, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer1[2].conv1 = ModulatedDeformConvPack(256, 64, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer1[2].conv2 = ModulatedDeformConvPack(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.encoder.layer1[2].conv3 = ModulatedDeformConvPack(64, 256, kernel_size=1, stride=1, bias=False)
        #
        # self.encoder.layer2[0].conv1 = ModulatedDeformConvPack(256, 128, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer2[0].conv2 = ModulatedDeformConvPack(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        # self.encoder.layer2[0].conv3 = ModulatedDeformConvPack(128, 512, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer2[0].downsample[0] = ModulatedDeformConvPack(256, 512, kernel_size=1, stride=2, bias=False)
        # self.encoder.layer2[1].conv1 = ModulatedDeformConvPack(512, 128, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer2[1].conv2 = ModulatedDeformConvPack(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        # self.encoder.layer2[1].conv3 = ModulatedDeformConvPack(128, 512, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer2[2].conv1 = ModulatedDeformConvPack(512, 128, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer2[2].conv2 = ModulatedDeformConvPack(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        # self.encoder.layer2[2].conv3 = ModulatedDeformConvPack(128, 512, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer2[3].conv1 = ModulatedDeformConvPack(512, 128, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer2[3].conv2 = ModulatedDeformConvPack(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        # self.encoder.layer2[3].conv3 = ModulatedDeformConvPack(128, 512, kernel_size=1, stride=1, bias=False)

        # self.encoder.layer3[0].conv1 = ModulatedDeformConvPack(512, 256, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer3[0].conv2 = ModulatedDeformConvPack(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        # self.encoder.layer3[0].conv3 = ModulatedDeformConvPack(256, 1024, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer3[0].downsample[0] = ModulatedDeformConvPack(512, 1024, kernel_size=1, stride=2, bias=False)
        # self.encoder.layer3[1].conv1 = ModulatedDeformConvPack(1024, 256, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer3[1].conv2 = ModulatedDeformConvPack(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        # self.encoder.layer3[1].conv3 = ModulatedDeformConvPack(256, 1024, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer3[2].conv1 = ModulatedDeformConvPack(1024, 256, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer3[2].conv2 = ModulatedDeformConvPack(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        # self.encoder.layer3[2].conv3 = ModulatedDeformConvPack(256, 1024, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer3[3].conv1 = ModulatedDeformConvPack(1024, 256, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer3[3].conv2 = ModulatedDeformConvPack(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        # self.encoder.layer3[3].conv3 = ModulatedDeformConvPack(256, 1024, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer3[4].conv1 = ModulatedDeformConvPack(1024, 256, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer3[4].conv2 = ModulatedDeformConvPack(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        # self.encoder.layer3[4].conv3 = ModulatedDeformConvPack(256, 1024, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer3[5].conv1 = ModulatedDeformConvPack(1024, 256, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer3[5].conv2 = ModulatedDeformConvPack(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        # self.encoder.layer3[5].conv3 = ModulatedDeformConvPack(256, 1024, kernel_size=1, stride=1, bias=False)

        # self.encoder.layer4[0].conv1 = ModulatedDeformConvPack(1024, 512, kernel_size=1, stride=1, bias=False)
        self.encoder.layer4[0].conv2 = ModulatedDeformConvPack(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        # self.encoder.layer4[0].conv3 = ModulatedDeformConvPack(512, 2048, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer4[0].downsample[0] = ModulatedDeformConvPack(1024, 2048, kernel_size=1, stride=2, bias=False)
        # self.encoder.layer4[1].conv1 = ModulatedDeformConvPack(2048, 512, kernel_size=1, stride=1, bias=False)
        self.encoder.layer4[1].conv2 = ModulatedDeformConvPack(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        # self.encoder.layer4[1].conv3 = ModulatedDeformConvPack(512, 2048, kernel_size=1, stride=1, bias=False)
        # self.encoder.layer4[2].conv1 = ModulatedDeformConvPack(2048, 512, kernel_size=1, stride=1, bias=False)
        self.encoder.layer4[2].conv2 = ModulatedDeformConvPack(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        # self.encoder.layer4[2].conv3 = ModulatedDeformConvPack(512, 2048, kernel_size=1, stride=1, bias=False)

        del self.encoder.fc, self.encoder.avgpool

    def forward(self, x):
        features = []
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)

        x = self.encoder.layer1(x);  features.append(x)  # 1/4
        x = self.encoder.layer2(x);  features.append(x)  # 1/8
        x = self.encoder.layer3(x);  features.append(x)  # 1/16
        x = self.encoder.layer4(x);  features.append(x)  # 1/32
        return features

    def list_blocks(self):
        lst = [m for m in self.encoder.children()]
        block0 = lst[:4]
        block1 = lst[4:5]
        block2 = lst[5:6]
        block3 = lst[6:7]
        block4 = lst[7:8]
        return block0, block1, block2, block3, block4


class Densenet(nn.Module):
    def __init__(self, backbone='densenet169', pretrained=True):
        super(Densenet, self).__init__()
        assert backbone in ENCODER_DENSENET
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        self.final_relu = nn.ReLU(inplace=True)
        del self.encoder.classifier

    def forward(self, x):
        lst = []
        for m in self.encoder.features.children():
            x = m(x)
            lst.append(x)
        features = [lst[4], lst[6], lst[8], self.final_relu(lst[11])]
        return features

    def list_blocks(self):
        lst = [m for m in self.encoder.features.children()]
        block0 = lst[:4]
        block1 = lst[4:6]
        block2 = lst[6:8]
        block3 = lst[8:10]
        block4 = lst[10:]
        return block0, block1, block2, block3, block4


'''
Decoder
'''
class ConvCompressH(nn.Module):
    ''' Reduce feature height by factor of two '''
    def __init__(self, in_c, out_c, ks=3):
        super(ConvCompressH, self).__init__()
        assert ks % 2 == 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=ks, stride=(2, 2), padding=ks//2),
            # nn.Conv2d(in_c, out_c, kernel_size=ks, stride=(1, 1), padding=ks // 2),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class GlobalHeightConv(nn.Module):
    def __init__(self, in_c, out_c):
        super(GlobalHeightConv, self).__init__()
        self.layer = nn.Sequential(
            ConvCompressH(in_c, in_c//2),
            ConvCompressH(in_c//2, in_c//2),
            ConvCompressH(in_c//2, in_c//4),
            ConvCompressH(in_c // 4, out_c),
        )

    def forward(self, x, out_h, out_w):
        x = self.layer(x)
        assert out_h % x.shape[2] == 0 and out_w % x.shape[3] == 0
        h1 = x.shape[2] / 2
        x = x.contiguous().view(x.shape[0], x.shape[1], int(h1), 2, x.shape[3])#[bs, c, h/2, h, w]
        x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4]) #[bs, c*h, h, w]
        factor_h = out_h // x.shape[2]
        factor_w = out_w // x.shape[3]
        x = torch.cat([x[..., -1:,:], x, x[..., :1,:]], 2)
        x = torch.cat([x[..., -1:], x, x[..., :1]], 3)
        x = F.interpolate(x, size=(out_h + 2 * factor_h, out_w + 2 * factor_w), mode='bilinear', align_corners=False)
        # x = F.interpolate(x, size=(out_c + 2 * factor_c), mode='bilinear', align_corners=False)
        # x = F.interpolate(x, size=(x.shape[2], out_w + 2 * factor), mode='bilinear', align_corners=False)
        # x = x[..., factor_c:-factor_c, :,:]
        x = x[..., factor_h:-factor_h,:]
        x = x[..., factor_w:-factor_w]
        return x  # [bs,c,h,w]


class GlobalHeightStage(nn.Module):
    def __init__(self, c1, c2, c3, c4, out_scale=8):
    # def __init__(self, c1, c2, c3, c4, out_scale=16):
        ''' Process 4 blocks from encoder to single multiscale features '''
        super(GlobalHeightStage, self).__init__()
        self.cs = c1, c2, c3, c4
        self.out_scale = out_scale
        self.ghc_lst = nn.ModuleList([
            GlobalHeightConv(c1, c1//out_scale),
            GlobalHeightConv(c2, c2//out_scale),
            GlobalHeightConv(c3, c3//out_scale),
            GlobalHeightConv(c4, c4//out_scale),
        ])

    def forward(self, conv_list, out_h, out_w):
        assert len(conv_list) == 4
        bs = conv_list[0].shape[0]
        for f, x, out_c in zip(self.ghc_lst, conv_list, self.cs):
            temp1 = f(x, out_h, out_w)
            temp2 = temp1.reshape(bs,-1, out_h, out_w)

        feature = torch.cat([
            f(x, out_h, out_w).reshape(bs,-1, out_h, out_w)
            for f, x, out_c in zip(self.ghc_lst, conv_list, self.cs)
        ], dim=1)
        return feature

'''
OmniNet
'''
class OmniNet(nn.Module):
    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])

    def __init__(self, backbone, use_rnn):
        super(OmniNet, self).__init__()
        self.backbone = backbone
        self.use_rnn = use_rnn
        self.out_scale = 8
        self.step_cols = 16
        self.step_rows = 16
        self.rnn_hidden_size = 512

        # Encoder
        if backbone.startswith('res'):
            if 'deform_v1' in backbone:
                print('####################deform_v1#######################')
                self.feature_extractor = Resnet_Deform_V1()
            elif 'deform_v2' in backbone:
                print('####################deform_v2#######################')
                self.feature_extractor = Resnet_Deform_V2()
            else:
                self.feature_extractor = Resnet(backbone, pretrained=True)
        elif backbone.startswith('dense'):
            self.feature_extractor = Densenet(backbone, pretrained=True)
        else:
            raise NotImplementedError()

        # Inference channels number from each block of the encoder
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 1024, 1024)
            c1, c2, c3, c4 = [b.shape[1] for b in self.feature_extractor(dummy)]
            c_last = (c1*8 + c2*4 + c3*2 + c4*1) // self.out_scale

        # Convert features from 4 blocks of the encoder into B x C x 1 x W'
        self.reduce_height_module = GlobalHeightStage(c1, c2, c3, c4, self.out_scale)

        # 1D prediction
        if self.use_rnn:
            self.bi_rnn = nn.LSTM(input_size=1024,#[c]
                                  hidden_size=self.rnn_hidden_size,
                                  num_layers=2,
                                  dropout=0.5,
                                  batch_first=False,
                                  bidirectional=True)
            self.drop_out = nn.Dropout(0.5)
            self.linear = nn.Linear(in_features=1024,
                                    out_features=2 * self.step_cols * self.step_rows) #[1:cor; 2:cor+bon]
            self.linear.bias.data[0*self.step_cols:1*self.step_cols].fill_(-1) # corner
            self.linear.bias.data[1 * self.step_rows:2 * self.step_rows].fill_(0.425)  # bon

        else: #[未修改]
            self.linear = nn.Sequential(
                nn.Linear(c_last, self.rnn_hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(self.rnn_hidden_size, 3 * self.step_cols),
            )
            self.linear[-1].bias.data[0*self.step_cols:1*self.step_cols].fill_(-1)
            self.linear[-1].bias.data[1*self.step_cols:2*self.step_cols].fill_(-0.478)
            self.linear[-1].bias.data[2*self.step_cols:3*self.step_cols].fill_(0.425)
        self.x_mean.requires_grad = False
        self.x_std.requires_grad = False
        wrap_lr_pad(self)

    def _prepare_x(self, x):
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        return (x[:, :3] - self.x_mean) / self.x_std

    def forward(self, x):
        if x.shape[2] != 1024 or x.shape[3] != 1024:
            raise NotImplementedError()
        x = self._prepare_x(x)
        conv_list = self.feature_extractor(x) #ResNet结果
        feature = self.reduce_height_module(conv_list, x.shape[2]//self.step_cols, x.shape[3]//self.step_rows)
        feature = feature.reshape(feature.shape[0], feature.shape[1], -1) #[bs, c, h*w]
        # rnn
        if self.use_rnn:
            feature = feature.permute(2, 0, 1)  # [w*h, bs, c]
            output, hidden = self.bi_rnn(feature)  # [w*h, bs, num_directions * hidden_size=2*512=1024]
            output = self.drop_out(output)
            output = self.linear(output)  # [w*h, bs, step_cols* step_rows= 16*16=256]
            output = output.view(output.shape[0], output.shape[1], 2, self.step_rows, self.step_cols) #[w*h, bs,type(cor/bon)=1/2/3, step_cols, step_rows]
            output = output.view(64, 64, output.shape[1], output.shape[2], self.step_rows, self.step_cols) #[w, h, b, type(cor/bon)=1/2/3, step_cols, step_rows]
            output = output.permute(2, 3, 0, 4, 1, 5)  # [bs, type(cor/bon)=1/2/3, w, step_rows, h, step_cols]
            output = output.contiguous().view(output.shape[0], output.shape[1], output.shape[2], output.shape[3], -1) # [bs, type(cor/bon)=1/2/3, w, step_rows, h*step_cols=1024]
            output = output.contiguous().view(output.shape[0], output.shape[1], -1, output.shape[4])# [bs, type(cor/bon)=1/2/3, w*step_rows=1024, h*step_cols=1024]
            output = output.permute(1, 0, 2, 3)  # [type(cor/bon)=1/2/3, bs, w*step_rows=1024, h*step_cols=1024]
        else: #[未修改]
            feature = feature.permute(0, 2, 1)
            output = self.linear(feature)
            output = output.view(output.shape[0], output.shape[1], 3, self.step_cols)
            output = output.permute(0, 2, 1, 3)
            output = output.contiguous().view(output.shape[0], 3, -1)

        # output.shape => B x 2 x W
        cor = output[0]  # bs x H x W
        bon = output[1]  # bs x H x W
        return cor, bon

'''
LayoutNet
'''

'''
DulaNet
'''

model = Resnet_Deform_V2()
print(model)