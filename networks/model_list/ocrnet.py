import torch
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['ocrnet']

# common 3x3 conv with BN and ReLU
def conv_bn(inp, oup, stride, if_bias=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=if_bias),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


# common 1x1 conv with BN and ReLU
def conv_1x1_bn(inp, oup, if_bias=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=if_bias),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

# ==========================[2] public function module ============================
def channel_shuffle(x, groups):
    # 如果 x 是一个 Variable，那么 x.data 则是一个 Tensor
    # 解析维度信息
    batchsize, num_channels, height, width = x.data.size()
    # 整除
    channels_per_group = num_channels // groups
    # reshape，view函数旨在reshape张量形状
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    # transpose操作2D矩阵的转置，即只能转换两个维度
    # contiguous一般与transpose，permute,view搭配使用，使之变连续
    # 即使用transpose或permute进行维度变换后，调用contiguous，然后方可使用view对维度进行变形
    # 1，2维度互换
    # Size(batchsize, channels_per_group, groups, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

# original ShuffleNetV2 Block, together with stride=2 and stride=1 modules
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        assert benchmodel in (1, 2)
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:  # 对应block  c，有通道拆分
            # assert inp == oup_inc
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
        elif self.benchmodel == 2:  # 对应block  d，无通道拆分
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )

    # python staticmethod 返回函数的静态方法
    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        # torch.cat((A,B),dim)，第二维度拼接，通道数维度
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))
        return channel_shuffle(out, 2)  # 两路通道混合

# for recognition
# This module also can change the channels!
class ParallelDownBlock(nn.Module):
    def __init__(self, chn_in, chn_out, mode='max', stride=(2, 2)):
        super(ParallelDownBlock, self).__init__()
        assert mode in ('max', 'mean')
        assert stride in ((2, 2), (1, 2), (1, 4))
        if stride == (2, 2):
            chn_mid = chn_in // 2
            self.branch1 = nn.Sequential(
                conv_1x1_bn(inp=chn_in, oup=chn_mid, if_bias=False),
                conv_bn(inp=chn_mid, oup=chn_in, stride=2, if_bias=False),
            )
            if mode == 'max':
                self.branch2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            else:
                self.branch2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.last_conv = conv_1x1_bn(inp=chn_in * 2, oup=chn_out, if_bias=False)
        elif stride == (1, 2):
            chn_mid = chn_in // 4
            self.branch1 = nn.Sequential(
                conv_1x1_bn(inp=chn_in, oup=chn_mid, if_bias=False),
                nn.Conv2d(chn_mid, chn_in, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), bias=False),
                nn.BatchNorm2d(chn_in),
                nn.ReLU(inplace=True)
            )
            if mode == 'max':
                self.branch2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
            else:
                self.branch2 = nn.AvgPool2d(kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
            self.last_conv = conv_1x1_bn(inp=chn_in * 2, oup=chn_out, if_bias=False)
        elif stride == (1, 4):
            chn_mid = chn_in // 4
            self.branch1 = nn.Sequential(
                conv_1x1_bn(inp=chn_in, oup=chn_mid, if_bias=False),
                nn.Conv2d(chn_mid, chn_in, kernel_size=(1, 7), stride=(1, 4), padding=(0, 3), bias=False),
                nn.BatchNorm2d(chn_in),
                nn.ReLU(inplace=True)
            )
            if mode == 'max':
                self.branch2 = nn.MaxPool2d(kernel_size=(3, 5), stride=(1, 4), padding=(1, 2))
            else:
                self.branch2 = nn.AvgPool2d(kernel_size=(3, 5), stride=(1, 4), padding=(1, 2))
            self.last_conv = conv_1x1_bn(inp=chn_in * 2, oup=chn_out, if_bias=False)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.last_conv(x)
        return x

# for recognition
class GlobalAvgContextEnhanceBlock(nn.Module):
    def __init__(self, chn_in, size_hw_in, chn_shrink_ratio=6):
        """
        :param chn_in:
        :param size_hw_in: (h, w) or [h, w]
        :param chn_shrink_ratio:
        """
        super(GlobalAvgContextEnhanceBlock, self).__init__()
        self.H, self.W = size_hw_in
        chn_out = chn_in // chn_shrink_ratio
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(chn_in, chn_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LayerNorm([chn_out, 1, 1]),
            # nn.ReLU(inplace=True)
        )
        self.BN = nn.BatchNorm2d(chn_out)

    def forward(self, x):
        x_avg = self.layers(x)  # size(B, chn_out)
        x_avg = x_avg.repeat((1, 1, self.H, self.W))  # size(B, chn_out, H, W)
        x_avg = self.BN(x_avg)
        return torch.cat([x, x_avg], dim=1)

class SSNetRegOriginal(nn.Module):  # SSNetRegOriginal
    def __init__(self, class_num):
        super(SSNetRegOriginal, self).__init__()
        self.stage0 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            InvertedResidual(24, 24, 1, 1)
        )
        # stage 1
        self.down_layer1 = ParallelDownBlock(chn_in=24, chn_out=40, mode='max', stride=(2, 2))  # (72, 24)
        self.stage1 = nn.Sequential(
            InvertedResidual(40, 40, 1, 1),
            InvertedResidual(40, 40, 1, 1)
        )
        self.avg_context1 = ParallelDownBlock(chn_in=40, chn_out=40, mode='mean', stride=(1, 4))

        # stage 2
        self.down_layer2 = ParallelDownBlock(chn_in=40, chn_out=64, mode='max', stride=(1, 2))  # (36, 24)
        self.stage2 = nn.Sequential(
            InvertedResidual(64, 64, 1, 1),
            InvertedResidual(64, 64, 1, 1),
        )

        # fc enhance, out channel == 72, (64 + 64 // 8)
        self.enhance2 = GlobalAvgContextEnhanceBlock(chn_in=64, size_hw_in=(24, 36), chn_shrink_ratio=8)
        self.avg_context2 = ParallelDownBlock(chn_in=72, chn_out=72, mode='mean', stride=(1, 2))

        # stage 3
        self.down_layer3 = ParallelDownBlock(chn_in=72, chn_out=80, mode='max', stride=(1, 2))  # (18, 24)
        self.stage3 = nn.Sequential(
            InvertedResidual(80, 80, 1, 1),
            InvertedResidual(80, 80, 1, 1),
            InvertedResidual(80, 80, 1, 1),
        )

        # chn_in = 80 + 72 + 40, chn_out = 192 + 192 // 8 = 216
        self.enhance_last = GlobalAvgContextEnhanceBlock(chn_in=192, size_hw_in=(24, 18), chn_shrink_ratio=8)

        fc_out_chn = 216
        post_in_chn = fc_out_chn
        post_out_chn = post_in_chn // 2
        self.postprocessor = nn.Sequential(
            nn.Conv2d(in_channels=post_in_chn, out_channels=post_out_chn, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.BatchNorm2d(post_out_chn),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=post_out_chn, out_channels=post_out_chn, kernel_size=(13, 1), stride=(1, 1), padding=(6, 0)),
            nn.BatchNorm2d(post_out_chn),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(4, 3), stride=(4, 1), padding=(0, 1))
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=post_out_chn, out_channels=class_num, kernel_size=1, stride=1)
        )

    def get_net_embedded_size_for_ctc(self):
        #T=input length
        return 18

    def forward(self, x):
        x0 = self.stage0(x)  # up to 0.015

        x1 = self.down_layer1(x0)
        x1 = x1 + self.stage1(x1)  # up to 0.037

        x2 = self.down_layer2(x1)
        x2 = x2 + self.stage2(x2)
        x2 = self.enhance2(x2)

        x3_1 = self.down_layer3(x2)
        x3_2 = self.stage3[0](x3_1)
        x3_3 = self.stage3[1](x3_2)
        x3_4 = self.stage3[2](x3_1 + x3_3)
        x3 = x3_2 + x3_4

        x_concat = torch.cat([self.avg_context1(x1), self.avg_context2(x2), x3], dim=1)
        # print(x_cat.shape)
        x = self.enhance_last(x_concat)  # up to 0.066, out size(1, 216, 24, 24)
        x = self.container(self.postprocessor(x))  # size(1, class_num, 6, 12)
        logits = torch.mean(x, dim=2)  # size(1, class_num, 18)""""""
        logits = logits.permute(2, 0, 1)
        return logits

def ocrnet(**kwargs):
    model = SSNetRegOriginal(kwargs['num_classes'])
    #when loading checkpoint to(device) should be made after that
    model.to(kwargs['device'])

    return model
