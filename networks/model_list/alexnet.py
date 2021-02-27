import torch
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['AlexNet', 'alexnet']

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

#the net is taken from https://github.com/JackonYang/captcha-tensorflow
#this is not AlexNet anymore
class AlexNet(nn.Module):

    def __init__(self, num_classes=None, nr_digits=8):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.nr_digits = nr_digits
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 3, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(3),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.nr_digits * self.num_classes),
            #nn.Softmax(dim=1) #cross entropy expects raw inputs in pytorch
        )

    def init_weights(self):
        def init_sequential(m):
            if type(m) in [nn.Conv2d, nn.Linear]:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.features.apply(init_sequential)
        self.classifier.apply(init_sequential)

    def forward(self, x):
        #print(x.shape)
        x = self.features(x)
        #print(x.shape)
        x = self.classifier(x)
        #print(x.shape)
        x = x.view((x.shape[0], self.nr_digits, self.num_classes))
        return x

# class AlexNet(nn.Module):
#
#     def __init__(self, num_classes=11, nr_digits=8):
#         super(AlexNet, self).__init__()
#         self.num_classes = num_classes
#         self.nr_digits = nr_digits
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
#             nn.ReLU(inplace=True),
#             LRN(local_size=5, alpha=0.0001, beta=0.75),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
#             nn.ReLU(inplace=True),
#             LRN(local_size=5, alpha=0.0001, beta=0.75),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(256, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(1024, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, self.num_classes),
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         print('----------------',x.shape)
#         x = x.view(self.nr_digits, 1024)
#         print('----------------', x.shape)
#         x = self.classifier(x)
#         print('----------------', x.shape)
#         x = x.view((self.nr_digits * self.num_classes))
#         return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model_path = 'model_list/alexnet.pth.tar'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['state_dict'])
    return model
