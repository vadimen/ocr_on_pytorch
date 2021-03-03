import torch
import torch.nn as nn


__all__ = ['OCRNET', 'ocrnet']

#the net is taken from https://github.com/JackonYang/captcha-tensorflow
class OCRNET(nn.Module):

    def __init__(self, num_classes=None, nr_digits=8):
        super(OCRNET, self).__init__()
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
        x = self.features(x)
        x = self.classifier(x)
        x = x.view((x.shape[0], self.nr_digits, self.num_classes))
        return x

def ocrnet(pretrained=False, **kwargs):
    model = OCRNET(**kwargs)
    if pretrained:
        model_path = '../checkpoint_v1_1.pth.tar'
        if torch.cuda.is_available():
            pretrained_model = torch.load(model_path)
        else:
            pretrained_model = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(pretrained_model['state_dict'])
    return model
