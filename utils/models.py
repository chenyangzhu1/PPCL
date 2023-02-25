import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# import settings
from torch.autograd import Variable
import numpy as np
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class ImgNet(nn.Module):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()
        self.alexnet = torchvision.models.alexnet(pretrained=True)
        self.alexnet.classifier = nn.Sequential(
            *list(self.alexnet.classifier.children())[:6])
        self.fc_encode = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        x = self.alexnet.features(x)
        x = x.view(x.size(0), -1)
        feat = self.alexnet.classifier(x)
        hid = self.fc_encode(feat)
        code = torch.tanh(self.alpha * hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class Img_subnet(nn.Module):
    def __init__(self, code_len):
        super(Img_subnet, self).__init__()
        #这里后面要换成vgg16
        self.alexnet = torchvision.models.alexnet(pretrained=True)
        self.alexnet.classifier = nn.Sequential(
            *list(self.alexnet.classifier.children())[:6])
        self.fc_encode = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        x = self.alexnet.features(x)
        x = x.view(x.size(0), -1)
        feat = self.alexnet.classifier(x)
        hid = self.fc_encode(feat)
        code = torch.tanh(self.alpha * hid)

        return feat, hid,code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

class Combine_net(nn.Module):
    def __init__(self,code_len):
        super(Combine_net,self).__init__()
        self.fc1=nn.Linear(532,510)
        self.fc2=nn.Linear(510,code_len)

    def forward(self,x):
        x=self.fc1(x)
        x=self.fc2(x)
        return x
        

class domain_classifier_I(nn.Module):
    def __init__(self):
        super(domain_classifier_I, self).__init__()
        self.alexnet = torchvision.models.alexnet(pretrained=True)
        self.alexnet.classifier = nn.Sequential(
            *list(self.alexnet.classifier.children())[:6])
        cl1 = nn.Linear(4096, 2048)
        cl2 = nn.Linear(2048, 2048)

        self.domain_classifier = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = self.alexnet.features(x)
        x = x.view(x.size(0), -1)
        feat = self.alexnet.classifier(x)
        C_I = self.domain_classifier(feat)
        C_I1 = C_I.max(1, keepdim=True)[1]

        return C_I, C_I1

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class domain_classifier_T(nn.Module):
    def __init__(self, txt_feat_len):
        super(domain_classifier_T, self).__init__()
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module(
            'd_fc1', nn.Linear(txt_feat_len, 4096))
        self.domain_classifier.add_module('d_fc2', nn.Linear(4096, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax())

    def forward(self, x):

        C_I = self.domain_classifier(x)
        C_I1 = C_I.max(1, keepdim=True)[1]

        return C_I, C_I1

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class TxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc2 = nn.Linear(4096, code_len)
        self.alpha = 1.0

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module(
            'd_fc1', nn.Linear(txt_feat_len, 4096))
        self.domain_classifier.add_module('d_fc2', nn.Linear(4096, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax())

    def forward(self, x):
        feat = F.relu(self.fc1(x))
        hid = self.fc2(feat)
        code = torch.tanh(self.alpha * hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x, alpha):
        x = self.encoder(x)
        x = self.decoder(x)
        reverse_feat = ReverseLayerF.apply(x, alpha)
        return x, reverse_feat


class Autoencoder1(nn.Module):
    def __init__(self):
        super(Autoencoder1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x, alpha, eps):
        noise_x = self.encoder(x)
        noise_x = self.decoder(noise_x)
        noise_x = noise_x * eps
        noise = torch.clamp(x + noise_x, 0, 1)
        atkdata = ReverseLayerF.apply(noise, alpha)

        # return noise_x, atkdata
        return noise_x, atkdata


class JDSHImgNet(nn.Module):
    def __init__(self, code_len):
        super(JDSHImgNet, self).__init__()
        self.alexnet = torchvision.models.alexnet(pretrained=True)
        self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
        self.hash_layer = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        with torch.no_grad():
            x = self.alexnet.features(x)
            x = x.view(x.size(0), -1)
            feat = self.alexnet.classifier(x)

        hid = self.hash_layer(feat)
        feat = F.normalize(feat, dim=1)
        code = torch.tanh(self.alpha * hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class JDSHTxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(JDSHTxtNet, self).__init__()

        self.net = nn.Sequential(nn.Linear(txt_feat_len, 4096),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(4096, code_len),
                                 )

        self.alpha = 1.0

    def forward(self, x):
        hid = self.net(x)
        code = torch.tanh(self.alpha * hid)
        return 1, hid, code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)
