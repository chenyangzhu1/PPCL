import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.stats import truncnorm #extra import equivalent to tf.trunc initialise
import numpy as np

class ImgNet(nn.Module):

    def __init__(self, code_len):
        # Construct nn.Module superclass from the derived classs MultibranchLeNet
        super(ImgNet, self).__init__()
        # Construct MultibranchLeNet architecture
        self.conv1 = nn.Sequential()
        self.conv1.add_module('c1_conv', nn.Conv2d(3, 32, kernel_size=5))
        self.conv1.add_module('c1_relu', nn.ReLU(True))
        self.conv1.add_module('c1_pool', nn.MaxPool2d(2))

        self.conv2 = nn.Sequential()
        self.conv2.add_module('c2_conv', nn.Conv2d(32, 48, kernel_size=5))
        self.conv2.add_module('c2_relu', nn.ReLU(True))
        self.conv2.add_module('c2_pool', nn.MaxPool2d(2))

        self.feature_classifier = nn.Sequential()
        self.feature_classifier.add_module('f_fc1', nn.Linear(48 * 53 * 53, 100))
        # self.feature_classifier.add_module('f_bn1', nn.BatchNorm1d(100))
        self.feature_classifier.add_module('f_relu1', nn.ReLU(True))
        self.feature_classifier.add_module('f_fc2', nn.Linear(100, 100))
        # self.feature_classifier.add_module('f_bn1', nn.BatchNorm1d(100))
        self.feature_classifier.add_module('f_relu2', nn.ReLU(True))
        self.feature_classifier.add_module('f_fc3', nn.Linear(100, code_len))

        self.alpha = 1.0

    def forward(self, input):

        out = self.conv1(input)
        out = self.conv2(out)
        feat = out.view(out.size(0), -1)
        hid = self.feature_classifier(feat)
        code = torch.tanh(self.alpha * hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


class ImgNet1(nn.Module):
    def __init__(self, code_len):
        super(ImgNet1, self).__init__()
        self.alexnet = torchvision.models.alexnet(pretrained=True)
        self.alexnet.classifier = nn.Sequential(*list(self.alexnet.classifier.children())[:6])
        self.fc_encode = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        x = self.alexnet.features(x)
        x = x.view(x.size(0), -1)
        feat = self.alexnet.classifier(x)
        hid = self.fc_encode(feat)
        # code = F.tanh(self.alpha * hid)
        code = torch.tanh(self.alpha * hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)

class VGG_16(nn.Module):
    def __init__(self, code_len):
        super(VGG_16, self).__init__()
        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg16.classifier = nn.Sequential(*list(self.vgg16.classifier.children())[:-1])
        self.fc_encode = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        x = self.vgg16.features(x)
        x = x.view(x.size(0), -1)
        feat = self.vgg16.classifier(x)
        hid = self.fc_encode(feat)
        # code = F.tanh(self.alpha * hid)
        code = torch.tanh(self.alpha * hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


class VGG_19(nn.Module):
    def __init__(self, code_len):
        super(VGG_19, self).__init__()
        self.vgg19 = torchvision.models.vgg19(pretrained=True)
        self.vgg19.classifier = nn.Sequential(*list(self.vgg19.classifier.children())[:-1])
        self.fc_encode = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        x = self.vgg19.features(x)
        x = x.view(x.size(0), -1)
        feat = self.vgg19.classifier(x)
        hid = self.fc_encode(feat)
        # code = F.tanh(self.alpha * hid)
        code = torch.tanh(self.alpha * hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


class resnet18(nn.Module):
    def __init__(self, code_len):
        super(resnet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        self.resnet18.classifier = nn.Sequential(*list(self.resnet18.children())[:-1])
        self.fc_encode = nn.Linear(512, code_len)
        self.alpha = 1.0

    def forward(self, x):
        x = self.resnet18.classifier(x)
        feat = x.view(x.size(0), -1)
        # feat = self.resnet18.classifier(x)
        hid = self.fc_encode(feat)
        # code = F.tanh(self.alpha * hid)
        code = torch.tanh(self.alpha * hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)

class resnet50(nn.Module):
    def __init__(self, code_len):
        super(resnet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet50.classifier = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.fc_encode = nn.Linear(2048, code_len)
        self.alpha = 1.0

    def forward(self, x):
        x = self.resnet50.classifier(x)
        feat = x.view(x.size(0), -1)
        # feat = self.resnet50.classifier(x)
        hid = self.fc_encode(feat)
        # code = F.tanh(self.alpha * hid)
        code = torch.tanh(self.alpha * hid)

        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)

class TxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, 4096)
        self.fc2 = nn.Linear(4096, code_len)
        self.alpha = 1.0

    def forward(self, x):
        feat = F.relu(self.fc1(x))
        hid = self.fc2(feat)
        # code = F.tanh(self.alpha * hid)
        code = torch.tanh(self.alpha * hid)
        return feat, hid, code

    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


def truncated_normal_(self,tensor,mean=0,std=0.09):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size+(4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor


# class LightCNN_Text(nn.Module):
#
#     """
#     A CNN for text classification.
#     Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
#     """
#
#     def __init__(self, code_len, vocab_size, embedding_dim, filter_sizes, num_filters, l2_reg_lambda=0.0001):
#         super(LightCNN_Text, self).__init__()
#         self.embedding_size = 300
#         self.filter_sizes = filter_sizes
#
#         self.embed = nn.Embedding(vocab_size, embedding_dim)
#         for i, filter_size in enumerate(self.filter_sizes):
#             self.layer[i] = self._make_layer(filter_size, num_filters)
#             self.convs = nn.ModuleList([self.layer[i] for i in self.filter_sizes])
#
#         # self.linear = nn.Linear(512*block.expansion, num_classes)
#         # self.conv1 = nn.Conv2d(self.embedding_size, self.embedding_size, kernel_size=3, stride=1, padding=1, bias=False)
#         # self.bn1 = nn.BatchNorm2d(self.embedding_size)
#         # self.re1 = nn.LeakyReLU(inplace=True)
#
#         self.fc1 = nn.Linear(len(self.filter_sizes) * self.embedding_size, 512)
#         self.fc2 = nn.Linear(512, code_len)
#         self.alpha = 1.0
#
#     def _make_layer(self, filter_size, num_filters):
#         layers = []
#         layers.append(nn.BatchNorm2d(3))
#
#         # +Dialated Conv atrous_conv2d
#         if filter_size == 5:
#             layers.append(nn.Conv2d(3, num_filters, kernel_size=3, bias=False, dilation=3))
#
#             # Separable Conv/depth
#             layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=1, bias=False))
#
#         else:
#             layers.append(nn.Conv2d(3, num_filters, kernel_size=filter_size, bias=False))
#
#         layers.append(nn.Conv2d(num_filters, self.embedding_size, kernel_size=1, bias=False))
#
#         # Batch Normalzation
#         layers.append(nn.BatchNorm2d(self.embedding_size))
#         layers.append(self.embedding_size, self.embedding_size, kernel_size=3, stride=1, padding=1, bias=False)
#         layers.append(nn.BatchNorm2d(self.embedding_size))
#         layers.append(nn.LeakyReLU(inplace=True))
#         layers.append(F.max_pool2d(inplace=True))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.embed(x)  # (N, W, D)
#
#         x = x.unsqueeze(1)  # (N, Ci, W, D)
#
#         # x = self.layer(x)  # (N, Ci, W, D)
#         #
#         # x = self.re1(self.bn1(self.conv1(x)))
#
#         x = [conv(x) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
#
#         x = torch.cat(x, 3)
#
#         x = self.dropout(x)  # (N, len(Ks)*Co)
#
#         feat = self.fc1(x)  # (N, C)
#         hid = self.fc2(feat)  # (N, C)
#         code = torch.tanh(self.alpha * hid)
#
#         return feat, hid, code
#
#
#     def set_alpha(self, epoch):
#         self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)


class LightCNN_Text(nn.Module):

    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, code_len, vocab_size, embedding_dim, filter_sizes, num_filters, l2_reg_lambda=0.0001):
        super(LightCNN_Text, self).__init__()
        self.embedding_size = 300
        self.filter_sizes = filter_sizes

        self.embed = nn.Embedding(vocab_size, embedding_dim)
        for i, filter_size in enumerate(self.filter_sizes):
            self.layer[i] = self._make_layer(filter_size, num_filters)
            self.convs = nn.ModuleList([self.layer[i] for i in self.filter_sizes])

        # self.linear = nn.Linear(512*block.expansion, num_classes)
        # self.conv1 = nn.Conv2d(self.embedding_size, self.embedding_size, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.embedding_size)
        # self.re1 = nn.LeakyReLU(inplace=True)

        self.fc1 = nn.Linear(len(self.filter_sizes) * self.embedding_size, 512)
        self.fc2 = nn.Linear(512, code_len)
        self.alpha = 1.0

    def _make_layer(self, filter_size, num_filters):
        layers = []
        layers.append(nn.BatchNorm2d(3))

        # +Dialated Conv atrous_conv2d
        if filter_size == 5:
            layers.append(nn.Conv2d(3, num_filters, kernel_size=3, bias=False, dilation=3))

            # Separable Conv/depth
            layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=1, bias=False))

        else:
            layers.append(nn.Conv2d(3, num_filters, kernel_size=filter_size, bias=False))

        layers.append(nn.Conv2d(num_filters, self.embedding_size, kernel_size=1, bias=False))

        # Batch Normalzation
        layers.append(nn.BatchNorm2d(self.embedding_size))
        layers.append(self.embedding_size, self.embedding_size, kernel_size=3, stride=1, padding=1, bias=False)
        layers.append(nn.BatchNorm2d(self.embedding_size))
        layers.append(nn.LeakyReLU(inplace=True))
        layers.append(F.max_pool2d(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        # x = self.layer(x)  # (N, Ci, W, D)
        #
        # x = self.re1(self.bn1(self.conv1(x)))

        x = [conv(x) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)

        x = torch.cat(x, 3)

        x = self.dropout(x)  # (N, len(Ks)*Co)

        feat = self.fc1(x)  # (N, C)
        hid = self.fc2(feat)  # (N, C)
        code = torch.tanh(self.alpha * hid)

        return feat, hid, code


    def set_alpha(self, epoch):
        self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)

    # def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0001):
    #     layers = []
    #
    #     for i, filter_size in enumerate(filter_sizes):
    #         conv_bn1 = nn.BatchNorm1d(self.embedded_chars_expanded, momentum=0.9)
    #         if filter_size == 5:
    #             # filter_shape = [3, embedding_size, 1, 1]
    #             # W = torch.tensor(truncnorm.rvs(-1, 1, size=[10, 10]))
    #             # b = torch.zeros([num_filters]) + 0.1
    #
    #             # conv2 =torch.nn.functional.conv2d(conv_bn1, W, bias=b, strides=[1, 1, 1, 1], padding=0, dilation=5)
    #             self.conv2 = nn.Conv2d(3, planes, kernel_size=1, bias=False)
    #
    #             # filter_shape3 = [3, embedding_size, 1, 1]
    #             # # W3 = torch.tensor(truncnorm.rvs(-1, 1, size=[10, 10]))
    #             # W3 = torch.tensor(truncated_normal_(-1, 1, size=filter_shape3))
    #             # conv = torch.nn.functional.conv2d(conv2, W3, bias=b, strides=[1, 1, 1, 1], padding=0)
    #
    #             conv = torch.nn.functional.conv2d(conv2, W3, bias=b, strides=[1, 1, 1, 1], padding=0)
    #             # self.conv2 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
    #
    #             ksize_1 = [1, sequence_length - 2, 1, 1]
    #
    #         else:
    #
    #             filter_shape2 = [filter_size, embedding_size, 1, 1]
    #             W2 = torch.tensor(truncnorm.rvs(-1, 1, size=[10, 10]))
    #             b = torch.zeros([num_filters]) + 0.1
    #             conv = torch.nn.functional.conv2d(conv2, W2, bias=b, strides=[1, 1, 1, 1], padding=0)
    #             ksize_1 = [1, sequence_length - filter_size + 1, 1, 1]
    #
    #         # Pointwise Convolution Layer
    #         filter_shape1 = [1, 1, 1, num_filters]
    #         W1 = torch.tensor(truncnorm.rvs(-1, 1, size=[10, 10]))
    #         conv1 = torch.nn.functional.conv2d(conv, W1, bias=b, strides=[1, 1, 1, 1], padding=0)
    #
    #         # Batch Normalzation
    #         conv_bn2 = nn.BatchNorm2d(conv1, momentum=0.9)
    #
    #         # Apply nonlinearity
    #         # h = tf.nn.leaky_relu(tf.nn.bias_add(conv_bn2, b), 0.1, name="leakyRelu")
    #         h = nn.LeakyReLU(conv_bn2)
    #         # h = nn.LeakyReLU(0.2, inplace=True)
    #
    #         # Maxpooling over the outputs
    #
    #         pooled = torch.nn.functional.max_pool(h, kernal_size=ksize_1, strides=[1, 1, 1, 1], padding=0)
    #         pooled_outputs.append(pooled)

    # def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0001):
    #     # Keeping track of l2 regularization loss (optional)
    #
    #     self.l2_loss = torch.Tensor(0.0)
    #
    #     self.W = Variable(np.random.uniform([vocab_size, embedding_size], -1.0, 1.0))
    #     self.embedded_chars = torch.index_select(self.W, 0, self.input_x)
    #     self.embedded_chars_expanded = torch.unsqueeze(self.embedded_chars, -1)
    #
    #     # Create a convolution + maxpool layer for each filter size
    #     pooled_outputs = []
    #
    #     for i, filter_size in enumerate(filter_sizes):
    #         conv_bn1 = nn.BatchNorm1d(self.embedded_chars_expanded, momentum=0.9)
    #         if filter_size == 5:
    #             # W = tf.Variable(tf.random.truncated_normal(filter_shape, stddev=0.1), name="W")
    #             # b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
    #             # conv2 = tf.nn.atrous_conv2d(
    #             #     # self.embedded_chars_expanded,
    #             #     conv_bn1,
    #             #     W,
    #             #     rate=2,
    #             #     padding="SAME",
    #             #     name="conv2"
    #             # )
    #             filter_shape = [3, embedding_size, 1, 1]
    #             W = torch.tensor(truncnorm.rvs(-1, 1, size=[10, 10]))
    #             b = torch.zeros([num_filters]) + 0.1
    #
    #             conv2 =torch.nn.functional.conv2d(conv_bn1, W, bias=b, strides=[1, 1, 1, 1], padding=0, dilation=5)
    #
    #             filter_shape3 = [3, embedding_size, 1, 1]
    #             # W3 = torch.tensor(truncnorm.rvs(-1, 1, size=[10, 10]))
    #             W3 = torch.tensor(truncated_normal_(-1, 1, size=filter_shape3))
    #             conv = torch.nn.functional.conv2d(conv2, W3, bias=b, strides=[1, 1, 1, 1], padding=0)
    #
    #             ksize_1 = [1, sequence_length - 2, 1, 1]
    #
    #         else:
    #
    #             filter_shape2 = [filter_size, embedding_size, 1, 1]
    #             W2 = torch.tensor(truncnorm.rvs(-1, 1, size=[10, 10]))
    #             b = torch.zeros([num_filters]) + 0.1
    #             conv = torch.nn.functional.conv2d(conv2, W2, bias=b, strides=[1, 1, 1, 1], padding=0)
    #             ksize_1 = [1, sequence_length - filter_size + 1, 1, 1]
    #
    #         # Pointwise Convolution Layer
    #         filter_shape1 = [1, 1, 1, num_filters]
    #         W1 = torch.tensor(truncnorm.rvs(-1, 1, size=[10, 10]))
    #         conv1 = torch.nn.functional.conv2d(conv, W1, bias=b, strides=[1, 1, 1, 1], padding=0)
    #
    #         # Batch Normalzation
    #         conv_bn2 = nn.BatchNorm2d(conv1, momentum=0.9)
    #
    #         # Apply nonlinearity
    #         # h = tf.nn.leaky_relu(tf.nn.bias_add(conv_bn2, b), 0.1, name="leakyRelu")
    #         h = nn.LeakyReLU(conv_bn2)
    #         # h = nn.LeakyReLU(0.2, inplace=True)
    #
    #         # Maxpooling over the outputs
    #
    #         pooled = torch.nn.functional.max_pool(h, kernal_size=ksize_1, strides=[1, 1, 1, 1], padding=0)
    #         pooled_outputs.append(pooled)
    #
    #     # Combine all the pooled features
    #     num_filters_total = num_filters * len(filter_sizes)
    #     self.h_pool = torch.concat(pooled_outputs, 3)
    #     self.h_pool_flat = torch.reshape(self.h_pool, [-1, num_filters_total])
    #
    #     # Add dropout
    #     # with tf.name_scope("dropout"):
    #     #      self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
    #     self.dropout = nn.Dropout(self.h_pool_flat)


# class LightCNN_Text(nn.Module):
#
#     """
#     A CNN for text classification.
#     Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
#     """
#
#     def __init__(self, code_len, vocab_size, embedding_dim, filter_sizes, num_filters, l2_reg_lambda=0.0001):
#         super(LightCNN_Text, self).__init__()
#         self.embedding_size = 300
#         self.filter_sizes = filter_sizes
#
#         self.embed = nn.Embedding(vocab_size, embedding_dim)
#         self.layer = self._make_layer(num_filters)
#         # self.linear = nn.Linear(512*block.expansion, num_classes)
#         self.conv1 = nn.Conv2d(self.embedding_size, self.embedding_size, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(self.embedding_size)
#         self.re1 = nn.LeakyReLU(inplace=True)
#
#         self.fc1 = nn.Linear(self.embedding_size, 512)
#         self.fc2 = nn.Linear(512, code_len)
#         self.alpha = 1.0
#
#     def _make_layer(self, num_filters):
#         layers = []
#         for i, filter_size in enumerate(self.filter_sizes):
#             layers.append(nn.BatchNorm2d(3))
#
#             # +Dialated Conv atrous_conv2d
#             if filter_size == 5:
#                layers.append(nn.Conv2d(3, num_filters, kernel_size=3, bias=False, dilation=3))
#
#                # Separable Conv/depth
#                layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=1, bias=False))
#
#             else:
#                layers.append(nn.Conv2d(3, num_filters, kernel_size=filter_size, bias=False))
#
#             layers.append(nn.Conv2d(num_filters, self.embedding_size, kernel_size=1, bias=False))
#
#             # Batch Normalzation
#             layers.append(nn.BatchNorm2d(self.embedding_size))
#             layers.append(self.embedding_size, self.embedding_size, kernel_size=3, stride=1, padding=1, bias=False)
#             layers.append(nn.BatchNorm2d(self.embedding_size))
#             layers.append(nn.LeakyReLU(inplace=True))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.embed(x)  # (N, W, D)
#
#         x = x.unsqueeze(1)  # (N, Ci, W, D)
#
#         x = self.layer(x)  # (N, Ci, W, D)
#
#         x = self.re1(self.bn1(self.conv1(x)))
#
#         x = torch.cat(x, 1)
#
#         x = self.dropout(x)  # (N, len(Ks)*Co)
#
#         feat = self.fc1(x)  # (N, C)
#         hid = self.fc2(feat)  # (N, C)
#         code = torch.tanh(self.alpha * hid)
#
#         return feat, hid, code
#
#
#     def set_alpha(self, epoch):
#         self.alpha  = math.pow((1.0 * epoch + 1.0), 0.5)
#
#     # def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0001):
#     #     layers = []
#     #
#     #     for i, filter_size in enumerate(filter_sizes):
#     #         conv_bn1 = nn.BatchNorm1d(self.embedded_chars_expanded, momentum=0.9)
#     #         if filter_size == 5:
#     #             # filter_shape = [3, embedding_size, 1, 1]
#     #             # W = torch.tensor(truncnorm.rvs(-1, 1, size=[10, 10]))
#     #             # b = torch.zeros([num_filters]) + 0.1
#     #
#     #             # conv2 =torch.nn.functional.conv2d(conv_bn1, W, bias=b, strides=[1, 1, 1, 1], padding=0, dilation=5)
#     #             self.conv2 = nn.Conv2d(3, planes, kernel_size=1, bias=False)
#     #
#     #             # filter_shape3 = [3, embedding_size, 1, 1]
#     #             # # W3 = torch.tensor(truncnorm.rvs(-1, 1, size=[10, 10]))
#     #             # W3 = torch.tensor(truncated_normal_(-1, 1, size=filter_shape3))
#     #             # conv = torch.nn.functional.conv2d(conv2, W3, bias=b, strides=[1, 1, 1, 1], padding=0)
#     #
#     #             conv = torch.nn.functional.conv2d(conv2, W3, bias=b, strides=[1, 1, 1, 1], padding=0)
#     #             # self.conv2 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#     #
#     #             ksize_1 = [1, sequence_length - 2, 1, 1]
#     #
#     #         else:
#     #
#     #             filter_shape2 = [filter_size, embedding_size, 1, 1]
#     #             W2 = torch.tensor(truncnorm.rvs(-1, 1, size=[10, 10]))
#     #             b = torch.zeros([num_filters]) + 0.1
#     #             conv = torch.nn.functional.conv2d(conv2, W2, bias=b, strides=[1, 1, 1, 1], padding=0)
#     #             ksize_1 = [1, sequence_length - filter_size + 1, 1, 1]
#     #
#     #         # Pointwise Convolution Layer
#     #         filter_shape1 = [1, 1, 1, num_filters]
#     #         W1 = torch.tensor(truncnorm.rvs(-1, 1, size=[10, 10]))
#     #         conv1 = torch.nn.functional.conv2d(conv, W1, bias=b, strides=[1, 1, 1, 1], padding=0)
#     #
#     #         # Batch Normalzation
#     #         conv_bn2 = nn.BatchNorm2d(conv1, momentum=0.9)
#     #
#     #         # Apply nonlinearity
#     #         # h = tf.nn.leaky_relu(tf.nn.bias_add(conv_bn2, b), 0.1, name="leakyRelu")
#     #         h = nn.LeakyReLU(conv_bn2)
#     #         # h = nn.LeakyReLU(0.2, inplace=True)
#     #
#     #         # Maxpooling over the outputs
#     #
#     #         pooled = torch.nn.functional.max_pool(h, kernal_size=ksize_1, strides=[1, 1, 1, 1], padding=0)
#     #         pooled_outputs.append(pooled)
#
#     # def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0001):
#     #     # Keeping track of l2 regularization loss (optional)
#     #
#     #     self.l2_loss = torch.Tensor(0.0)
#     #
#     #     self.W = Variable(np.random.uniform([vocab_size, embedding_size], -1.0, 1.0))
#     #     self.embedded_chars = torch.index_select(self.W, 0, self.input_x)
#     #     self.embedded_chars_expanded = torch.unsqueeze(self.embedded_chars, -1)
#     #
#     #     # Create a convolution + maxpool layer for each filter size
#     #     pooled_outputs = []
#     #
#     #     for i, filter_size in enumerate(filter_sizes):
#     #         conv_bn1 = nn.BatchNorm1d(self.embedded_chars_expanded, momentum=0.9)
#     #         if filter_size == 5:
#     #             # W = tf.Variable(tf.random.truncated_normal(filter_shape, stddev=0.1), name="W")
#     #             # b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
#     #             # conv2 = tf.nn.atrous_conv2d(
#     #             #     # self.embedded_chars_expanded,
#     #             #     conv_bn1,
#     #             #     W,
#     #             #     rate=2,
#     #             #     padding="SAME",
#     #             #     name="conv2"
#     #             # )
#     #             filter_shape = [3, embedding_size, 1, 1]
#     #             W = torch.tensor(truncnorm.rvs(-1, 1, size=[10, 10]))
#     #             b = torch.zeros([num_filters]) + 0.1
#     #
#     #             conv2 =torch.nn.functional.conv2d(conv_bn1, W, bias=b, strides=[1, 1, 1, 1], padding=0, dilation=5)
#     #
#     #             filter_shape3 = [3, embedding_size, 1, 1]
#     #             # W3 = torch.tensor(truncnorm.rvs(-1, 1, size=[10, 10]))
#     #             W3 = torch.tensor(truncated_normal_(-1, 1, size=filter_shape3))
#     #             conv = torch.nn.functional.conv2d(conv2, W3, bias=b, strides=[1, 1, 1, 1], padding=0)
#     #
#     #             ksize_1 = [1, sequence_length - 2, 1, 1]
#     #
#     #         else:
#     #
#     #             filter_shape2 = [filter_size, embedding_size, 1, 1]
#     #             W2 = torch.tensor(truncnorm.rvs(-1, 1, size=[10, 10]))
#     #             b = torch.zeros([num_filters]) + 0.1
#     #             conv = torch.nn.functional.conv2d(conv2, W2, bias=b, strides=[1, 1, 1, 1], padding=0)
#     #             ksize_1 = [1, sequence_length - filter_size + 1, 1, 1]
#     #
#     #         # Pointwise Convolution Layer
#     #         filter_shape1 = [1, 1, 1, num_filters]
#     #         W1 = torch.tensor(truncnorm.rvs(-1, 1, size=[10, 10]))
#     #         conv1 = torch.nn.functional.conv2d(conv, W1, bias=b, strides=[1, 1, 1, 1], padding=0)
#     #
#     #         # Batch Normalzation
#     #         conv_bn2 = nn.BatchNorm2d(conv1, momentum=0.9)
#     #
#     #         # Apply nonlinearity
#     #         # h = tf.nn.leaky_relu(tf.nn.bias_add(conv_bn2, b), 0.1, name="leakyRelu")
#     #         h = nn.LeakyReLU(conv_bn2)
#     #         # h = nn.LeakyReLU(0.2, inplace=True)
#     #
#     #         # Maxpooling over the outputs
#     #
#     #         pooled = torch.nn.functional.max_pool(h, kernal_size=ksize_1, strides=[1, 1, 1, 1], padding=0)
#     #         pooled_outputs.append(pooled)
#     #
#     #     # Combine all the pooled features
#     #     num_filters_total = num_filters * len(filter_sizes)
#     #     self.h_pool = torch.concat(pooled_outputs, 3)
#     #     self.h_pool_flat = torch.reshape(self.h_pool, [-1, num_filters_total])
#     #
#     #     # Add dropout
#     #     # with tf.name_scope("dropout"):
#     #     #      self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
#     #     self.dropout = nn.Dropout(self.h_pool_flat)

# class LightCNN_Text(nn.Module):
#
#     """
#     A CNN for text classification.
#     Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
#     """
#     def __init__(self, sequence_length, num_classes, vocab_size,
#       embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0001):
#         # Keeping track of l2 regularization loss (optional)
#         # l2_loss = tf.constant(0.0)
#         self.l2_loss = torch.Tensor(0.0)
#
#         # Embedding layer
#         # with tf.device('/cpu:0'), tf.name_scope("embedding"):
#         # self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
#         # self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
#
#         self.W = Variable(np.random.uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
#         self.embedded_chars = torch.index_select(self.W, 0, self.input_x)
#         self.embedded_chars_expanded = torch.unsqueeze(self.embedded_chars, -1)
#
#         # Create a convolution + maxpool layer for each filter size
#         pooled_outputs = []
#
#         for i, filter_size in enumerate(filter_sizes):
#             # conv_bn1 = tf.layers.batch_normalization(self.embedded_chars_expanded, momentum=0.9)
#             conv_bn1 = nn.BatchNorm1d(self.embedded_chars_expanded, momentum=0.9)
#             if filter_size == 5:
#                 # W = tf.Variable(tf.random.truncated_normal(filter_shape, stddev=0.1), name="W")
#                 # b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
#                 # conv2 = tf.nn.atrous_conv2d(
#                 #     # self.embedded_chars_expanded,
#                 #     conv_bn1,
#                 #     W,
#                 #     rate=2,
#                 #     padding="SAME",
#                 #     name="conv2"
#                 # )
#                 filter_shape = [3, embedding_size, 1, 1]
#                 W = torch.tensor(truncnorm.rvs(-1, 1, size=[10, 10]))
#                 b = torch.zeros([num_filters]) + 0.1
#
#                 # conv2 = torch.nn.functional.conv2d(conv_bn1, W, bias=b, strides=[1, 1, 1, 1], padding=0)
#                 conv2 =torch.nn.functional.conv2d(conv_bn1, W, bias=b, strides=[1, 1, 1, 1], padding=0, dilation=5)
#                 # torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=5)
#
#                 # Seperable Conv/depth
#                 # filter_shape3 = [3, embedding_size, 1, 1]
#                 # W3 = tf.Variable(tf.random.truncated_normal(filter_shape3, stddev=0.1), name="W3")
#                 # conv = tf.nn.conv2d(
#                 #     conv2,
#                 #     W,
#                 #     strides=[1, 1, 1, 1],
#                 #     padding="VALID",
#                 #     name="conv")
#                 #
#                 # print(conv.shape)
#
#                 filter_shape3 = [3, embedding_size, 1, 1]
#                 # W3 = torch.tensor(truncnorm.rvs(-1, 1, size=[10, 10]))
#                 W3 = torch.tensor(truncated_normal_(-1, 1, size=filter_shape3))
#                 conv = torch.nn.functional.conv2d(conv2, W3, bias=b, strides=[1, 1, 1, 1], padding=0)
#
#                 ksize_1 = [1, sequence_length - 2, 1, 1]
#
#             else:
#                 # W2 = tf.Variable(tf.random.truncated_normal(filter_shape2, stddev=0.1), name="W2")
#                 # b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
#                 #
#                 # conv = tf.nn.conv2d(
#                 #     # self.embedded_chars_expanded,
#                 #     conv_bn1,
#                 #     W2,
#                 #     strides=[1, 1, 1, 1],
#                 #     padding="VALID",
#                 #     name="conv")
#                 # print(conv.shape)
#
#                 filter_shape2 = [filter_size, embedding_size, 1, 1]
#                 W2 = torch.tensor(truncnorm.rvs(-1, 1, size=[10, 10]))
#                 b = torch.zeros([num_filters]) + 0.1
#                 conv = torch.nn.functional.conv2d(conv2, W3, bias=b, strides=[1, 1, 1, 1], padding=0)
#                 ksize_1 = [1, sequence_length - filter_size + 1, 1, 1]
#
#             # Pointwise Convolution Layer
#             # filter_shape1 = [1, 1, 1, num_filters]
#             # W1 = tf.Variable(tf.random.truncated_normal(filter_shape1, stddev=0.1), name="W1")
#             # conv1 = tf.nn.conv2d(
#             #     conv,
#             #     W1,
#             #     strides=[1, 1, 1, 1],
#             #     padding="VALID",
#             #     name="conv1")
#
#             # Pointwise Convolution Layer
#             filter_shape1 = [1, 1, 1, num_filters]
#             W1 = torch.tensor(truncnorm.rvs(-1, 1, size=[10, 10]))
#             conv1 = torch.nn.functional.conv2d(conv, W1, bias=b, strides=[1, 1, 1, 1], padding=0)
#
#             # Batch Normalzation
#             # conv_bn2 = tf.layers.batch_normalization(conv1, momentum=0.9)
#             conv_bn2 = nn.BatchNorm2d(conv1, momentum=0.9)
#
#             # Apply nonlinearity
#             # h = tf.nn.leaky_relu(tf.nn.bias_add(conv_bn2, b), 0.1, name="leakyRelu")
#             h = nn.LeakyReLU(conv_bn2)
#             # h = nn.LeakyReLU(0.2, inplace=True)
#
#             # Maxpooling over the outputs
#             # pooled = tf.nn.max_pool2d(
#             #     h,
#             #     ksize=ksize_1,
#             #     strides=[1, 1, 1, 1],
#             #     padding='VALID',
#             #     name="pool")
#             # pooled_outputs.append(pooled)
#
#             pooled = torch.nn.functional.max_pool(h, kernal_size=ksize_1,
#                                                   strides=[1, 1, 1, 1], padding=0)
#             pooled_outputs.append(pooled)
#
#         # Combine all the pooled features
#         num_filters_total = num_filters * len(filter_sizes)
#         self.h_pool = torch.concat(pooled_outputs, 3)
#         self.h_pool_flat = torch.reshape(self.h_pool, [-1, num_filters_total])
#
#         # Add dropout
#         # with tf.name_scope("dropout"):
#         #      self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
#         self.dropout = nn.Dropout(self.h_pool_flat)
#
#
#     def forward(self, x):
#         # x = self.embed(x)  # (N, W, D)
#         x = self.h_pool_flat
#         x = self.dropout(x)  # (N, len(Ks)*Co)
#
#         x = x.unsqueeze(1)  # (N, Ci, W, D)
#
#         x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
#
#         x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
#
#         x = torch.cat(x, 1)
#
#         x = self.dropout(x)  # (N, len(Ks)*Co)
#         logit = self.fc1(x)  # (N, C)
#         return logit
#
#
#     def __init__(self, args):
#         super(LightCNN_Text, self).__init__()
#         self.args = args
#
#         V = args.embed_num
#         D = args.embed_dim
#         C = args.class_num
#         Ci = 1
#         Co = args.kernel_num
#         Ks = args.kernel_sizes
#
#         self.embed = nn.Embedding(V, D)
#         self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
#         self.dropout = nn.Dropout(args.dropout)
#         self.fc1 = nn.Linear(len(Ks) * Co, C)
#
#         if self.args.static:
#             self.embed.weight.requires_grad = False
#
#     def forward(self, x):
#         x = self.embed(x)  # (N, W, D)
#
#         x = x.unsqueeze(1)  # (N, Ci, W, D)
#
#         x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
#
#         x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
#
#         x = torch.cat(x, 1)
#
#         x = self.dropout(x)  # (N, len(Ks)*Co)
#         logit = self.fc1(x)  # (N, C)
#         return logit