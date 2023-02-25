import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision.datasets as dsets
from torchvision import transforms
from torch.autograd import Variable
import torchvision
# import math
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision.datasets as dsets
from torchvision import transforms
from torch.autograd import Variable
import torchvision
import math
import numpy as np
# import settings
import os

def compress(train_loader, test_loader, model_I, model_T, modeln, Clf_I, Clf_T, train_dataset, test_dataset, eps_1):
    re_BI = list([])
    re_BNI = list([])
    re_BT = list([])
    re = list([])
    re_L = list([])
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct_1 = 0
    correct_2 = 0
    correct_3 = 0

    for _, (data_I, data_T, _, _) in enumerate(train_loader):

        batch_size_ = data_I.size(0)
        # define domain label
        domain_label_1 = torch.zeros(batch_size_)
        domain_label_1 = domain_label_1.long().cuda()
        domain_label_1 = Variable(domain_label_1)

        domain_label_2 = torch.ones(batch_size_)
        domain_label_2 = domain_label_2.long().cuda()
        domain_label_2 = Variable(domain_label_2)

        #clean image database
        var_data_I = Variable(data_I.cuda())
        _, hid_I, code_I = model_I(var_data_I)
        _, R_I = Clf_I(var_data_I)
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())
        correct_1 += R_I.eq(domain_label_2.view_as(R_I)).sum().item()

        #noise image database
        _, noise = modeln(var_data_I,1, eps_1)
        atkdata = torch.clamp(var_data_I + noise * eps_1, 0, 1)
        _, hid_NI, codeN_I = model_I(atkdata)
        _, RN_I = Clf_I(atkdata)
        code_NI = torch.sign(codeN_I)
        re_BNI.extend(code_NI.cpu().data.numpy())
        correct_2 += RN_I.eq(domain_label_1.view_as(RN_I)).sum().item()

        #clean text database
        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        _, hid_T, code_T = model_T(var_data_T)
        _, R_T = Clf_T(var_data_T)

        code_T = torch.sign(code_T)
        re_BT.extend(code_T.cpu().data.numpy())
        correct_3 += R_T.eq(domain_label_2.view_as(R_T)).sum().item()

    qu_BI = list([])
    qu_BT = list([])
    qu_BN = list([])
    for _, (data_I, data_T, _, _) in enumerate(test_loader):

        batch_size_ = data_I.size(0)
        # define domain label
        domain_label_1 = torch.zeros(batch_size_)
        domain_label_1 = domain_label_1.long().cuda()
        domain_label_1 = Variable(domain_label_1)

        domain_label_2 = torch.ones(batch_size_)
        domain_label_2 = domain_label_2.long().cuda()
        domain_label_2 = Variable(domain_label_2)


        #clean image query
        var_data_I = Variable(data_I.cuda())
        _, hid_I, code_I = model_I(var_data_I)
        _, C_I = Clf_I(var_data_I)

        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())
        correct1 += C_I.eq(domain_label_2.view_as(C_I)).sum().item()

        #clean text query
        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        _, hid_T, code_T = model_T(var_data_T)
        _, C_T = Clf_T(var_data_T)

        code_T = torch.sign(code_T)
        qu_BT.extend(code_T.cpu().data.numpy())
        correct2 += C_T.eq(domain_label_2.view_as(C_T)).sum().item()

        #noise image query
        _, noise = modeln(var_data_I,1, eps_1)
        atkdata = torch.clamp(var_data_I + noise * eps_1, 0, 1)
        _, hid_NI, code_NI = model_I(atkdata)
        _, N_I = Clf_I(var_data_I)

        code_NI = torch.sign(code_NI)
        qu_BN.extend(code_NI.cpu().data.numpy())
        correct3 += N_I.eq(domain_label_1.view_as(N_I)).sum().item()

    re_BI = np.array(re_BI)
    re_BNI = np.array(re_BNI)
    re_BT = np.array(re_BT)
    re_L = train_dataset.train_labels

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_BN = np.array(qu_BN)
    qu_L = test_dataset.train_labels

    correct1 = 100. * correct1 / len(test_dataset)
    correct2 = 100. * correct2 / len(test_dataset)
    correct3 = 100. * correct3 / len(test_dataset)
    correct_1 = 100. * correct_1 / len(train_dataset)
    correct_2 = 100. * correct_2 / len(train_dataset)
    correct_3 = 100. * correct_3 / len(train_dataset)

    return re_BI, re_BT, re_L, re_BNI, qu_BI, qu_BT, qu_L, qu_BN, correct1, correct2, correct3, correct_1, correct_2, correct_3


def calculate_hamming(B1, B2):
    """
    :param B1:  vector [n]
    :param B2:  vector [r*n]
    :return: hamming distance [r]
    """
    leng = B2.shape[1]  # max inner product value
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH


def calculate_map(qu_B, re_B, qu_L, re_L):
    """
       :param qu_B: {-1,+1}^{mxq} query bits
       :param re_B: {-1,+1}^{nxq} retrieval bits
       :param qu_L: {0,1}^{mxl} query label
       :param re_L: {0,1}^{nxl} retrieval label
       :return:
    """
    num_query = qu_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, tsum)  # [1,2, tsum]
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map


def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):
    """
    :param qu_B: {-1,+1}^{mxq} query bits
    :param re_B: {-1,+1}^{nxq} retrieval bits
    :param qu_L: {0,1}^{mxl} query label
    :param re_L: {0,1}^{nxl} retrieval label
    :param topk:
    :return:
    """
    num_query = qu_L.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum.astype(int))  # 在指定的间隔内返回均匀间隔的数字
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0  # 等于1时+1
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

class DCG(object):

    def __init__(self, k=10, gain_type='exp2'):
        """
        :param k: int DCG@k
        :param gain_type: 'exp2' or 'identity'
        """
        self.k = k
        self.discount = self._make_discount(256)
        if gain_type in ['exp2', 'identity']:
            self.gain_type = gain_type
        else:
            raise ValueError('gain type not equal to exp2 or identity')

    def evaluate(self, targets):
        """
        :param targets: ranked list with relevance
        :return: float
        """
        gain = self._get_gain(targets)
        discount = self._get_discount(min(self.k, len(gain)))
        return np.sum(np.divide(gain, discount))

    def _get_target(self, qu_B, re_B, qu_L, re_L, topk):

        num_query = qu_L.shape[0]
        topkmap = 0

        for iter in range(num_query):
            gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
            # gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
            tsum = np.sum(gnd)
            if tsum == 0:
                continue
            hamm = calculate_hamming(qu_B[iter, :], re_B)
            ind = np.argsort(hamm)
            gnd = gnd[ind]

            gain = self._get_gain(gnd)
            discount = self._get_discount(min(self.k, len(gain)))
            topkmap_ = np.sum(np.divide(gain, discount))
            topkmap = topkmap + topkmap_
        return topkmap

    def _get_gain(self, targets):
        t = targets[:self.k]
        if self.gain_type == 'exp2':
            return np.power(2.0, t) - 1.0
        else:
            return t

    def _get_discount(self, k):
        if k > len(self.discount):
            self.discount = self._make_discount(2 * len(self.discount))
        return self.discount[:k]

    @staticmethod
    def _make_discount(n):
        x = np.arange(1, n + 1, 1)
        discount = np.log2(x + 1)
        return discount


class NDCG(DCG):

    def __init__(self, k=10, gain_type='exp2'):
        """
        :param k: int NDCG@k
        :param gain_type: 'exp2' or 'identity'
        """
        super(NDCG, self).__init__(k, gain_type)

    def _get_target(self, qu_B, re_B, qu_L, re_L, topk):

        num_query = qu_L.shape[0]
        ndcg = 0

        for iter in range(num_query):
            gnd = (np.dot(qu_L[iter, :], re_L.transpose())).astype(np.float32)
            # gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
            tsum = np.sum(gnd)
            if tsum == 0:
                continue
            hamm = calculate_hamming(qu_B[iter, :], re_B)
            ind = np.argsort(hamm)
            gnd = gnd[ind]

            dcg = super(NDCG, self).evaluate(gnd)
            ideal = np.sort(gnd)[::-1]
            idcg = super(NDCG, self).evaluate(ideal)
            ndcg_ = dcg / idcg
            # gain = self._get_gain(gnd)
            # discount = self._get_discount(min(self.k, len(gain)))
            # topkmap_ = np.sum(np.divide(gain, discount))
            ndcg = ndcg + ndcg_
        ndcg = ndcg / num_query

        return ndcg

    def evaluate(self, targets):
        """
        :param targets: ranked list with relevance
        :return: float
        """
        dcg = super(NDCG, self).evaluate(targets)
        ideal = np.sort(targets)[::-1]
        idcg = super(NDCG, self).evaluate(ideal)
        return dcg / idcg

    def maxDCG(self, targets):
        """
        :param targets: ranked list with relevance
        :return:
        """
        ideal = np.sort(targets)[::-1]
        return super(NDCG, self).evaluate(ideal)


if __name__ == "__main__":
    targets = [3, 2, 3, 0, 1, 2, 3, 2]
    dcg6 = DCG(6, 'identity')
    ndcg6 = NDCG(6, 'identity')
    assert 6.861 < dcg6.evaluate(targets) < 6.862
    assert 0.785 < ndcg6.evaluate(targets) < 0.786
    ndcg10 = NDCG(10)
    assert 0 < ndcg10.evaluate(targets) < 1.0
    assert 0 < ndcg10.evaluate([1, 2, 3]) < 1.0