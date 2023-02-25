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


def compress_wiki(train_loader, test_loader, modeli, modelt, train_dataset, test_dataset, classes=10):
    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    for _, (data_I, _, data_T, target, _) in enumerate(train_loader):
        var_data_I = Variable(data_I.cuda())
        _, _, code_I = modeli(var_data_I)
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())
        re_L.extend(target)

        # data_T = data_T + 1
        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())

        _, _, code_T = modelt(var_data_T)
        code_T = torch.sign(code_T)
        re_BT.extend(code_T.cpu().data.numpy())

    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])
    for _, (data_I, _, data_T, target, _) in enumerate(test_loader):
        var_data_I = Variable(data_I.cuda())
        _, _, code_I = modeli(var_data_I)
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())
        qu_L.extend(target)

        # data_T = data_T + 1
        var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        _, _, code_T = modelt(var_data_T)
        code_T = torch.sign(code_T)
        qu_BT.extend(code_T.cpu().data.numpy())

    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)
    re_L = np.eye(classes)[np.array(re_L)]

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_L = np.eye(classes)[np.array(qu_L)]

    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L


def compress(train_loader, test_loader, model_I, model_T, train_dataset, test_dataset, args):
    re_BI = list([])
    re_BT = list([])
    re_L = list([])
    for _, (data_I, _, data_T, labels, _) in enumerate(train_loader):
        var_data_I = Variable(data_I.cuda())
        with torch.no_grad():
            _, _, code_I = model_I(var_data_I)
        code_I = torch.sign(code_I)
        re_BI.extend(code_I.cpu().data.numpy())

        # var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        # data_T = data_T + 1
        var_data_T = Variable(torch.LongTensor(data_T.numpy()).cuda())
        # noise_data1 = np.random.uniform(0, 1, args.embed_dim)
        # te_noise1 = noise_data1[np.newaxis, :]
        # te_noise_te1 = np.tile(te_noise1, (args.batch_size, 1))
        # te_noise_tensor1 = torch.from_numpy(te_noise_te1).type(torch.FloatTensor)
        # te_noise_tensor1 = Variable(te_noise_tensor1.cuda())

        batch_size_ = labels.size(0)
        noise_data1 = np.random.uniform(0, 1, 50 * args.embed_dim)
        noise_data1 = np.reshape(noise_data1, (50, args.embed_dim))
        te_noise1 = noise_data1[np.newaxis, :, :]
        te_noise_te1 = np.tile(te_noise1, (batch_size_, 1, 1))
        te_noise_tensor1 = torch.from_numpy(te_noise_te1).type(torch.FloatTensor)
        te_noise_tensor1 = Variable(te_noise_tensor1.cuda())

        with torch.no_grad():
            _, _, code_T = model_T(var_data_T, te_noise_tensor1)
        code_T = torch.sign(code_T)
        re_BT.extend(code_T.cpu().data.numpy())

    qu_BI = list([])
    qu_BT = list([])
    qu_L = list([])
    for _, (data_I, _, data_T, labels, _) in enumerate(test_loader):
        var_data_I = Variable(data_I.cuda())

        with torch.no_grad():
            _, _, code_I = model_I(var_data_I)
        code_I = torch.sign(code_I)
        qu_BI.extend(code_I.cpu().data.numpy())

        # var_data_T = Variable(torch.FloatTensor(data_T.numpy()).cuda())
        # data_T = data_T + 1
        var_data_T = Variable(torch.LongTensor(data_T.numpy()).cuda())
        # noise_data1 = np.random.uniform(0, 1, args.embed_dim)
        # te_noise1 = noise_data1[np.newaxis, :]
        # te_noise_te1 = np.tile(te_noise1, (args.batch_size, 1))
        # te_noise_tensor1 = torch.from_numpy(te_noise_te1).type(torch.FloatTensor)
        # te_noise_tensor1 = Variable(te_noise_tensor1.cuda())

        batch_size_ = labels.size(0)
        noise_data1 = np.random.uniform(0, 1, 50 * args.embed_dim)
        noise_data1 = np.reshape(noise_data1, (50, args.embed_dim))
        te_noise1 = noise_data1[np.newaxis, :, :]
        te_noise_te1 = np.tile(te_noise1, (batch_size_, 1, 1))
        te_noise_tensor1 = torch.from_numpy(te_noise_te1).type(torch.FloatTensor)
        te_noise_tensor1 = Variable(te_noise_tensor1.cuda())

        with torch.no_grad():
            _, _, code_T = model_T(var_data_T, te_noise_tensor1)
        code_T = torch.sign(code_T)
        qu_BT.extend(code_T.cpu().data.numpy())

    re_BI = np.array(re_BI)
    re_BT = np.array(re_BT)
    re_L = train_dataset.train_labels

    qu_BI = np.array(qu_BI)
    qu_BT = np.array(qu_BT)
    qu_L = test_dataset.train_labels
    return re_BI, re_BT, re_L, qu_BI, qu_BT, qu_L


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
        # gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.int)
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
        # gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.int)
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap


def euclidean_dist(x, y):
    """
    Args:
    x: pytorch Variable, with shape [m, d]
    y: pytorch Variable, with shape [n, d]
    Returns:
    dist: pytorch Variable, with shape [m, n]
    """

    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist.addmm_(1, -2, x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def euclidean_dist1(test_matrix, train_matrix):
    num_test = test_matrix.shape[0]
    num_train = train_matrix.shape[0]
    dists = np.zeros((num_test, num_train))
    # because(X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train, so
    d1 = -2 * np.dot(test_matrix, train_matrix.T)  # shape (num_test, num_train)
    d2 = np.sum(np.square(test_matrix), axis=1, keepdims=True)  # shape (num_test, 1)
    d3 = np.sum(np.square(train_matrix), axis=1)  # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


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