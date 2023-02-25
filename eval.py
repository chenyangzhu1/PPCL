import torch
import torch.nn.functional as F  # torch.tanh
import torch.nn as nn
from torch.autograd import Variable
import scipy.io as sio
from utils.metric import compress, calculate_top_map
import utils.datasets_mir as datasets_mir
from utils.models import ImgNet, TxtNet, Autoencoder, Autoencoder1, domain_classifier_I, domain_classifier_T
import time
import os
from utils.utils import *
import utils.ramps as ramps
import logging
from utils.models import *
import argparse
import numpy as np
from numpy import *
from utils import metric
from tqdm import tqdm
import os
device_ids="6"
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
parser = argparse.ArgumentParser(description="ADSH demo")
parser.add_argument('--bits', default='32', type=str,
                    help='binary code length (default: 8,12,16,24,32,48,64,96,128)')
parser.add_argument('--eps-list', default='0.6', type=str,
                    help='binary code length (default: 0.1,0.2)')
parser.add_argument('--gpu', default='3,4', type=str,
                    help='selected gpu (default: 1)')
parser.add_argument('--batch-size', default=256, type=int,
                    help='batch size (default: 64)')
parser.add_argument('--BETA', default=0.6, type=float,
                    help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--NUM-EPOCH', default=200, type=int,
                    help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--LR-IMG', default=0.001, type=float,
                    help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--LR-TXT', default=0.01, type=float,
                    help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--alpha', default=0.8, type=float,
                    help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--MOMENTUM', default=0.9, type=float,
                    help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--WEIGHT-DECAY', default=5e-4, type=float,
                    help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--ema-decay', default=0.999, type=float,
                    help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--consistency', default=100.0, type=float,
                    help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--consistency-rampup', default=30, type=int,
                    help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--NUM-WORKERS', default=8, type=int,
                    help='number of epochs (default: 3)')
parser.add_argument('--LAMBDA1', default=0.1, type=float,
                    help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--LAMBDA2', default=0.1, type=float,
                    help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--EVAL', default=True, type=bool,
                    help='selected gpu (default: 1)')
parser.add_argument('--EPOCH-INTERVAL', default=2, type=int,
                    help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--EVAL-INTERVAL', default=20, type=int,
                    help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('--MU', default=1.5, type=float,
                    help='hyper-parameter: learning rate (default: 10**-3)')
parser.add_argument('-max-norm', type=float, default=3.0,
                    help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128,
                    help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=128,
                    help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5',
                    help='comma-separated kernel size to use for convolution')
parser.add_argument('-vocab-size', type=int, default=1387,
                    help='number of each kind of kernel')


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)  # 合并在一起

    total0 = total.unsqueeze(0).expand(int(total.size(0)),
                                       int(total.size(0)),
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)),
                                       int(total.size(0)),
                                       int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)  # 计算高斯核中的|x-y|

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]


    test_ker = torch.exp(-L2_distance / bandwidth_list[0])
    for i in range(1, kernel_num):
        test_ker =test_ker+ torch.exp(-L2_distance / bandwidth_list[i])
    return test_ker  # 将多个核合并在一起


class Session:
    def __init__(self):

        self.train_dataset = datasets_mir.MIRFlickr(
            train=True, transform=datasets_mir.mir_train_transform)
        self.test_dataset = datasets_mir.MIRFlickr(
            train=False, database=False, transform=datasets_mir.mir_test_transform)
        self.database_dataset = datasets_mir.MIRFlickr(train=False, database=True,
                                                       transform=datasets_mir.mir_test_transform)
        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=args.batch_size,
                                                        pin_memory=True,
                                                        shuffle=True,
                                                        num_workers=args.NUM_WORKERS,
                                                        drop_last=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                       batch_size=args.batch_size,
                                                       pin_memory=True,
                                                       shuffle=False,
                                                       num_workers=args.NUM_WORKERS)

        self.database_loader = torch.utils.data.DataLoader(dataset=self.database_dataset,
                                                           batch_size=args.batch_size,
                                                           pin_memory=True,
                                                           shuffle=False,
                                                           num_workers=args.NUM_WORKERS)

        self.best_it = 0
        self.best_ti = 0

    def define_model(self, coed_length):

        self.CodeNet_N = Autoencoder1()

        self.CodeNet_I = JDSHImgNet(code_len=coed_length)


        self.FeatNet_I = ImgNet(code_len=coed_length)

        self.txt_feat_len = datasets_mir.txt_feat_len
        self.CodeNet_T = JDSHTxtNet(code_len=coed_length, txt_feat_len=self.txt_feat_len)

        self.Clf_I = domain_classifier_I()
        self.Clf_T = domain_classifier_T(txt_feat_len=self.txt_feat_len)
        """
        PPCL
        """
        self.imgsubnet = Img_subnet(code_len=500)
        self.comnet = Combine_net(code_len=coed_length)

        self.opt_I = torch.optim.SGD(self.imgsubnet.parameters(
        ), lr=args.LR_IMG, momentum=args.MOMENTUM, weight_decay=args.WEIGHT_DECAY)
        self.opt_T = torch.optim.SGD(self.CodeNet_T.parameters(
        ), lr=args.LR_TXT, momentum=args.MOMENTUM, weight_decay=args.WEIGHT_DECAY)
        self.opt_com = torch.optim.SGD(self.comnet.parameters(
        ), lr=args.LR_IMG, momentum=args.MOMENTUM, weight_decay=args.WEIGHT_DECAY)
        self.opt_N = torch.optim.Adam(
            self.CodeNet_N .parameters(), lr=args.LR_IMG)

    def load_checkpoints(self):
        path = os.path.join("/data/zcy/PIP/mir/models/PPCL_ckp.pkl")
        #PIP训练出来的生成器
        self.CodeNet_N.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage.cuda()))
        ckp_path = os.path.join("/data/zcy/PIP/mir/models", 'latest.pth')
        obj = torch.load(ckp_path, map_location=lambda storage, loc: storage.cuda())
        #目标模型的图像网络和文本网络
        self.CodeNet_I.load_state_dict(obj['ImgNet'])
        self.CodeNet_T.load_state_dict(obj['TxtNet'])


    def evalJDSH(self, eps, bit):
        logger.info('--------------------Evaluation: Calculate top MAP-------------------')

        # Change model to 'eval' mode (BN uses moving mean/var).
        self.CodeNet_I.eval().cuda()
        self.CodeNet_T.eval().cuda()
        self.Clf_I.eval().cuda()
        self.Clf_T.eval().cuda()
        self.CodeNet_N.eval().cuda()


        path = os.path.join(logdir, str(args.bits) + 'bits-Inoise-record.pkl')

        re_BI, re_BT, re_L, re_BNI, qu_BI, qu_BT, qu_L, qu_BN, correct1, correct2, correct3, correct_1, correct_2, correct_3 = metric.compress(
            self.database_loader,
            self.test_loader, self.CodeNet_I, self.CodeNet_T, self.CodeNet_N, self.Clf_I, self.Clf_T,
            self.database_dataset, self.test_dataset, eps)

        sio.savemat('./result/pip/' + str(eps) + '/' + str(bit) + '/'
                    + 'code.mat', {
                        're_BNI': re_BNI,
                        're_BI': re_BI,
                        're_BT': re_BT,
                        're_L': re_L,
                        'qu_BI': qu_BI,
                        'qu_BT': qu_BT,
                        'qu_BN': qu_BN,
                        'qu_L': qu_L})

        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_I2I = calculate_top_map(qu_B=qu_BI, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_I2NI = calculate_top_map(qu_B=qu_BI, re_B=re_BNI, qu_L=qu_L, re_L=re_L, topk=50)

        MAP_NI2T = calculate_top_map(qu_B=qu_BN, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_NI2I = calculate_top_map(qu_B=qu_BN, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I1 = calculate_top_map(qu_B=qu_BN, re_B=re_BNI, qu_L=qu_L, re_L=re_L, topk=50)

        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2NI = calculate_top_map(qu_B=qu_BT, re_B=re_BNI, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2T = calculate_top_map(qu_B=qu_BT, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)

        logger.info("JDSH_Eval")
        logger.info('MAP of Image to Text: %.3f, MAP of Image to Image: %.3f, MAP of Image to Noise Image: %.3f' % (
            MAP_I2T, MAP_I2I, MAP_I2NI))
        logger.info(
            'MAP of Noise Image to Text: %.3f, MAP of Noise Image to Image: %.3f, MAP of Noise Image to Noise Image: %.3f' % (
                MAP_NI2T, MAP_NI2I, MAP_T2I1))
        logger.info('MAP of Text to Image: %.3f, MAP of Text to Noise Image: %.3f, MAP of Text to Text: %.3f' % (
            MAP_T2I, MAP_T2NI, MAP_T2T))
        logger.info('label accuracy of database clean Image: %.3f, clean Text: %.3f, Noise Image: %.3f' % (
            correct1, correct2, correct3))
        logger.info('label accuracy of query clean Image: %.3f, clean Text: %.3f, Noise Image: %.3f' % (
            correct_1, correct_2, correct_3))
        logger.info('--------------------------------------------------------------------')
    

def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def gaussian_kernel_matrix(dist, S1, S2):
    k1 = 1 / (torch.sum(S1, 1) + 0.01)
    k2 = 1 / (torch.sum(S2, 1) + 0.01)

    dist1 = dist * S1
    dist2 = dist * S2

    dist1 = torch.sum(dist1, 0)
    dist2 = torch.sum(dist2, 0)

    return dist1 * k1, dist2 * k2


def pairwise_distance(x, y):
    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.norm(x - y, p=2, dim=1)
    output = torch.transpose(output, 0, 1)

    return output


def mkdir_multi(path):
    # 判断路径是否存在
    isExists = os.path.exists(path)

    if not isExists:
        # 如果不存在，则创建目录（多层）
        os.makedirs(path)
        print('successfully creat path！')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print('path already exists！')
        return False


def _logging():
    global logger
    # logfile = os.path.join(logdir, 'log.log')
    logfile = os.path.join(logdir, 'log.log')
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    _format = logging.Formatter("%(name)-4s: %(levelname)-4s: %(message)s")
    fh.setFormatter(_format)
    ch.setFormatter(_format)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return


def main():
    print(torch.cuda.is_available())
    global logdir, args

    args = parser.parse_args()

    sess = Session()

    bits = [int(bit) for bit in args.bits.split(',')]
    epss = [float(eps) for eps in args.eps_list.split(',')]
    for bit in bits:
        for eps in epss:
            logdir = './result/ppcl/' + str(eps) + '/' + str(bit) + '/'
            mkdir_multi(logdir)
            _logging()

            if args.EVAL == True:
                sess.define_model(bit)

                sess.load_checkpoints()
                sess.evalJDSH(eps, bit)
            else:
                logger.info(
                    '--------------------------Construct Models--------------------------')
                sess.define_model(bit)

                logger.info(
                    '--------------------------Adversarial Training--------------------------')
                for epoch in tqdm(range(args.NUM_EPOCH)):
                    # train the Model
                    iter_time1 = time.time()
                    sess.train(epoch, eps)
                    iter_time1 = time.time() - iter_time1
                    logger.info('[pre_train time: %.4f]', iter_time1)
                    if (epoch + 1) % args.EVAL_INTERVAL == 0:
                        iter_time1_1 = time.time()
                        sess.eval(epoch, eps, bit)
                        iter_time1_1 = time.time() - iter_time1_1
                        logger.info(
                            '[pre_train eval2 time: %.4f]', iter_time1_1)


if __name__ == "__main__":
    main()
