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
# os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
device_ids="7"
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
parser.add_argument('--EVAL', default=False, type=bool,
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

    # 计算多核中每个核的bandwidth
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

        self.FeatNet_I = ImgNet(code_len=coed_length)

        self.txt_feat_len = datasets_mir.txt_feat_len
        self.CodeNet_T = TxtNet(code_len=coed_length,
                                txt_feat_len=self.txt_feat_len)
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


    def train(self, epoch, eps):
        self.CodeNet_N.cuda().train()
        # self.CodeNet_I.cuda().train()
        self.FeatNet_I.cuda().eval()
        self.CodeNet_T.cuda().train()
        self.imgsubnet.cuda().train()
        self.comnet.cuda().train()


        # self.CodeNet_I.set_alpha(epoch)
        self.CodeNet_T.set_alpha(epoch)
        i = 0
        len_dataloader = len(self.train_loader)
        for idx, (img, F_T, labels, _) in enumerate(self.train_loader):

            i = i + 1

            labels=Variable(labels.cuda())
            S=labels.mm(labels.t())
            S=torch.where(S>0,1,S)
            S=torch.where(S<=0,-1,S)
            # S.cuda()
            img = Variable(img.cuda())
            F_T = Variable(torch.FloatTensor(F_T.numpy()).cuda())

            batch_size_ = img.size(0)

            # ---- noise image ----------------
            x, atkdata = self.CodeNet_N(img, 1, eps)

            F_I, _, _ = self.FeatNet_I(img)
            _, clean_y, code_T = self.CodeNet_T(F_T)

            """
            PPCL
            """


            _, clean_x,_ = self.imgsubnet(img)
            Opair = torch.cat((clean_x, clean_y), dim=1)
            mmfea_O = self.comnet(Opair)
            _, noise_x,_ = self.imgsubnet(atkdata)
            Apair = torch.cat((noise_x, clean_y), dim=1)
            mmfea_A = self.comnet(Apair)

            mmfea_O = F.normalize(mmfea_O)
            mmfea_A = F.normalize(mmfea_A)

            JM = F.mse_loss(mmfea_A, mmfea_O)  # 0.0359

            TAO = mmfea_O.mm(mmfea_A.t())/2  # [32,32]

            JS_temp = torch.log(1+torch.exp(TAO))-torch.mul(S, TAO)

            JS = torch.sum(JS_temp, dim=0)/JS_temp.size(0)
            JS = torch.sum(JS, dim=0)/JS.size(0)  # 0.7448

            ATAO = mmfea_A.mm(mmfea_A.t())/2
            JO_temp = torch.log(1+torch.exp(ATAO))-torch.mul(S, ATAO)

            JO = torch.sum(JO_temp, dim=0)/JO_temp.size(0)
            JO = torch.sum(JO, dim=0)/JO.size(0)  # 0.7712

            n = int(mmfea_O.size()[0])
            m = int(mmfea_A.size()[0])
            kernels = guassian_kernel(mmfea_O, mmfea_A,
                                      kernel_mul=2.0, kernel_num=5, fix_sigma=None)
            KOO = kernels[:n, :n]  # [256,256]
            KAA = kernels[n:, n:]
            KOA = kernels[:n, n:]
            KAO = kernels[n:, :n]

            S_postive=torch.zeros(S.size()).cuda()
            S_negative=torch.zeros(S.size()).cuda()
            one=torch.ones_like(S).cuda()
            S_postive=torch.where(S>0,one,S_postive)
            S_negative=torch.where(S<=0,one,S_negative)

            SpOO = S_postive.mul(KOO)
            SpOO = torch.sum(SpOO, dim=0)
            SpOO = torch.sum(SpOO, dim=0)

            SpAA = S_postive.mul(KAA)
            SpAA = torch.sum(SpAA, dim=0)
            SpAA = torch.sum(SpAA, dim=0)

            SpOA = S_postive.mul(KOA)
            SpOA = torch.sum(SpOA, dim=0)
            SpOA = torch.sum(SpOA, dim=0)

            Spsum = torch.sum(S_postive, dim=0)
            Spsum = torch.sum(Spsum, dim=0)

            Dps = (SpOO+SpAA-2*SpOA)/Spsum

            SnOO = S_negative.mul(KOO)
            SnOO = torch.sum(SnOO, dim=0)
            SnOO = torch.sum(SnOO, dim=0)

            SnAA = S_negative.mul(KAA)
            SnAA = torch.sum(SnAA, dim=0)
            SnAA = torch.sum(SnAA, dim=0)

            SnOA = S_negative.mul(KOA)
            SnOA = torch.sum(SnOA, dim=0)
            SnOA = torch.sum(SnOA, dim=0)

            Snsum = torch.sum(S_negative, dim=0)
            Snsum = torch.sum(Snsum, dim=0)

            Dns = (SpOO+SpAA-2*SpOA)/Snsum

            JD = Dps-Dns  # 0.6582
            alpha = 1e-2
            beta = 1e-5
            delta = 1e-5
            JP = JM+alpha*JS+beta*JD
            loss = JP+delta*JO
            self.opt_I.zero_grad()
            self.opt_T.zero_grad()
            self.opt_N.zero_grad()
            self.opt_com.zero_grad()

            loss.backward()

            self.opt_I.step()
            self.opt_T.step()
            self.opt_N.step()
            self.opt_com.step()



            if (idx + 1) % (len(self.train_dataset) // args.batch_size / args.EPOCH_INTERVAL) == 0:
                logger.info('Epoch [%d/%d], Iter [%d/%d] Loss1: %.4f'
                            % (epoch + 1, args.NUM_EPOCH, idx + 1, len(self.train_dataset) // args.batch_size,
                                loss.item()))


    def eval(self, epoch, eps, bit):
        logger.info(
            '--------------------Evaluation: Calculate top MAP-------------------')

        # Change model to 'eval' mode (BN uses moving mean/var).
        self.CodeNet_T.eval().cuda()
        self.CodeNet_N.eval().cuda()
        self.imgsubnet.eval().cuda()
        self.comnet.eval().cuda()
        self.Clf_I.eval().cuda()
        self.Clf_T.eval().cuda()
        path = os.path.join(logdir, str(args.bits) + 'bits-Inoise-record.pkl')
        torch.save(self.CodeNet_N.state_dict(), path)

  


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
                sess.load_checkpoints()
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
