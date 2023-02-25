import numpy as np
import scipy.io as scio

TXT_DIR = "/home/pengfei/code/Datasets/NUS-WIDE-TC10/nus-wide-tc10-yall.mat"

txt_set = scio.loadmat(TXT_DIR)
txts = txt_set['YAll']

txt_embeddings=[]
len_max = 0
for i in txts:
    txt_embeddings.append(np.where(i==1)[0])


lengs = []
for i in txt_embeddings:
    lengs.append(len(i))
lengs = sorted(lengs)

# set the embedding length == 50

# for padding

tep=[]
for i in txt_embeddings:
    lp = 50 - len(i)
    if lp < 0:
        ten = i[:lp]
    else:
        ten = np.pad(i, (0, lp), 'constant', constant_values=(0, 1386))
    tep.append(ten)