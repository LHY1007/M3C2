import os
import torch
import torch.nn as nn
# import curve
import platform
import dataset
from torch.utils.data import DataLoader,Dataset
import numpy as np
import argparse, time, random
import yaml
from yaml.loader import SafeLoader
from evaluation import *
import glob
import model
import net
from saver import Saver
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import roc_curve, auc
from tensorboardX import SummaryWriter
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import gc
from sklearn import metrics
import h5py
from utils import *
import pandas as pd
from skimage import io

from apex import amp
"""
label 2016={ 0:'G2_O', 1:'G3_O', 2:'G2_A', 3:'G3_A', 4:'G2_OA', 5:'G3_OA', 6:'GBM'}
label 2021={ 0:'G2_O', 1:'G3_O', 2:'G2_A', 3:'G3_A', 4:'G4_A', 5:'GBM'}
"""


def train(opt):

    gpuID = opt['gpus']

    ############## Init #####################################


    device = torch.device('cuda:{}'.format(gpuID[0])) if gpuID else torch.device('cpu')


    Res_pretrain= net.Res50_pretrain().cuda(gpuID[0])
    # Res_pretrain.to(device)
    Res_pretrain = nn.DataParallel(Res_pretrain, device_ids=gpuID)
    Res_pretrain.eval()

    ###############  Datasets #######################



    trainDataset = dataset.Our_Dataset(phase='Train',opt=opt)
    valDataset = dataset.Our_Dataset(phase='Val', opt=opt)
    testDataset = dataset.Our_Dataset(phase='Test',opt=opt)
    trainLoader = DataLoader(trainDataset, batch_size=opt['batchSize'],
                             num_workers=opt['nThreads'], shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=opt['Val_batchSize'],
                             num_workers=opt['nThreads'], shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=opt['Test_batchSize'],
                            num_workers=opt['nThreads'], shuffle=True)


    train_bar = tqdm(trainLoader)
    for packs in train_bar:
        img = packs[0][0] #(N,3,224,224)
        imgPath = packs[2][0]
        patches_coor = packs[3][0] # list N,2
        if torch.cuda.is_available():
            img = img.cuda(gpuID[0])
        N=img.detach().cpu().numpy().shape[0]


        feature = Res_pretrain(img) # N 1024
        feature = torch.unsqueeze(feature, dim=0)
        feature_save = feature.detach().cpu().numpy()
        feature_save=np.float16(feature_save)
        print(feature_save.shape)
        if not os.path.exists(opt['dataDir'] +'Res50_feature_2000_fixdim0_512/'):
            os.makedirs(opt['dataDir'] +'Res50_feature_2000_fixdim0_512/')
        with h5py.File(opt['dataDir'] +'Res50_feature_2000_fixdim0_512/'+ imgPath+'.h5', 'w') as f:
            f['Res_feature'] = feature_save
            f['patches_coor'] = patches_coor

    train_bar = tqdm(valLoader)
    for packs in train_bar:
        img = packs[0][0]  # (N,3,224,224)
        imgPath = packs[2][0]
        patches_coor = packs[3][0]  # list N,2
        if torch.cuda.is_available():
            img = img.cuda(gpuID[0])
        N = img.detach().cpu().numpy().shape[0]

        feature = Res_pretrain(img)  # N 1024
        feature = torch.unsqueeze(feature, dim=0)
        feature_save = feature.detach().cpu().numpy()
        feature_save = np.float16(feature_save)
        print(feature_save.shape)
        if not os.path.exists(opt['dataDir'] + 'Res50_feature_2000_fixdim0_512/'):
            os.makedirs(opt['dataDir'] + 'Res50_feature_2000_fixdim0_512/')
        with h5py.File(opt['dataDir'] + 'Res50_feature_2000_fixdim0_512/' + imgPath + '.h5', 'w') as f:
            f['Res_feature'] = feature_save
            f['patches_coor'] = patches_coor
    #
    train_bar = tqdm(testLoader)
    for packs in train_bar:
        img = packs[0][0]  # (N,3,224,224)
        imgPath = packs[2][0]
        patches_coor = packs[3][0]  # list N,2
        if torch.cuda.is_available():
            img = img.cuda(gpuID[0])
        N = img.detach().cpu().numpy().shape[0]

        feature = Res_pretrain(img)  # N 1024
        feature = torch.unsqueeze(feature, dim=0)
        feature_save = feature.detach().cpu().numpy()
        feature_save = np.float16(feature_save)
        print(feature_save.shape)
        if not os.path.exists(opt['dataDir'] + 'Res50_feature_2000_fixdim0_512/'):
            os.makedirs(opt['dataDir'] + 'Res50_feature_2000_fixdim0_512/')
        with h5py.File(opt['dataDir'] + 'Res50_feature_2000_fixdim0_512/' + imgPath + '.h5', 'w') as f:
            f['Res_feature'] = feature_save
            f['patches_coor'] = patches_coor
        # features_WSI=[]
        # for i in range(N):
        #     feature=Res_pretrain(torch.unsqueeze(img[i],dim=0))
        #     feature=feature.detach().cpu().numpy()
        #     features_WSI.append(feature)
        # features_WSI=np.asarray(features_WSI) #(N,1024)
        # feature_save = np.expand_dims(features_WSI, axis=0)
        # feature_save = np.float16(feature_save)
        # print(feature_save.shape)
        # if not os.path.exists(opt['dataDir'] + 'Res50_feature_1200_iso_512/'):
        #     os.makedirs(opt['dataDir'] + 'Res50_feature_1200_iso_512/')
        # with h5py.File(opt['dataDir'] + 'Res50_feature_1200_iso_512/' + imgPath + '.h5', 'w') as f:
        #     f['Res_feature'] = feature_save
        #     f['patches_coor'] = patches_coor


        a=1









if __name__ == '__main__':





    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='config/miccai.yml')
    args = parser.parse_args()
    with open(args.opt) as f:
        opt = yaml.load(f, Loader=SafeLoader)
    #
    # k1 = h5py.File(opt['dataDir'] + 'Res50_feature_1/TCGA-DH-5143-01Z-00-DX1.h5')['Res50_feature'][:][0,100,:]
    # k2 = h5py.File( './TCGA-DH-5143-01Z-00-DX1.h5')['Res50_feature'][:][0,100,:]

    # k3 = h5py.File(opt['dataDir'] + 'Res50_feature_1/TCGA-06-0876-01Z-00-DX1.h5')['Res50_feature'][:][0,0,100,:]
    # k4 = h5py.File(opt['dataDir'] + 'Res50_feature_2/TCGA-06-0876-01Z-00-DX1.h5')['Res50_feature'][:][0,100,:]
    #
    # k5 = h5py.File(opt['dataDir'] + 'Res50_feature_1/TCGA-CS-6290-01A-01-TS1.h5')['Res50_feature'][:][0,0,100,:]
    # k6 = h5py.File(opt['dataDir'] + 'Res50_feature_2/TCGA-CS-6290-01A-01-TS1.h5')['Res50_feature'][:][0,100,:]


    a = 1
    train(opt)



    a=1





























