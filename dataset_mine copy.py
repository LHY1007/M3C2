from __future__ import print_function

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader,Dataset
import numpy as np
import os
from PIL import Image
from skimage import io,transform
import cv2
import torch
import platform
import pandas as pd
import argparse, time, random
import yaml
from yaml.loader import SafeLoader
from tqdm import tqdm
import h5py
import gc
import math
import scipy.interpolate
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from torchvision.transforms import Compose
import transform.transforms_group as our_transform

class Our_Dataset(Dataset):
    def __init__(self, phase,opt,if_end2end=False):
        super(Our_Dataset, self).__init__()
        self.opt = opt
        self.patc_bs=64
        self.phase=phase
        self.if_end2end=if_end2end

        CPTAC_label = pd.read_excel(opt['CPTAC_label_path'], header=0)
        IvYGAP_label = pd.read_excel(opt['IvYGAP_label_path'], sheet_name='Sheet1', header=0)
        TCGA_label = pd.read_excel(opt['TCGA_label_path'], sheet_name='wsi_level', header=0)
        combined_labels = pd.concat([TCGA_label, CPTAC_label], ignore_index=True)
        excel_wsi = combined_labels.values

        PATIENT_LIST=excel_wsi[:,0]
        np.random.seed(self.opt['seed'])
        random.seed(self.opt['seed'])
        PATIENT_LIST=list(PATIENT_LIST)
        # IvYGAP_label
        IvYGAP_label = IvYGAP_label.values

        PATIENT_LIST=np.unique(PATIENT_LIST)
        np.random.shuffle(PATIENT_LIST)
        NUM_PATIENT_ALL=len(PATIENT_LIST) # 952
        TRAIN_PATIENT_LIST=PATIENT_LIST[0:int(NUM_PATIENT_ALL * 0.8)]
        # VAL_PATIENT_LIST = PATIENT_LIST[int(NUM_PATIENT_ALL * 0.9):]
        TEST_PATIENT_LIST = PATIENT_LIST[int(NUM_PATIENT_ALL * 0.80):int(NUM_PATIENT_ALL * 0.90)]
        self.TRAIN_LIST=[]
        self.VAL_LIST = []
        self.TEST_LIST = []
        self.I_TEST_LIST = []

        for i in range(excel_wsi.shape[0]):# 2612
            if excel_wsi[:,0][i] in TRAIN_PATIENT_LIST:
                self.TRAIN_LIST.append(excel_wsi[i,:])
            # elif excel_wsi[:,0][i] in VAL_PATIENT_LIST:
            #     self.VAL_LIST.append(excel_wsi[i,:])
            elif excel_wsi[:,0][i] in TEST_PATIENT_LIST:
                self.TEST_LIST.append(excel_wsi[i,:])

        for i in range(IvYGAP_label.shape[0]):# 2612
            self.I_TEST_LIST.append(IvYGAP_label[i,:])
        self.LIST= np.asarray(self.TRAIN_LIST) if self.phase == 'Train' else np.asarray(self.VAL_LIST) if self.phase == 'Val' else np.asarray(self.TEST_LIST) if self.phase == 'Test' else np.asarray(self.I_TEST_LIST)


        self.train_iter_count=0
        self.Flat=0
        self.WSI_all=[]

    def __getitem__(self, index):
        feature_all_20,feature_all_10, = self.read_feature(index)

        label=self.label_gene(index)

        return torch.from_numpy(np.array(feature_all_20)).float(),torch.from_numpy(np.array(feature_all_10)).float(),\
            torch.from_numpy(label)

    def read_feature(self, index):

        root = '/Res50_feature_2500_fixdim0_norm'

        patient_id = self.LIST[index, 0]


        if patient_id[0].startswith('T'):
            base_path = self.opt['dataDir'] + 'TCGA'
        elif patient_id[0].startswith('W'):
            base_path = self.opt['dataDir'] + 'IvYGAP'
        elif patient_id[0].startswith('C'):
            base_path = self.opt['dataDir'] + 'CPTAC'
        else:
            raise ValueError("Unknown data source")

        patch_20 = h5py.File(base_path + root + '_20x/' + self.LIST[index, 1] + '.h5')['Res_feature'][:]
        patch_10 = h5py.File(base_path + root + '/' + self.LIST[index, 1] + '.h5')['Res_feature'][:]
        return patch_20[0], patch_10[0]#, patch_1_25[0]


    def label_gene(self,index):


        if self.LIST[index, 4]=='WT':
            label_IDH=0
        elif self.LIST[index, 4]=='Mutant':
            label_IDH=1
        if self.LIST[index, 5] == 'non-codel':
            label_1p19q = 0
        elif self.LIST[index, 5] == 'codel':
            label_1p19q = 1
        if self.LIST[index, 6] == -2 or self.LIST[index, 6] == -1:
            label_CDKN = 1
        else:
            label_CDKN = 0

        if self.LIST[index, 2]=='oligoastrocytoma':
            label_His = 0
        elif self.LIST[index, 2] == 'astrocytoma':
            label_His = 1
        elif self.LIST[index, 2] == 'oligodendroglioma':
            label_His = 2
        elif self.LIST[index, 2] == 'glioblastoma':
            label_His = 3

        if self.LIST[index, 2]=='glioblastoma':
            label_His_2class = 1
        else:
            label_His_2class = 0

        if self.LIST[index, 3]=='G2':
            label_Grade=0
        elif self.LIST[index, 3] == 'G3':
            label_Grade = 1
        else:
            label_Grade=2 #### Useless


        if self.LIST[index, 4]=='WT':
            label_Diag = 0
        elif self.LIST[index, 5] == 'codel':
            label_Diag = 3
        else:
            if self.LIST[index, 6] == -2 or self.LIST[index, 6] == -1 or self.LIST[index, 3] =='G4':
                label_Diag = 1
            else:
                label_Diag = 2


        label=np.asarray([label_IDH,label_1p19q,label_CDKN,label_His,label_Grade,label_Diag,label_His_2class])

        return  label


    def shuffle_list(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        np.random.shuffle(self.LIST)



    def __len__(self):
        return self.LIST.shape[0]

class Our_Dataset_vis(Dataset):
    def __init__(self, phase,opt,if_end2end=False):
        super(Our_Dataset_vis, self).__init__()
        self.opt = opt
        self.patc_bs=64
        self.phase=phase
        self.if_end2end=if_end2end
        self.dataDir = (opt['dataDir']+'extract_224/') if opt['imgSize'][0]==224 else  (opt['dataDir']+'extract_512/')

        excel_label_wsi = pd.read_excel(opt['label_path'],sheet_name='wsi_level',header=0)
        excel_wsi =excel_label_wsi.values
        PATIENT_LIST=excel_wsi[:,0]
        np.random.seed(self.opt['seed'])
        random.seed(self.opt['seed'])
        PATIENT_LIST=list(PATIENT_LIST)


        PATIENT_LIST=np.unique(PATIENT_LIST)
        np.random.shuffle(PATIENT_LIST)
        NUM_PATIENT_ALL=len(PATIENT_LIST) # 952
        TEST_PATIENT_LIST=PATIENT_LIST[0:int(NUM_PATIENT_ALL)]
        TEST_WSI_LIST=os.listdir(r'/home/zeiler/WSI_proj/miccai/vis_results/set0/')
        self.TRAIN_LIST=[]
        self.VAL_LIST = []
        self.TEST_LIST = []
        for i in range(excel_wsi.shape[0]):# 2612
            if excel_wsi[:,1][i]+'.h5' in TEST_WSI_LIST:
                self.TEST_LIST.append(excel_wsi[i,:])
        self.LIST= np.asarray(self.TEST_LIST)


        self.train_iter_count=0
        self.Flat=0
        self.WSI_all=[]

    def __getitem__(self, index):
        feature_all ,read_details,self.LIST[index, 1]= self.read_feature(index)

        label=self.label_gene(index)

        return torch.from_numpy(np.array(feature_all)).float(),torch.from_numpy(label),read_details,self.LIST[index, 1]

    def read_feature(self, index):
        read_details = np.load(self.opt['dataDir'] + 'read_details/' + self.LIST[index, 1] + '.npy', allow_pickle=True)[
            0]
        num_patches = read_details.shape[0]
        root = self.opt['dataDir']+'Res50_feature_'+str(self.opt['fixdim'])+'_fixdim0/'
        patch_all = h5py.File(root + self.LIST[index, 1] + '.h5')['Res_feature'][:]  # (1,N,1024)
        return patch_all[0],read_details,self.LIST[index, 1]


    def label_gene(self,index):


        if self.LIST[index, 4]=='WT':
            label_IDH=0
        elif self.LIST[index, 4]=='Mutant':
            label_IDH=1
        if self.LIST[index, 5] == 'non-codel':
            label_1p19q = 0
        elif self.LIST[index, 5] == 'codel':
            label_1p19q = 1
        if self.LIST[index, 6] == -2 or self.LIST[index, 6] == -1:
            label_CDKN = 1
        else:
            label_CDKN = 0

        if self.LIST[index, 2]=='oligoastrocytoma':
            label_His = 0
        elif self.LIST[index, 2] == 'astrocytoma':
            label_His = 1
        elif self.LIST[index, 2] == 'oligodendroglioma':
            label_His = 2
        elif self.LIST[index, 2] == 'glioblastoma':
            label_His = 3

        if self.LIST[index, 2]=='glioblastoma':
            label_His_2class = 1
        else:
            label_His_2class = 0

        if self.LIST[index, 3]=='G2':
            label_Grade=0
        elif self.LIST[index, 3] == 'G3':
            label_Grade = 1
        else:
            label_Grade=2 #### Useless


        if self.LIST[index, 4]=='WT':
            label_Diag = 0
        elif self.LIST[index, 5] == 'codel':
            label_Diag = 3
        else:
            if self.LIST[index, 6] == -2 or self.LIST[index, 6] == -1 or self.LIST[index, 3] =='G4':
                label_Diag = 1
            else:
                label_Diag = 2


        label=np.asarray([label_IDH,label_1p19q,label_CDKN,label_His,label_Grade,label_Diag,label_His_2class])

        return  label


    def shuffle_list(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        np.random.shuffle(self.LIST)



    def __len__(self):
        return self.LIST.shape[0]

if __name__ == '__main__':
    # epoch_seed1 = np.arange(1000)
    # np.random.seed(100)
    # random.seed(100)
    # epoch_seed = np.arange(5)
    # np.random.shuffle(epoch_seed)
    # for i in range(5):
    #     np.random.seed(epoch_seed[i])
    #     random.seed(epoch_seed[i])
    #     np.random.shuffle(epoch_seed1)
    #     print(epoch_seed1[:20])


    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='config/mine.yml')
    args = parser.parse_args()
    with open(args.opt) as f:
        opt = yaml.load(f, Loader=SafeLoader)
    trainDataset = Our_Dataset(phase='Train', opt=opt)
    for i in range(100):
        _,x_,y_=trainDataset._getitem__(index=2000-i)