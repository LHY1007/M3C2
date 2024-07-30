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
from torchvision.transforms import Compose,  ToTensor, ToPILImage, CenterCrop, Resize
def train_transform(degree=180):
    return Compose([
        our_transform.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),
    ])
class Our_Dataset(Dataset):
    def __init__(self, phase,opt):
        super(Our_Dataset, self).__init__()
        self.opt = opt
        self.patc_bs=64
        self.phase=phase
        # self.test_mode=opt['test_mode'] # WSI  Patient
        self.name = opt['name'].split('_')

        excel_label_wsi = pd.read_excel(opt['label_path'],sheet_name='wsi_level',header=0)
        excel_wsi =excel_label_wsi.values
        PATIENT_LIST=excel_wsi[:,0]
        np.random.seed(self.opt['seed'])
        random.seed(self.opt['seed'])
        PATIENT_LIST=list(PATIENT_LIST)




        PATIENT_LIST=np.unique(PATIENT_LIST)
        np.random.shuffle(PATIENT_LIST)
        NUM_PATIENT_ALL=len(PATIENT_LIST) # 952
        TRAIN_PATIENT_LIST=PATIENT_LIST[0:int(NUM_PATIENT_ALL* 0.8)]
        VAL_PATIENT_LIST = PATIENT_LIST[int(NUM_PATIENT_ALL * 0.8):int(NUM_PATIENT_ALL * 0.9)]
        TEST_PATIENT_LIST = PATIENT_LIST[int(NUM_PATIENT_ALL * 0.9):]
        self.TRAIN_LIST=[]
        self.VAL_LIST = []
        self.TEST_LIST = []
        self.My_transform=train_transform()
        for i in range(excel_wsi.shape[0]):# 2612
            if excel_wsi[:,0][i] in TRAIN_PATIENT_LIST:
                self.TRAIN_LIST.append(excel_wsi[i,:])
            elif excel_wsi[:,0][i] in VAL_PATIENT_LIST:
                self.VAL_LIST.append(excel_wsi[i,:])
            elif excel_wsi[:,0][i] in TEST_PATIENT_LIST:
                self.TEST_LIST.append(excel_wsi[i,:])
        self.LIST= np.asarray(self.TRAIN_LIST) if self.phase == 'Train' else (np.asarray(self.VAL_LIST) if self.phase == 'Val' else np.asarray(self.TEST_LIST))
        # df = pd.DataFrame(self.LIST, columns=list(excel_label_wsi))
        # df.to_excel("vis/Val.xlsx", index=False)

        self.train_iter_count=0
        self.Flat=0
        self.WSI_all=[]

    def __getitem__(self, index):



        if self.name[2]=='img':
            self.read_img(index)
        elif self.name[2]=='fea':
            patch_all,coor_all=self.read_feature(index)
        label=self.label_gene(index)

        return torch.from_numpy(np.array(patch_all)).float(),torch.from_numpy(np.array(label)).long(), self.LIST[index, 1],coor_all

    def read_feature(self, index):
        root = self.opt['dataDir']+'TCGA/Res50_feature_'+str(self.opt['fixdim'])+'_fixdim0/'
        patch_all=h5py.File(root+self.LIST[index, 1]+'.h5')['Res_feature'][:] #(1,N,1024)
        coor_all = h5py.File(root + self.LIST[index, 1] + '.h5')['patches_coor'][:]
        return patch_all ,coor_all
    def read_feature1(self, index,k):
        root = self.opt['dataDir']+'Res50_feature_1200_fixdim0_aug/aug_set'+str(k)+'/'
        patch_all=h5py.File(root+self.LIST[index, 1]+'.h5')['Res_feature'][:] #(1,N,1024)
        coor_all = h5py.File(root + self.LIST[index, 1] + '.h5')['patches_coor'][:]
        return patch_all ,coor_all



    def read_img(self,index):
        wsi_path = self.dataDir + self.LIST[index, 1]
        patch_all = []
        patch_all_ori=[]
        coor_all=[]
        coor_all_ori = []
        self.img_dir = os.listdir(wsi_path)

        read_details=np.load(self.opt['dataDir']+'read_details/'+self.LIST[index, 1]+'.npy',allow_pickle=True)[0]
        num_patches = read_details.shape[0]
        print(num_patches)
        max_num=2500
        Use_patch_num = num_patches if num_patches <= max_num else max_num
        if num_patches <= max_num:
            times=int(np.floor(max_num/num_patches))
            remaining=max_num % num_patches
            for i in range(Use_patch_num):
                img_temp=io.imread(wsi_path + '/' + str(read_details[i][0]) + '_' + str(read_details[i][1]) + '.jpg')
                img_temp = cv2.resize(img_temp, (224, 224))
                patch_all_ori.append(img_temp)
                coor_all_ori.append(read_details[i])
            patch_all=patch_all_ori
            coor_all = coor_all_ori

            ####### fixdim0
            if times>1:
                for k in range(times-1):
                    patch_all=patch_all+patch_all_ori
                    coor_all=coor_all+coor_all_ori
            if not remaining==0:
                patch_all = patch_all + patch_all_ori[0:remaining]
                coor_all = coor_all + coor_all_ori[0:remaining]

        else:
            for i in range(Use_patch_num):
                img_temp = io.imread(wsi_path + '/' + str(read_details[int(np.around(i*(num_patches/max_num)))][0])+'_'+str(read_details[int(np.around(i*(num_patches/max_num)))][1])+'.jpg')
                img_temp = cv2.resize(img_temp, (224, 224))
                patch_all.append(img_temp)
                coor_all.append(read_details[int(np.around(i*(num_patches/max_num)))])

        patch_all = np.asarray(patch_all)

        # data augmentation
        patch_all = patch_all.reshape(-1, 224, 3)  # (num_patches*224,224,3)
        patch_all = patch_all.reshape(-1, 224, 224, 3)  # (num_patches,224,224,3)


        patch_all = patch_all / 255.0
        patch_all = np.transpose(patch_all, (0, 3, 1, 2))
        patch_all = patch_all.astype(np.float32)


        return patch_all,coor_all

    def label_gene(self,index):
        his_label_map= {'glioblastoma'}
        grade_label_map = {'2':0,'3':1,'4':2}
        #grade 2021 = {0: 'G2', 1: 'G3_O', 2: 'G4'}
        # His 2021 = {0: 'A', 1: 'O', 2: 'GBM'}
        #label 2021={ 0:'G2_O', 1:'G3_O', 2:'G2_A', 3:'G3_A', 4:'G4_A', 5:'GBM'}
        if self.name[1]=='IDH':
            if self.LIST[index, 4]=='WT':
                label=0
            elif self.LIST[index, 4]=='Mutant':
                label=1
        elif self.name[1] == '1p19q':
            if self.LIST[index, 5] == 'non-codel':
                label = 0
            elif self.LIST[index, 5] == 'codel':
                label = 1
        elif self.name[1] == 'CDKN':
            if self.LIST[index, 6] == -2 or self.LIST[index, 6] == -1:
                label = 1
            else:
                label = 0

        elif self.name[1] == 'Diag':
            if self.LIST[index, 4] == 'WT':
                label = 0
            elif self.LIST[index, 5] == 'codel':
                label = 3
            else:
                if self.LIST[index, 6] == -2 or self.LIST[index, 6] == -1 or self.LIST[index, 3] == 'G4':
                    label = 1
                else:
                    label = 2
        elif self.name[1] == 'Grade':
            if self.LIST[index, 4] == 'WT':
                label = 2
            elif self.LIST[index, 5] == 'codel':
                label = 0 if self.LIST[index, 3] =='G2' else 1
            else:
                if self.LIST[index, 6] == -2 or self.LIST[index, 6] == -1 or self.LIST[index, 3] =='G4':
                    label = 2
                else:
                    label = 0 if self.LIST[index, 3] == 'G2' else 1
        elif self.name[1] == 'His':
            # if self.LIST[index, 2]=='astrocytoma':
            #     label = 0
            # elif self.LIST[index, 2] == 'oligoastrocytoma':
            #     label = 1
            # elif self.LIST[index, 2] == 'oligodendroglioma':
            #     label = 2
            # elif self.LIST[index, 2] == 'glioblastoma':
            #     label = 3
            if self.LIST[index, 2] == 'glioblastoma':
                label = 1
            else:
                label = 0



        return  label


    def shuffle_list(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        np.random.shuffle(self.LIST)



    def __len__(self):
        return self.LIST.shape[0]



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='config/miccai.yml')
    args = parser.parse_args()
    with open(args.opt) as f:
        opt = yaml.load(f, Loader=SafeLoader)
    trainDataset = Our_Dataset(phase='Val', opt=opt)
    for i in range(100):
        trainDataset._getitem__(i)