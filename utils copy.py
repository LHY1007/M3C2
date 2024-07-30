import pandas as pd
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np
import math
import os
import glob
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import math
import os
import torchvision.models as models
import torch.cuda

import platform
import dataset
from torch.utils.data import DataLoader,Dataset
import argparse, time, random
import yaml
from yaml.loader import SafeLoader
from evaluation import *
import glob

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
from sklearn import metrics
from scipy import interp
""" curriculum learning based training strategy """

class CL_strategy():
    def __init__(self):
        super(CL_strategy, self).__init__()


def one_hot(data1, N=0):
    if N != 0:
        num = N
    else:
        num = int(np.max(data1) + 1)
    return (np.arange(num) == data1[:, None]).astype(np.int16)

def calculate_auc(predicted_all, label_all, num_classes):
    predicted_all=np.array(predicted_all)
    #out_cls_all_softmax_His = F.softmax(torch.from_numpy(np.array(predicted_all)), dim=1).numpy()
    label_all_np = np.array(label_all)
    label_all_onehot = one_hot(label_all_np, num_classes)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], predicted_all[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), predicted_all[:, 0:num_classes].ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    return roc_auc
""" spatial_pyramid_pooling """
import math
def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size=[]):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer

    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = int((h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2)
        w_pad = int((w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2)
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if (i == 0):
            spp = x.view(num_sample, -1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)
    return spp

def remove_all_file(path):
    if os.path.isdir(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)
            os.remove(path_file)







class NoneDict(dict):
    def __missing__(self, key):
        return None

def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def validation_CDKN_multiclass(opt,model,resnet, dataloader, saver, ep, eva_cm,gpuID):
    model.eval()
    # resnet.eval()
    if 1:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        label_all = []
        predicted_all = []
        pred_all=[]
        cm_y = []
        cm_pred = []

    test_bar = tqdm(dataloader)
    bs=opt['Val_batchSize']
    count=0
    for packs in test_bar:
        img = packs[0][0]
        label = packs[1]
        imgPath = packs[2]
        count += 1

        if torch.cuda.is_available():
            img = img.cuda(gpuID[0])
            label = label.cuda(gpuID[0])
        # img = resnet(img)
        if opt['name'].split('_')[0] == 'CLAM':
            results_dict = model(img[0], label[0])
        else:
            results_dict = model(img)
        pred_ori = results_dict['logits']
        _, pred = torch.max(pred_ori.data, 1)

        pred = pred.tolist()
        gt = label.tolist()

        for j in range(bs):
            label_all.append(gt[j])
            predicted_all.append(pred_ori.detach().cpu().numpy()[j][1])
            pred_all.append(pred[j])
            if (gt[j] == 0 or gt[j] == 1) and (pred[j] == 0 or pred[j] == 1):
                tn += 1
            if (gt[j] == 0 or gt[j] == 1) and (pred[j] == 2 or pred[j] == 3):
                fp += 1
            if (gt[j] == 2 or gt[j] == 3) and (pred[j] == 0 or pred[j] == 1):
                fn += 1
            if (gt[j] == 2 or gt[j] == 3) and (pred[j] == 2 or pred[j] == 3):
                tp += 1
            gt_cm_label = gt[j]
            pred_cm_label = pred[j]
            cm_y = np.append(cm_y, gt_cm_label)
            cm_pred = np.append(cm_pred, pred_cm_label)


    Acc = (tp + tn) / (tp + tn + fp + fn)
    Sen = (tp) / (tp + fn+0.000001)  # recall
    Spec = (tn) / (tn + fp+0.000001)
    precision = (tp) / (tp + fp+0.000001)
    recall = Sen
    f1_score = (2 * precision * recall) / (precision + recall+0.000001)
    AUC = 0
    if eva_cm:
        cm = confusion_matrix(cm_y, cm_pred)
    else:
        cm = None
    list=(Acc, cm, f1_score, Sen,Spec,AUC,precision)

    return list
def validation_Binary_IDH(opt,model,resnet, dataloader, saver, ep, eva_cm,gpuID):
    model.eval()
    if 1:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        label_all = []
        predicted_all = []
        pred_all=[]
        cm_y = []
        cm_pred = []

    test_bar = tqdm(dataloader)
    bs=opt['Val_batchSize']
    count=0
    for packs in test_bar:
        img = packs[0]
        label = packs[1]
        count += 1

        if torch.cuda.is_available():
            img = img.cuda(gpuID[0])
            label = label.cuda(gpuID[0])
        label_IDH = label[:, 0]
        results_dict = model(img)
        pred_ori = results_dict['logits'][0]
        _, pred = torch.max(pred_ori.data, 1)

        pred = pred.tolist()
        gt = label_IDH.tolist()
        # print(gt)

        for j in range(bs):
            label_all.append(gt[j])
            predicted_all.append(pred_ori.detach().cpu().numpy()[j][1])
            pred_all.append(pred[j])
            if gt[j] == 0 and pred[j] == 0:
                tn += 1
            if gt[j] == 0 and pred[j] == 1:
                fp += 1
            if gt[j] == 1 and pred[j] == 0:
                fn += 1
            if gt[j] == 1 and pred[j] == 1:
                tp += 1
            gt_cm_label = gt[j]
            pred_cm_label = pred[j]
            cm_y = np.append(cm_y, gt_cm_label)
            cm_pred = np.append(cm_pred, pred_cm_label)


    Acc = (tp + tn) / (tp + tn + fp + fn)
    Sen = (tp) / (tp + fn+0.000001)  # recall
    Spec = (tn) / (tn + fp+0.000001)
    precision = (tp) / (tp + fp+0.000001)
    recall = Sen
    f1_score = (2 * precision * recall) / (precision + recall+0.000001)
    AUC = metrics.roc_auc_score(y_true=np.array(label_all), y_score=np.array(predicted_all))
    if eva_cm:
        cm = confusion_matrix(cm_y, cm_pred)
    else:
        cm = None

    list=(Acc, cm, f1_score, Sen,Spec,AUC,precision)

    return list

def validation_Stage1(opt,Mine_model_init,Mine_model_His,Mine_model_Cls, dataloader, eva_cm,gpuID):


    Mine_model_init.eval()
    Mine_model_His.eval()
    Mine_model_Cls.eval()
    if 1:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        label_all = []
        predicted_all = []
        pred_all = []
        cm_y = []
        cm_pred = []

    test_bar = tqdm(dataloader)
    bs = opt['Val_batchSize']
    count = 0
    for packs in test_bar:
        img = packs[0]
        label = packs[1]
        count += 1

        if torch.cuda.is_available():
            img = img.cuda(gpuID[0])
            label = label.cuda(gpuID[0])
        label_IDH = label[:, 6]
        init_feature = Mine_model_init(img)  # (BS,2500,1024)
        hidden_states, encoded_His = Mine_model_His(init_feature)
        results_dict, weight_His_GBM, weight_His_O = Mine_model_Cls(encoded_His)
        pred_ori = results_dict['logits_His_2class']
        _, pred = torch.max(pred_ori.data, 1)

        pred = pred.tolist()
        gt = label_IDH.tolist()
        # print(gt)

        for j in range(bs):
            label_all.append(gt[j])
            predicted_all.append(pred_ori.detach().cpu().numpy()[j][1])
            pred_all.append(pred[j])
            if gt[j] == 0 and pred[j] == 0:
                tn += 1
            if gt[j] == 0 and pred[j] == 1:
                fp += 1
            if gt[j] == 1 and pred[j] == 0:
                fn += 1
            if gt[j] == 1 and pred[j] == 1:
                tp += 1
            gt_cm_label = gt[j]
            pred_cm_label = pred[j]
            cm_y = np.append(cm_y, gt_cm_label)
            cm_pred = np.append(cm_pred, pred_cm_label)

    Acc = (tp + tn) / (tp + tn + fp + fn)
    Sen = (tp) / (tp + fn + 0.000001)  # recall
    Spec = (tn) / (tn + fp + 0.000001)
    precision = (tp) / (tp + fp + 0.000001)
    recall = Sen
    f1_score = (2 * precision * recall) / (precision + recall + 0.000001)
    AUC = metrics.roc_auc_score(y_true=np.array(label_all), y_score=np.array(predicted_all))


    list = (Acc, None, f1_score, Sen, Spec, AUC, precision)

    return list


def validation_Binary_Grade(opt,model,resnet, dataloader, saver, ep, eva_cm,gpuID):
    model.eval()
    if 1:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        label_all = []
        predicted_all = []
        pred_all=[]
        cm_y = []
        cm_pred = []

    test_bar = tqdm(dataloader)
    bs=opt['Val_batchSize']
    count=0
    for packs in test_bar:
        img = packs[0]
        label = packs[1]
        count += 1

        if torch.cuda.is_available():
            img = img.cuda(gpuID[0])
            label = label.cuda(gpuID[0])
        label_IDH = label[:, 4]
        results_dict = model(img)
        pred_ori = results_dict['logits'][4][:,0:2]
        _, pred = torch.max(pred_ori.data, 1)

        pred = pred.tolist()
        gt = label_IDH.tolist()
        # print(gt)

        for j in range(bs):
            if not gt[j]==2:
                label_all.append(gt[j])
                predicted_all.append(F.softmax(pred_ori, dim=1).detach().cpu().numpy()[j][1])
                pred_all.append(pred[j])
                if gt[j] == 0 and pred[j] == 0:
                    tn += 1
                if gt[j] == 0 and pred[j] == 1:
                    fp += 1
                if gt[j] == 1 and pred[j] == 0:
                    fn += 1
                if gt[j] == 1 and pred[j] == 1:
                    tp += 1
                gt_cm_label = gt[j]
                pred_cm_label = pred[j]
                cm_y = np.append(cm_y, gt_cm_label)
                cm_pred = np.append(cm_pred, pred_cm_label)


    Acc = (tp + tn) / (tp + tn + fp + fn)
    Sen = (tp) / (tp + fn+0.000001)  # recall
    Spec = (tn) / (tn + fp+0.000001)
    precision = (tp) / (tp + fp+0.000001)
    recall = Sen
    f1_score = (2 * precision * recall) / (precision + recall+0.000001)
    AUC = metrics.roc_auc_score(y_true=np.array(label_all), y_score=np.array(predicted_all))
    if eva_cm:
        cm = confusion_matrix(cm_y, cm_pred)
    else:
        cm = None

    list=(Acc, cm, f1_score, Sen,Spec,AUC,precision)

    return list



def validation_Binary_1p19q(opt,Mine_model_init,Mine_model_IDH,Mine_model_1p19q, dataloader, saver, ep, eva_cm,gpuID):
    Mine_model_init.eval()
    Mine_model_IDH.eval()
    Mine_model_1p19q.eval()
    if 1:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        label_all = []
        predicted_all = []
        pred_all=[]
        cm_y = []
        cm_pred = []

    test_bar = tqdm(dataloader)
    bs=opt['Val_batchSize']
    count=0
    for packs in test_bar:
        img = packs[0]
        label = packs[1]
        count += 1

        if torch.cuda.is_available():
            img = img.cuda(gpuID[0])
            label = label.cuda(gpuID[0])
        label_IDH = label[:, 1]

        init_feature = Mine_model_init(img)  # (BS,2500,1024)
        feature_IDH, results_dict = Mine_model_IDH(init_feature)
        feature_1p19q, results_dict = Mine_model_1p19q(feature_IDH)
        pred_1p19q = results_dict['logits']

        _, pred = torch.max(pred_1p19q.data, 1)

        pred = pred.tolist()
        gt = label_IDH.tolist()
        # print(gt)

        for j in range(bs):
            label_all.append(gt[j])
            predicted_all.append(pred_1p19q.detach().cpu().numpy()[j][1])
            pred_all.append(pred[j])
            if gt[j] == 0 and pred[j] == 0:
                tn += 1
            if gt[j] == 0 and pred[j] == 1:
                fp += 1
            if gt[j] == 1 and pred[j] == 0:
                fn += 1
            if gt[j] == 1 and pred[j] == 1:
                tp += 1
            gt_cm_label = gt[j]
            pred_cm_label = pred[j]
            cm_y = np.append(cm_y, gt_cm_label)
            cm_pred = np.append(cm_pred, pred_cm_label)


    Acc = (tp + tn) / (tp + tn + fp + fn)
    Sen = (tp) / (tp + fn+0.000001)  # recall
    Spec = (tn) / (tn + fp+0.000001)
    precision = (tp) / (tp + fp+0.000001)
    recall = Sen
    f1_score = (2 * precision * recall) / (precision + recall+0.000001)
    AUC = metrics.roc_auc_score(y_true=np.array(label_all), y_score=np.array(predicted_all))
    if eva_cm:
        cm = confusion_matrix(cm_y, cm_pred)
    else:
        cm = None

    list=(Acc, cm, f1_score, Sen,Spec,AUC,precision)

    return list

def validation_Binary_CDKN(opt,model,resnet, dataloader, saver, ep, eva_cm,gpuID):
    model.eval()
    if 1:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        label_all = []
        predicted_all = []
        pred_all=[]
        cm_y = []
        cm_pred = []

    test_bar = tqdm(dataloader)
    bs=opt['Val_batchSize']
    count=0
    for packs in test_bar:
        img = packs[0]
        label = packs[1]
        count += 1

        if torch.cuda.is_available():
            img = img.cuda(gpuID[0])
            label = label.cuda(gpuID[0])
        label_IDH = label[:, 2]
        results_dict = model(img)
        pred_ori = results_dict['logits'][2]
        _, pred = torch.max(pred_ori.data, 1)

        pred = pred.tolist()
        gt = label_IDH.tolist()
        # print(gt)

        for j in range(bs):
            label_all.append(gt[j])
            predicted_all.append(pred_ori.detach().cpu().numpy()[j][1])
            pred_all.append(pred[j])
            if gt[j] == 0 and pred[j] == 0:
                tn += 1
            if gt[j] == 0 and pred[j] == 1:
                fp += 1
            if gt[j] == 1 and pred[j] == 0:
                fn += 1
            if gt[j] == 1 and pred[j] == 1:
                tp += 1
            gt_cm_label = gt[j]
            pred_cm_label = pred[j]
            cm_y = np.append(cm_y, gt_cm_label)
            cm_pred = np.append(cm_pred, pred_cm_label)


    Acc = (tp + tn) / (tp + tn + fp + fn)
    Sen = (tp) / (tp + fn+0.000001)  # recall
    Spec = (tn) / (tn + fp+0.000001)
    precision = (tp) / (tp + fp+0.000001)
    recall = Sen
    f1_score = (2 * precision * recall) / (precision + recall+0.000001)
    AUC = metrics.roc_auc_score(y_true=np.array(label_all), y_score=np.array(predicted_all))
    if eva_cm:
        cm = confusion_matrix(cm_y, cm_pred)
    else:
        cm = None

    list=(Acc, cm, f1_score, Sen,Spec,AUC,precision)

    return list


def validation_Binary(opt,model,model_stem, dataloader, saver, ep, eva_cm,gpuID):
    model.train()

    if 1:
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        label_all = []
        predicted_all = []
        pred_all=[]
        cm_y = []
        cm_pred = []

    test_bar = tqdm(dataloader)
    bs=opt['Val_batchSize']
    count=0
    for packs in test_bar:
        img = packs[0][0]
        label = packs[1]
        count += 1

        if torch.cuda.is_available():
            img = img.cuda(gpuID[0])
            label = label.cuda(gpuID[0])
        if opt['name'].split('_')[0] == 'CLAM':
            results_dict = model(img[0], label[0])
        else:
            results_dict = model(img)
        pred_ori = results_dict['logits']
        _, pred = torch.max(pred_ori.data, 1)

        pred = pred.tolist()
        gt = label.tolist()
        # print(pred)
        # print(gt)

        for j in range(bs):
            label_all.append(gt[j])
            pred_ori_softmax = F.softmax(pred_ori, dim=1)
            predicted_all.append(pred_ori_softmax.detach().cpu().numpy()[j][1])
            pred_all.append(pred[j])
            if gt[j] == 0 and pred[j] == 0:
                tn += 1
            if gt[j] == 0 and pred[j] == 1:
                fp += 1
            if gt[j] == 1 and pred[j] == 0:
                fn += 1
            if gt[j] == 1 and pred[j] == 1:
                tp += 1
            gt_cm_label = gt[j]
            pred_cm_label = pred[j]
            cm_y = np.append(cm_y, gt_cm_label)
            cm_pred = np.append(cm_pred, pred_cm_label)


    Acc = (tp + tn) / (tp + tn + fp + fn)
    Sen = (tp) / (tp + fn+0.000001)  # recall
    Spec = (tn) / (tn + fp+0.000001)
    precision = (tp) / (tp + fp+0.000001)
    recall = Sen
    f1_score = (2 * precision * recall) / (precision + recall+0.000001)
    AUC = metrics.roc_auc_score(y_true=np.array(label_all), y_score=np.array(predicted_all))

    dataframe = pd.DataFrame({'label': label_all, 'score': predicted_all, 'sen': Sen, 'spec': Spec})
    dataframe.to_excel('plot/'+opt['name']+'.xlsx', index=False)
    list=(Acc, 0, f1_score, Sen,Spec,AUC,precision)

    return list


def validation_cifar(opt,model,resnet, dataloader, saver, ep, eva_cm,gpuID):
    model.eval()
    # resnet.eval()
    if 1:

        count_cifar = 0
        correct_cifar = 0

        label_all_cifar = []
        predicted_all_cifar = []
        pred_all_cifar = []
        cm_y_cifar = []
        cm_pred_cifar = []
    test_bar = tqdm(dataloader)
    bs = opt['Val_batchSize']
    count = 0
    for packs in test_bar:
        img = packs[0][0]
        label = packs[1]
        count += 1

        if torch.cuda.is_available():
            img = img.cuda(gpuID[0])
            label = label.cuda(gpuID[0])
        # img = resnet(img)
        if opt['name'].split('_')[0]=='CLAM':
            results_dict = model(img[0], label[0])
        else:
            results_dict = model(img)
        pred_ori = results_dict['logits']

        _, pred_cifar = torch.max(pred_ori.data, 1)
        pred_cifar = pred_cifar.tolist()
        gt_cifar = label.tolist()


        for j in range(bs):
            ##################   cifar

            gt_cm_label_cifar = gt_cifar[j]
            pred_cm_label_cifar = pred_cifar[j]
            cm_y_cifar = np.append(cm_y_cifar, gt_cm_label_cifar)
            cm_pred_cifar = np.append(cm_pred_cifar, pred_cm_label_cifar)
            label_all_cifar.append(gt_cifar[j])
            predicted_all_cifar.append(pred_ori.detach().cpu().numpy()[j])
            count_cifar += 1
            if gt_cifar[j] == pred_cifar[j]:
                correct_cifar += 1


    ################################################   cifar
    Acc_cifar = correct_cifar / count_cifar



    return Acc_cifar



def validation_His(opt,model,resnet, dataloader, saver, ep, eva_cm,gpuID):
    model.eval()
    # resnet.eval()
    if 1:

        count_His = 0
        count_His_NoOA = 0
        correct_His = 0
        correct_His2=0
        correct_His3 = 0
        A_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                        'AUC': 0}
        AO_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                        'AUC': 0}
        O_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                        'AUC': 0}
        GBM_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                        'AUC': 0}
        all_metrics = {'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}

        label_all_His = []
        predicted_all_His = []

    test_bar = tqdm(dataloader)
    bs = opt['Val_batchSize']
    count = 0
    for packs in test_bar:
        img = packs[0]
        label = packs[1]
        count += 1

        if torch.cuda.is_available():
            img = img.cuda(gpuID[0])
            label = label.cuda(gpuID[0])
        label_His = label[:, 3]
        if opt['name'].split('_')[0]=='CLAM':
            results_dict = model(img[0], label[0])
        else:
            results_dict = model(img)
        pred_ori = results_dict['logits'][3]

        _, pred_His = torch.max(pred_ori.data, 1)
        _, pred_His_2 = torch.max(pred_ori[:,1:].data, 1)
        pred_His = pred_His.tolist()  # [BS] AO A  O GBM //0 1 2 3
        pred_His_2 = pred_His_2.tolist()  #[BS] A  O GBM //0 1 2
        gt_His = label_His.tolist() #[BS] AO A  O GBM//0 1 2 3


        for j in range(bs):
            ##################   His
            # AO
            # if pred_His[j] == 0:
            #     if gt_His[j] == 0 :
            #         AO_metrics['tn'] += 1
            #     else:
            #         AO_metrics['fn'] += 1
            # else:
            #     if not gt_His[j] == 0:
            #         AO_metrics['tp'] += 1
            #     else:
            #         AO_metrics['fp'] += 1
            if not gt_His[j] == 0:
                # A
                if pred_His_2[j] == 0:
                    if gt_His[j] == 1:
                        A_metrics['tn'] += 1
                    else:
                        A_metrics['fn'] += 1
                else:
                    if not gt_His[j] == 1:
                        A_metrics['tp'] += 1
                    else:
                        A_metrics['fp'] += 1
                # O
                if pred_His_2[j] == 1:
                    if gt_His[j] == 2:
                        O_metrics['tn'] += 1
                    else:
                        O_metrics['fn'] += 1
                else:
                    if not gt_His[j] == 2:
                        O_metrics['tp'] += 1
                    else:
                        O_metrics['fp'] += 1
                # GBM
                if pred_His_2[j] == 2:
                    if gt_His[j] == 3:
                        GBM_metrics['tn'] += 1
                    else:
                        GBM_metrics['fn'] += 1
                else:
                    if not gt_His[j] == 3:
                        GBM_metrics['tp'] += 1
                    else:
                        GBM_metrics['fp'] += 1

            gt_cm_label_His = gt_His[j]
            pred_cm_label_His = pred_His[j]
            cm_y_His = np.append(cm_y_His, gt_cm_label_His)
            cm_pred_His = np.append(cm_pred_His, pred_cm_label_His)
            label_all_His.append(gt_His[j])
            predicted_all_His.append(pred_ori.detach().cpu().numpy()[j])
            count_His += 1

            if gt_His[j] == pred_His[j]:
                correct_His2 += 1

            if gt_His[j] == 0 and (pred_His_2[j] == 0 or pred_His_2[j] == 1):
                correct_His+=1
            if gt_His[j] == 1 and pred_His_2[j]==0:
                correct_His += 1
            if gt_His[j] == 2 and pred_His_2[j]==1:
                correct_His += 1
            if gt_His[j] == 3 and pred_His_2[j]==2:
                correct_His += 1


            if not gt_His[j] == 0:
                count_His_NoOA+= 1
            if gt_His[j] == 1 and pred_His_2[j]==0:
                correct_His3 += 1
            if gt_His[j] == 2 and pred_His_2[j]==1:
                correct_His3 += 1
            if gt_His[j] == 3 and pred_His_2[j]==2:
                correct_His3 += 1


    ################################################   His
    Acc_His = correct_His / count_His
    Acc_His2 = correct_His2 / count_His
    Acc_His3 = correct_His3 / count_His_NoOA
    #  Sensitivity
    A_metrics['sen'] = (A_metrics['tp']) / (A_metrics['tp'] + A_metrics['fn']+0.000001)
    # AO_metrics['sen'] = (AO_metrics['tp']) / (AO_metrics['tp'] + AO_metrics['fn']+0.000001)
    O_metrics['sen'] = (O_metrics['tp']) / (O_metrics['tp'] + O_metrics['fn']+0.000001)
    GBM_metrics['sen'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fn']+0.000001)
    all_metrics['sen'] = (A_metrics['sen'] + AO_metrics['sen'] + O_metrics['sen'] +
                          GBM_metrics['sen'] ) / 3
    #  Spec
    A_metrics['spec'] = (A_metrics['tn']) / (A_metrics['tn'] + A_metrics['fp']+0.000001)
    # AO_metrics['spec'] = (AO_metrics['tn']) / (AO_metrics['tn'] + AO_metrics['fp']+0.000001)
    O_metrics['spec'] = (O_metrics['tn']) / (O_metrics['tn'] + O_metrics['fp']+0.000001)
    GBM_metrics['spec'] = (GBM_metrics['tn']) / (GBM_metrics['tn'] + GBM_metrics['fp']+0.000001)
    all_metrics['spec'] = (A_metrics['spec'] + AO_metrics['spec'] + O_metrics['spec'] +
                           GBM_metrics['spec'] ) / 3
    #  Precision
    A_metrics['pre'] = (A_metrics['tp']) / (A_metrics['tp'] + A_metrics['fp']+0.000001)
    # AO_metrics['pre'] = (AO_metrics['tp']) / (AO_metrics['tp'] + AO_metrics['fp']+0.000001)
    O_metrics['pre'] = (O_metrics['tp']) / (O_metrics['tp'] + O_metrics['fp']+0.000001)
    GBM_metrics['pre'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fp']+0.000001)
    all_metrics['pre'] = (A_metrics['pre'] + AO_metrics['pre'] + O_metrics['pre'] +
                          GBM_metrics['pre'] ) / 3
    #  Recall
    A_metrics['recall'] = (A_metrics['tp']) / (A_metrics['tp'] + A_metrics['fn']+0.000001)
    # AO_metrics['recall'] = (AO_metrics['tp']) / (AO_metrics['tp'] + AO_metrics['fn']+0.000001)
    O_metrics['recall'] = (O_metrics['tp']) / (O_metrics['tp'] + O_metrics['fn']+0.000001)
    GBM_metrics['recall'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fn']+0.000001)
    all_metrics['recall'] = (A_metrics['recall'] + AO_metrics['recall'] + O_metrics['recall'] +
                             GBM_metrics['recall'] ) / 3
    #  F1
    A_metrics['f1'] = (2 * A_metrics['pre'] * A_metrics['recall']) / (
                A_metrics['pre'] + A_metrics['recall']+0.000001)
    # AO_metrics['f1'] = (2 * AO_metrics['pre'] * AO_metrics['recall']) / (
    #             AO_metrics['pre'] + AO_metrics['recall']+0.000001)
    O_metrics['f1'] = (2 * O_metrics['pre'] * O_metrics['recall']) / (
                O_metrics['pre'] + O_metrics['recall']+0.000001)
    GBM_metrics['f1'] = (2 * GBM_metrics['pre'] * GBM_metrics['recall']) / (GBM_metrics['pre'] + GBM_metrics['recall']+0.000001)
    all_metrics['f1'] = (A_metrics['f1'] + AO_metrics['f1'] + O_metrics['f1'] +
                          GBM_metrics['f1']) / 3
    # AUC
    # out_cls_all_softmax = F.softmax(torch.from_numpy(np.array(predicted_all_His)), dim=1).numpy()
    # label_all_np = np.array(label_all_His)
    # label_all_onehot = make_one_hot(label_all_np)
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(4):
    #     fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax[:, i])
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(4):
    #     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # mean_tpr /= 4
    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # all_metrics['AUC'] = roc_auc["macro"]

    list_His = ( 0,None, all_metrics['f1'], all_metrics['sen'], all_metrics['spec'], 0 ,
                 all_metrics['pre'],Acc_His3)

    return list_His


def validation_Diag(opt,model,resnet, dataloader, saver, ep, eva_cm,gpuID):
    model.eval()
    # resnet.eval()
    if 1:

        count_Diag = 0
        correct_Diag = 0
        G23_O_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                         'AUC': 0}
        G23_A_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                         'AUC': 0}
        G4_A_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                        'AUC': 0}
        GBM_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                       'AUC': 0}
        all_metrics = {'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}

        label_all_Diag = []
        predicted_all_Diag = []
        pred_all_Diag = []
        cm_y_Diag = []
        cm_pred_Diag = []
    test_bar = tqdm(dataloader)
    bs = opt['Val_batchSize']
    count = 0
    for packs in test_bar:
        img = packs[0][0]
        label = packs[1]
        imgPath = packs[2]
        count += 1

        if torch.cuda.is_available():
            img = img.cuda(gpuID[0])
            label = label.cuda(gpuID[0])
        # img = resnet(img)
        if opt['name'].split('_')[0]=='CLAM':
            results_dict = model(img[0], label[0])
        else:
            results_dict = model(img)
        pred_ori = results_dict['logits']

        _, pred_Diag = torch.max(pred_ori.data, 1)
        pred_Diag = pred_Diag.tolist()
        gt_Diag = label.tolist()


        for j in range(bs):
            ##################   Diag
            # GBM
            if pred_Diag[j] == 0:
                if gt_Diag[j] == 0:
                    GBM_metrics['tp'] += 1
                else:
                    GBM_metrics['fn'] += 1
            else:
                if not gt_Diag[j] == 0:
                    GBM_metrics['tn'] += 1
                else:
                    GBM_metrics['fp'] += 1
            # G4_A
            if pred_Diag[j] == 1:
                if gt_Diag[j] == 1:
                    G4_A_metrics['tp'] += 1
                else:
                    G4_A_metrics['fn'] += 1
            else:
                if not gt_Diag[j] == 1:
                    G4_A_metrics['tn'] += 1
                else:
                    G4_A_metrics['fp'] += 1
            # G23_A
            if pred_Diag[j] == 2:
                if gt_Diag[j] == 2:
                    G23_A_metrics['tp'] += 1
                else:
                    G23_A_metrics['fn'] += 1
            else:
                if not gt_Diag[j] == 2:
                    G23_A_metrics['tn'] += 1
                else:
                    G23_A_metrics['fp'] += 1
            # G23_O
            if pred_Diag[j] == 3:
                if gt_Diag[j] == 3:
                    G23_O_metrics['tn'] += 1
                else:
                    G23_O_metrics['fn'] += 1
            else:
                if not gt_Diag[j] == 3:
                    G23_O_metrics['tp'] += 1
                else:
                    G23_O_metrics['fp'] += 1

            gt_cm_label_Diag = gt_Diag[j]
            pred_cm_label_Diag = pred_Diag[j]
            cm_y_Diag = np.append(cm_y_Diag, gt_cm_label_Diag)
            cm_pred_Diag = np.append(cm_pred_Diag, pred_cm_label_Diag)
            label_all_Diag.append(gt_Diag[j])
            predicted_all_Diag.append(pred_ori.detach().cpu().numpy()[j])
            count_Diag += 1
            if gt_Diag[j] == pred_Diag[j]:
                correct_Diag += 1


    ################################################   Diag
    Acc_Diag = correct_Diag / count_Diag
    #  Sensitivity
    G23_O_metrics['sen'] = (G23_O_metrics['tp']) / (G23_O_metrics['tp'] + G23_O_metrics['fn'] + 0.000001)
    G23_A_metrics['sen'] = (G23_A_metrics['tp']) / (G23_A_metrics['tp'] + G23_A_metrics['fn'] + 0.000001)
    G4_A_metrics['sen'] = (G4_A_metrics['tp']) / (G4_A_metrics['tp'] + G4_A_metrics['fn'] + 0.000001)
    GBM_metrics['sen'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fn'] + 0.000001)
    all_metrics['sen'] = (G23_O_metrics['sen'] + G23_A_metrics['sen'] + G4_A_metrics['sen'] + GBM_metrics['sen']) / 4
    #  Spec
    G23_O_metrics['spec'] = (G23_O_metrics['tn']) / (G23_O_metrics['tn'] + G23_O_metrics['fp'] + 0.000001)
    G23_A_metrics['spec'] = (G23_A_metrics['tn']) / (G23_A_metrics['tn'] + G23_A_metrics['fp'] + 0.000001)
    G4_A_metrics['spec'] = (G4_A_metrics['tn']) / (G4_A_metrics['tn'] + G4_A_metrics['fp'] + 0.000001)
    GBM_metrics['spec'] = (GBM_metrics['tn']) / (GBM_metrics['tn'] + GBM_metrics['fp'] + 0.000001)
    all_metrics['spec'] = (G23_O_metrics['spec'] + G23_A_metrics['spec'] + G4_A_metrics['spec'] + GBM_metrics[
        'spec']) / 4
    #  Precision
    G23_O_metrics['pre'] = (G23_O_metrics['tp']) / (G23_O_metrics['tp'] + G23_O_metrics['fp'] + 0.000001)
    G23_A_metrics['pre'] = (G23_A_metrics['tp']) / (G23_A_metrics['tp'] + G23_A_metrics['fp'] + 0.000001)
    G4_A_metrics['pre'] = (G4_A_metrics['tp']) / (G4_A_metrics['tp'] + G4_A_metrics['fp'] + 0.000001)
    GBM_metrics['pre'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fp'] + 0.000001)
    all_metrics['pre'] = (G23_O_metrics['pre'] + G23_A_metrics['pre'] + G4_A_metrics['pre'] + GBM_metrics['pre']) / 4
    #  Recall
    G23_O_metrics['recall'] = (G23_O_metrics['tp']) / (G23_O_metrics['tp'] + G23_O_metrics['fn'] + 0.000001)
    G23_A_metrics['recall'] = (G23_A_metrics['tp']) / (G23_A_metrics['tp'] + G23_A_metrics['fn'] + 0.000001)
    G4_A_metrics['recall'] = (G4_A_metrics['tp']) / (G4_A_metrics['tp'] + G4_A_metrics['fn'] + 0.000001)
    GBM_metrics['recall'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fn'] + 0.000001)
    all_metrics['recall'] = (G23_O_metrics['recall'] + G23_A_metrics['recall'] + G4_A_metrics['recall'] + GBM_metrics[
        'recall']) / 4

    #  F1
    G23_O_metrics['f1'] = 2 * (G23_O_metrics['pre'] * G23_O_metrics['recall']) / (
                G23_O_metrics['pre'] + G23_O_metrics['recall'] + 0.000001)
    G23_A_metrics['f1'] = 2 * (G23_A_metrics['pre'] * G23_A_metrics['recall']) / (
                G23_A_metrics['pre'] + G23_A_metrics['recall'] + 0.000001)
    G4_A_metrics['f1'] = 2 * (G4_A_metrics['pre'] * G4_A_metrics['recall']) / (
                G4_A_metrics['pre'] + G4_A_metrics['recall'] + 0.000001)
    GBM_metrics['f1'] = 2 * (GBM_metrics['pre'] * GBM_metrics['recall']) / (
                GBM_metrics['pre'] + GBM_metrics['recall'] + 0.000001)
    all_metrics['f1'] = (G23_O_metrics['f1'] + G23_A_metrics['f1'] + G4_A_metrics['f1'] + GBM_metrics['f1']) / 4
    # AUC

    # out_cls_all_softmax = F.softmax(torch.from_numpy(np.array(predicted_all_Diag)), dim=1).numpy()
    # label_all_np = np.array(label_all_Diag)
    # label_all_onehot = make_one_hot(label_all_np)
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(6):
    #     fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax[:, i])
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(6)]))
    #
    # # Then interpolate all ROC curves at this points
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(6):
    #     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # # Finally average it and compute AUC
    # mean_tpr /= 6
    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # all_metrics['AUC'] = roc_auc["macro"]
    if eva_cm:
        cm_Diag = confusion_matrix(cm_y_Diag, cm_pred_Diag)
    else:
        cm_Diag = None
    list_Diag = (Acc_Diag, cm_Diag, all_metrics['f1'], all_metrics['sen'], all_metrics['spec'], all_metrics['AUC'],
                 all_metrics['pre'])

    return list_Diag

def validation_All(opt,Mine_model_init, Mine_model_IDH, Mine_model_1p19q,Mine_model_CDKN,Mine_model_Graph,Mine_model_His,Mine_model_Cls,Mine_model_Task, dataloader, saver, ep, eva_cm,gpuID,task = '',t=0):
    Mine_model_init.eval()
    Mine_model_IDH.eval()
    Mine_model_1p19q.eval()
    Mine_model_CDKN.eval()
    Mine_model_Graph.eval()
    Mine_model_His.eval()
    # Mine_model_Grade.eval()
    Mine_model_Cls.eval()
    Mine_model_Task.eval()
    if 1:
        tp_IDH = 0
        tn_IDH = 0
        fp_IDH = 0
        fn_IDH = 0
        label_all_IDH = []
        predicted_all_IDH = []

        tp_1p19q = 0
        tn_1p19q = 0
        fp_1p19q = 0
        fn_1p19q = 0
        label_all_1p19q = []
        predicted_all_1p19q = []

        tp_CDKN = 0
        tn_CDKN = 0
        fp_CDKN = 0
        fn_CDKN = 0
        label_all_CDKN = []
        predicted_all_CDKN = []

        tp_His_2class = 0
        tn_His_2class = 0
        fp_His_2class= 0
        fn_His_2class = 0
        label_all_His_2class = []
        predicted_all_His_2class = []


        count_Diag = 0
        correct_Diag = 0
        G23_O_metrics_Diag =    {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0,'sen_weight': 0,'sen_micro': 0, 'spec': 0,'spec_weight': 0,'spec_micro': 0,'pre': 0,'pre_weight': 0,'pre_micro': 0, 'recall': 0,'recall_weight': 0,'recall_micro': 0, 'f1': 0,'f1_weight': 0,'f1_micro': 0, 'AUC': 0}
        G23_A_metrics_Diag =    {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0,'sen_weight': 0,'sen_micro': 0, 'spec': 0,'spec_weight': 0,'spec_micro': 0,'pre': 0,'pre_weight': 0,'pre_micro': 0, 'recall': 0,'recall_weight': 0,'recall_micro': 0, 'f1': 0,'f1_weight': 0,'f1_micro': 0, 'AUC': 0}
        G4_A_metrics_Diag =     {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0,'sen_weight': 0,'sen_micro': 0, 'spec': 0,'spec_weight': 0,'spec_micro': 0,'pre': 0,'pre_weight': 0,'pre_micro': 0, 'recall': 0,'recall_weight': 0,'recall_micro': 0, 'f1': 0,'f1_weight': 0,'f1_micro': 0, 'AUC': 0}
        GBM_metrics_Diag =      {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0,'sen_weight': 0,'sen_micro': 0, 'spec': 0,'spec_weight': 0,'spec_micro': 0,'pre': 0,'pre_weight': 0,'pre_micro': 0, 'recall': 0,'recall_weight': 0,'recall_micro': 0, 'f1': 0,'f1_weight': 0,'f1_micro': 0, 'AUC': 0}
        all_metrics_Diag =      {'sen': 0,'sen_weight': 0,'sen_micro': 0, 'spec': 0,'spec_weight': 0,'spec_micro': 0,'pre': 0,'pre_weight': 0,'pre_micro': 0, 'recall': 0,'recall_weight': 0,'recall_micro': 0, 'f1': 0,'f1_weight': 0,'f1_micro': 0, 'AUC': 0}
        label_all_Diag = []
        predicted_all_Diag = []   
        pred_all_Diag = []

        count_His = 0
        correct_His = 0
        A_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        O_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        GBM_His_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        all_metrics_His = {'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        label_all_His = []
        predicted_all_His = []

        # count_Grade = 0
        # correct_Grade = 0
        # G2_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        # G3_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        # G4_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        # all_metrics_Grade = {'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        # label_all_Grade = []
        # predicted_all_Grade = []

        count_Task = 0
        correct_Task = 0
        G23_O_metrics_Task =    {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0,'sen_weight': 0,'sen_micro': 0, 'spec': 0,'spec_weight': 0,'spec_micro': 0,'pre': 0,'pre_weight': 0,'pre_micro': 0, 'recall': 0,'recall_weight': 0,'recall_micro': 0, 'f1': 0,'f1_weight': 0,'f1_micro': 0, 'AUC': 0}
        G23_A_metrics_Task =    {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0,'sen_weight': 0,'sen_micro': 0, 'spec': 0,'spec_weight': 0,'spec_micro': 0,'pre': 0,'pre_weight': 0,'pre_micro': 0, 'recall': 0,'recall_weight': 0,'recall_micro': 0, 'f1': 0,'f1_weight': 0,'f1_micro': 0, 'AUC': 0}
        G4_A_metrics_Task =     {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0,'sen_weight': 0,'sen_micro': 0, 'spec': 0,'spec_weight': 0,'spec_micro': 0,'pre': 0,'pre_weight': 0,'pre_micro': 0, 'recall': 0,'recall_weight': 0,'recall_micro': 0, 'f1': 0,'f1_weight': 0,'f1_micro': 0, 'AUC': 0}
        GBM_metrics_Task =      {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0,'sen_weight': 0,'sen_micro': 0, 'spec': 0,'spec_weight': 0,'spec_micro': 0,'pre': 0,'pre_weight': 0,'pre_micro': 0, 'recall': 0,'recall_weight': 0,'recall_micro': 0, 'f1': 0,'f1_weight': 0,'f1_micro': 0, 'AUC': 0}
        all_metrics_Task =      {'sen': 0,'sen_weight': 0,'sen_micro': 0, 'spec': 0,'spec_weight': 0,'spec_micro': 0,'pre': 0,'pre_weight': 0,'pre_micro': 0, 'recall': 0,'recall_weight': 0,'recall_micro': 0, 'f1': 0,'f1_weight': 0,'f1_micro': 0, 'AUC': 0}
        label_all_Task = []
        predicted_all_Task = []
        pred_all_Task = []


        # tp_Grade = 0
        # tn_Grade = 0
        # fp_Grade = 0
        # fn_Grade = 0
        # label_all_Grade = []
        # predicted_all_Grade = []

    test_bar = tqdm(dataloader)
    bs=opt['Val_batchSize']
    count=0
    for packs in test_bar:
        features20x,features10x, label = packs

        count += 1

        if torch.cuda.is_available():
            features20x = features20x.cuda(gpuID[0])
            features10x = features10x.cuda(gpuID[0])
            label = label.cuda(gpuID[0])

        init_feature_his, init_feature_mark, His, Mark, Pub_His, Pub_Mark = Mine_model_init(features20x,features10x)


        hidden_states, encoded_IDH = Mine_model_IDH(init_feature_mark)
        hidden_states, encoded_1p19q = Mine_model_1p19q(hidden_states)
        encoded_CDKN = Mine_model_CDKN(hidden_states)
        results_dict,_,_,_,_,_,_,_, Mark_output  = Mine_model_Graph(encoded_IDH, encoded_1p19q, encoded_CDKN)
        pred_IDH_ori = results_dict['logits_IDH']
        pred_1p19q_ori = results_dict['logits_1p19q']
        pred_CDKN_ori = results_dict['logits_CDKN']

        hidden_states, encoded_His = Mine_model_His(init_feature_his)
        # encoded_Grade = Mine_model_Grade(hidden_states)
        results_dict,_,__,__,His_output = Mine_model_Cls(encoded_His)
        pred_His_ori = results_dict['logits_His']
        pred_His_2class_ori = results_dict['logits_His_2class']
        pred_Grade_ori = results_dict['logits_Grade']
        # pred_Grade_ori = results_dict['logits_Grade']
        his_mark = Mine_model_Task(His_output.float(), Mark_output.float())

        _, pred_IDH = torch.max(pred_IDH_ori.data, 1)
        pred_IDH = pred_IDH.tolist()
        gt_IDH = label[:, 0].tolist()

        _, pred_1p19q = torch.max(pred_1p19q_ori.data, 1)
        pred_1p19q = pred_1p19q.tolist()
        gt_1p19q = label[:, 1].tolist()

        _, predicted_CDKN = torch.max(pred_CDKN_ori.data, 1)
        valid_indices = label[:, 2] != -1 # 计算有效的标签索引
        total_CDKN = valid_indices.sum().item()# 计算有效标签的总数
        if total_CDKN == 0:
            correct_CDKN = 0
        else:
            # 获取有效的预测和标签
            pred_CDKN = predicted_CDKN[valid_indices].tolist()
            gt_CDKN = label[:, 2][valid_indices].tolist()

        _, pred_His_2class = torch.max(pred_His_2class_ori.data, 1)
        pred_His_2class = pred_His_2class.tolist()
        gt_His_2class = label[:, 6].tolist()

        _, pred_Task = torch.max(his_mark.data, 1)
        pred_Task = pred_Task.tolist()
        gt_Task = label[:, 5].tolist()


        gt_Diag = label[:, 5].tolist()
        pred_Diag=Diag_predict(pred_IDH_ori,pred_1p19q_ori,pred_CDKN_ori,pred_His_2class_ori)



        pred_IDH_ori_softmax = F.softmax(pred_IDH_ori, dim=1)
        pred_1p19q_ori_softmax = F.softmax(pred_1p19q_ori, dim=1)
        pred_CDKN_ori_softmax = F.softmax(pred_CDKN_ori, dim=1)
        pred_His_2class_ori_softmax = F.softmax(pred_His_2class_ori, dim=1)
        pred_Diag_ori = torch.zeros(1, 4).cuda(gpuID[0])
        pred_Diag_ori[0, 0] = pred_IDH_ori_softmax[0, 0]
        pred_Diag_ori[0, 1] = pred_IDH_ori_softmax[0, 1] * pred_1p19q_ori_softmax[0, 0] * (1 - pred_CDKN_ori_softmax[0, 0] * (1 - pred_His_2class_ori_softmax[0, 1]))
        pred_Diag_ori[0, 2] = pred_IDH_ori_softmax[0, 1] * pred_1p19q_ori_softmax[0, 0] * pred_CDKN_ori_softmax[0, 0] * (1 - pred_His_2class_ori_softmax[0, 1])
        pred_Diag_ori[0, 3] = pred_IDH_ori_softmax[0, 1] * pred_1p19q_ori_softmax[0, 1]
        # print(pred_Diag_ori[0, 0]+pred_Diag_ori[0, 1]+pred_Diag_ori[0, 2]+pred_Diag_ori[0, 3])
        for j in range(bs):
            ######   IDH
            label_all_IDH.append(gt_IDH[j])
            #pred_IDH_ori_softmax = F.softmax(pred_IDH_ori, dim=1)
            predicted_all_IDH.append(pred_IDH_ori_softmax.detach().cpu().numpy()[j][1])
            if gt_IDH[j] == 0 and pred_IDH[j] == 0:
                tn_IDH += 1
            if gt_IDH[j] == 0 and pred_IDH[j] == 1:
                fp_IDH += 1
            if gt_IDH[j] == 1 and pred_IDH[j] == 0:
                fn_IDH += 1
            if gt_IDH[j] == 1 and pred_IDH[j] == 1:
                tp_IDH += 1

            ######   1p19q
            label_all_1p19q.append(gt_1p19q[j])
            #pred_1p19q_ori_softmax = F.softmax(pred_1p19q_ori, dim=1)
            predicted_all_1p19q.append(pred_1p19q_ori_softmax.detach().cpu().numpy()[j][1])
            if gt_1p19q[j] == 0 and pred_1p19q[j] == 0:
                tn_1p19q += 1
            if gt_1p19q[j] == 0 and pred_1p19q[j] == 1:
                fp_1p19q += 1
            if gt_1p19q[j] == 1 and pred_1p19q[j] == 0:
                fn_1p19q += 1
            if gt_1p19q[j] == 1 and pred_1p19q[j] == 1:
                tp_1p19q += 1

            ######   CDKN
            label_all_CDKN.append(gt_CDKN[j])
            #pred_CDKN_ori_softmax = F.softmax(pred_CDKN_ori, dim=1)
            predicted_all_CDKN.append(pred_CDKN_ori_softmax.detach().cpu().numpy()[j][1])
            if gt_CDKN[j] == 0 and pred_CDKN[j] == 0:
                tn_CDKN += 1
            if gt_CDKN[j] == 0 and pred_CDKN[j] == 1:
                fp_CDKN += 1
            if gt_CDKN[j] == 1 and pred_CDKN[j] == 0:
                fn_CDKN += 1
            if gt_CDKN[j] == 1 and pred_CDKN[j] == 1:
                tp_CDKN += 1

            ######   His_2class
            label_all_His_2class.append(gt_His_2class[j])
            pred_His_2class_ori_softmax = F.softmax(pred_His_2class_ori, dim=1)
            predicted_all_His_2class.append(pred_His_2class_ori_softmax.detach().cpu().numpy()[j][1])

            if gt_His_2class[j] == 0 and pred_His_2class[j] == 0:
                tn_His_2class += 1
            if gt_His_2class[j] == 0 and pred_His_2class[j] == 1:
                fp_His_2class += 1
            if gt_His_2class[j] == 1 and pred_His_2class[j] == 0:
                fn_His_2class += 1
            if gt_His_2class[j] == 1 and pred_His_2class[j] == 1:
                tp_His_2class += 1
##############################################################################


            label_all_Task.append(gt_Task[j])
            pred_all_Task.append(pred_Task[j])
            his_mark = F.softmax(his_mark, dim=-1)
            predicted_all_Task.append(his_mark.detach().cpu().numpy()[0])
            if pred_Task[j] == 0:
                if gt_Task[j] == 0:
                    correct_Task+=1
                    GBM_metrics_Task['tp'] += 1
                else:
                    GBM_metrics_Task['fn'] += 1
            else:
                if not gt_Task[j] == 0:
                    GBM_metrics_Task['tn'] += 1
                else:
                    GBM_metrics_Task['fp'] += 1
            # G234_A
            if pred_Task[j] == 1:
                if gt_Task[j] == 1:
                    correct_Task+=1
                    G4_A_metrics_Task['tp'] += 1
                else:
                    G4_A_metrics_Task['fn'] += 1
            else:
                if not gt_Task[j] == 1:
                    G4_A_metrics_Task['tn'] += 1
                else:
                    G4_A_metrics_Task['fp'] += 1
            # G23_O
            if pred_Task[j] == 2:
                if gt_Task[j] == 2:
                    correct_Task+=1
                    G23_O_metrics_Task['tp'] += 1
                else:
                    G23_O_metrics_Task['fn'] += 1
            else:
                if not gt_Task[j] == 2:
                    G23_O_metrics_Task['tn'] += 1
                else:
                    G23_O_metrics_Task['fp'] += 1
            
            count_Task+=1


#################################################################
            label_all_Diag.append(gt_Diag[j])
            pred_all_Diag.append(pred_Diag[j])
            predicted_all_Diag.append(pred_Diag_ori.detach().cpu().numpy()[0])

            if gt_Diag[j] == 0:
                if pred_Diag[j] == 0:
                    GBM_metrics_Diag['tp'] += 1
                else:
                    GBM_metrics_Diag['fn'] += 1
            else:
                if not pred_Diag[j] == 0:
                    GBM_metrics_Diag['tn'] += 1
                else:
                    GBM_metrics_Diag['fp'] += 1
            # G4_A
            if gt_Diag[j] == 1:
                if pred_Diag[j] == 1:
                    G4_A_metrics_Diag['tp'] += 1
                else:
                    G4_A_metrics_Diag['fn'] += 1
            else:
                if not pred_Diag[j] == 1:
                    G4_A_metrics_Diag['tn'] += 1
                else:
                    G4_A_metrics_Diag['fp'] += 1
            # G23_A
            if gt_Diag[j] == 2:
                if pred_Diag[j] == 2:
                    G23_A_metrics_Diag['tp'] += 1
                else:
                    G23_A_metrics_Diag['fn'] += 1
            else:
                if not pred_Diag[j] == 2:
                    G23_A_metrics_Diag['tn'] += 1
                else:
                    G23_A_metrics_Diag['fp'] += 1
            # G23_O
            if gt_Diag[j] == 3:
                if pred_Diag[j] == 3:
                    G23_O_metrics_Diag['tp'] += 1
                else:
                    G23_O_metrics_Diag['fn'] += 1
            else:
                if not pred_Diag[j] == 3:
                    G23_O_metrics_Diag['tn'] += 1
                else:
                    G23_O_metrics_Diag['fp'] += 1

            count_Diag += 1
            correct_Diag += Diag_process(label, pred_IDH_ori, pred_1p19q_ori, pred_CDKN_ori, pred_His_2class_ori)


    ##########################################   IDH
    Acc_IDH = (tp_IDH + tn_IDH) / (tp_IDH + tn_IDH + fp_IDH + fn_IDH)
    Sen_IDH = (tp_IDH) / (tp_IDH + fn_IDH + 0.000001)  # recall
    Spec_IDH = (tn_IDH) / (tn_IDH + fp_IDH + 0.000001)
    precision_IDH = (tp_IDH) / (tp_IDH + fp_IDH + 0.000001)
    recall_IDH = Sen_IDH
    f1_score_IDH = (2 * precision_IDH * recall_IDH) / (precision_IDH + recall_IDH + 0.000001)
    AUC_IDH = metrics.roc_auc_score(y_true=np.array(label_all_IDH), y_score=np.array(predicted_all_IDH))

    list_IDH = (Acc_IDH, None, f1_score_IDH, Sen_IDH, Spec_IDH, AUC_IDH, precision_IDH)

    ##########################################   1p19q
    Acc_1p19q = (tp_1p19q + tn_1p19q) / (tp_1p19q + tn_1p19q + fp_1p19q + fn_1p19q)
    Sen_1p19q = (tp_1p19q) / (tp_1p19q + fn_1p19q + 0.000001)  # recall
    Spec_1p19q = (tn_1p19q) / (tn_1p19q + fp_1p19q + 0.000001)
    precision_1p19q = (tp_1p19q) / (tp_1p19q + fp_1p19q + 0.000001)
    recall_1p19q = Sen_1p19q
    f1_score_1p19q = (2 * precision_1p19q * recall_1p19q) / (precision_1p19q + recall_1p19q + 0.000001)
    AUC_1p19q = metrics.roc_auc_score(y_true=np.array(label_all_1p19q), y_score=np.array(predicted_all_1p19q))

    list_1p19q = (Acc_1p19q, None, f1_score_1p19q, Sen_1p19q, Spec_1p19q, AUC_1p19q, precision_1p19q)
    if task != 'Itest':
        ##########################################   CDKN
        Acc_CDKN = (tp_CDKN + tn_CDKN) / (tp_CDKN + tn_CDKN + fp_CDKN + fn_CDKN)
        Sen_CDKN = (tp_CDKN) / (tp_CDKN + fn_CDKN + 0.000001)  # recall
        Spec_CDKN = (tn_CDKN) / (tn_CDKN + fp_CDKN + 0.000001)
        precision_CDKN = (tp_CDKN) / (tp_CDKN + fp_CDKN + 0.000001)
        recall_CDKN = Sen_CDKN
        f1_score_CDKN = (2 * precision_CDKN * recall_CDKN) / (precision_CDKN + recall_CDKN + 0.000001)
        AUC_CDKN = metrics.roc_auc_score(y_true=np.array(label_all_CDKN), y_score=np.array(predicted_all_CDKN))

        list_CDKN = (Acc_CDKN, None, f1_score_CDKN, Sen_CDKN, Spec_CDKN, AUC_CDKN, precision_CDKN)
    
    else:
        list_CDKN = (0, None, 0, 0, 0, 0, 0)
    ##########################################   His_2class
    Acc_His_2class = (tp_His_2class + tn_His_2class) / (tp_His_2class + tn_His_2class + fp_His_2class + fn_His_2class)
    Sen_His_2class = (tp_His_2class) / (tp_His_2class + fn_His_2class + 0.000001)  # recall
    Spec_His_2class = (tn_His_2class) / (tn_His_2class + fp_His_2class + 0.000001)
    precision_His_2class = (tp_His_2class) / (tp_His_2class + fp_His_2class + 0.000001)
    recall_His_2class = Sen_His_2class
    f1_score_His_2class = (2 * precision_His_2class * recall_His_2class) / (
            precision_His_2class + recall_His_2class + 0.000001)
    if task !='Itest':
        AUC_His_2class = metrics.roc_auc_score(y_true=np.array(label_all_His_2class),
                                            y_score=np.array(predicted_all_His_2class))
    else:
        AUC_His_2class=0

    list_His_2class = (
    Acc_His_2class, None, f1_score_His_2class, Sen_His_2class, Spec_His_2class, AUC_His_2class, precision_His_2class)
    ################################################   Task

    Acc_Task = correct_Task / count_Task
    # Sensitivity
    G23_O_metrics_Task['sen'] = (G23_O_metrics_Task['tp']) / (
                G23_O_metrics_Task['tp'] + G23_O_metrics_Task['fn'] + 0.000001)
    G23_A_metrics_Task['sen'] = (G23_A_metrics_Task['tp']) / (
                G23_A_metrics_Task['tp'] + G23_A_metrics_Task['fn'] + 0.000001)
    G4_A_metrics_Task['sen'] = (G4_A_metrics_Task['tp']) / (
                G4_A_metrics_Task['tp'] + G4_A_metrics_Task['fn'] + 0.000001)
    GBM_metrics_Task['sen'] = (GBM_metrics_Task['tp']) / (GBM_metrics_Task['tp'] + GBM_metrics_Task['fn'] + 0.000001)
    all_metrics_Task['sen'] = (G23_O_metrics_Task['sen'] + G23_A_metrics_Task['sen'] + G4_A_metrics_Task['sen'] +
                               GBM_metrics_Task['sen']) / 4

    label_all_Task = list(label_all_Task)
    all_metrics_Task['sen_micro'] =  GBM_metrics_Task['sen'] * label_all_Task.count(0) / len(label_all_Task) + \
    G4_A_metrics_Task['sen'] * label_all_Task.count(1) / len(label_all_Task) + \
    G23_A_metrics_Task['sen'] * label_all_Task.count(2) / len(label_all_Task) + \
    G23_O_metrics_Task['sen'] * label_all_Task.count(3) / len(label_all_Task)
    #print("sen_micro ",all_metrics_Task['sen_micro'])

    # Specificity
    G23_O_metrics_Task['spec'] = (G23_O_metrics_Task['tn']) / (
                G23_O_metrics_Task['tn'] + G23_O_metrics_Task['fp'] + 0.000001)
    G23_A_metrics_Task['spec'] = (G23_A_metrics_Task['tn']) / (
                G23_A_metrics_Task['tn'] + G23_A_metrics_Task['fp'] + 0.000001)
    G4_A_metrics_Task['spec'] = (G4_A_metrics_Task['tn']) / (
                G4_A_metrics_Task['tn'] + G4_A_metrics_Task['fp'] + 0.000001)
    GBM_metrics_Task['spec'] = (GBM_metrics_Task['tn']) / (GBM_metrics_Task['tn'] + GBM_metrics_Task['fp'] + 0.000001)
    all_metrics_Task['spec'] = (G23_O_metrics_Task['spec'] + G23_A_metrics_Task['spec'] + G4_A_metrics_Task['spec'] +
                                GBM_metrics_Task['spec']) / 4
    
    all_metrics_Task['spec_micro'] =  GBM_metrics_Task['spec'] * label_all_Task.count(0) / len(label_all_Task) + \
    G4_A_metrics_Task['spec'] * label_all_Task.count(1) / len(label_all_Task) + \
    G23_A_metrics_Task['spec'] * label_all_Task.count(2) / len(label_all_Task) + \
    G23_O_metrics_Task['spec'] * label_all_Task.count(3) / len(label_all_Task)

    # Precision
    G23_O_metrics_Task['pre'] = (G23_O_metrics_Task['tp']) / (
                G23_O_metrics_Task['tp'] + G23_O_metrics_Task['fp'] + 0.000001)
    G23_A_metrics_Task['pre'] = (G23_A_metrics_Task['tp']) / (
                G23_A_metrics_Task['tp'] + G23_A_metrics_Task['fp'] + 0.000001)
    G4_A_metrics_Task['pre'] = (G4_A_metrics_Task['tp']) / (
                G4_A_metrics_Task['tp'] + G4_A_metrics_Task['fp'] + 0.000001)
    GBM_metrics_Task['pre'] = (GBM_metrics_Task['tp']) / (GBM_metrics_Task['tp'] + GBM_metrics_Task['fp'] + 0.000001)
    all_metrics_Task['pre'] = (G23_O_metrics_Task['pre'] + G23_A_metrics_Task['pre'] + G4_A_metrics_Task['pre'] +
                               GBM_metrics_Task['pre']) / 4
    all_metrics_Task['pre_micro'] =  GBM_metrics_Task['pre'] * label_all_Task.count(0) / len(label_all_Task) + \
    G4_A_metrics_Task['pre'] * label_all_Task.count(1) / len(label_all_Task) + \
    G23_A_metrics_Task['pre'] * label_all_Task.count(2) / len(label_all_Task) + \
    G23_O_metrics_Task['pre'] * label_all_Task.count(3) / len(label_all_Task)

    # Recall
    G23_O_metrics_Task['recall'] = (G23_O_metrics_Task['tp']) / (
                G23_O_metrics_Task['tp'] + G23_O_metrics_Task['fn'] + 0.000001)
    G23_A_metrics_Task['recall'] = (G23_A_metrics_Task['tp']) / (
                G23_A_metrics_Task['tp'] + G23_A_metrics_Task['fn'] + 0.000001)
    G4_A_metrics_Task['recall'] = (G4_A_metrics_Task['tp']) / (
                G4_A_metrics_Task['tp'] + G4_A_metrics_Task['fn'] + 0.000001)
    GBM_metrics_Task['recall'] = (GBM_metrics_Task['tp']) / (GBM_metrics_Task['tp'] + GBM_metrics_Task['fn'] + 0.000001)
    all_metrics_Task['recall'] = (G23_O_metrics_Task['recall'] + G23_A_metrics_Task['recall'] + G4_A_metrics_Task[
        'recall'] + GBM_metrics_Task['recall']) / 4
    all_metrics_Task['recall_micro'] =  GBM_metrics_Task['recall'] * label_all_Task.count(0) / len(label_all_Task) + \
    G4_A_metrics_Task['recall'] * label_all_Task.count(1) / len(label_all_Task) + \
    G23_A_metrics_Task['recall'] * label_all_Task.count(2) / len(label_all_Task) + \
    G23_O_metrics_Task['recall'] * label_all_Task.count(3) / len(label_all_Task)

    # F1 Score
    G23_O_metrics_Task['f1'] = 2 * (G23_O_metrics_Task['pre'] * G23_O_metrics_Task['recall']) / (
                G23_O_metrics_Task['pre'] + G23_O_metrics_Task['recall'] + 0.000001)
    G23_A_metrics_Task['f1'] = 2 * (G23_A_metrics_Task['pre'] * G23_A_metrics_Task['recall']) / (
                G23_A_metrics_Task['pre'] + G23_A_metrics_Task['recall'] + 0.000001)
    G4_A_metrics_Task['f1'] = 2 * (G4_A_metrics_Task['pre'] * G4_A_metrics_Task['recall']) / (
                G4_A_metrics_Task['pre'] + G4_A_metrics_Task['recall'] + 0.000001)
    GBM_metrics_Task['f1'] = 2 * (GBM_metrics_Task['pre'] * GBM_metrics_Task['recall']) / (
                GBM_metrics_Task['pre'] + GBM_metrics_Task['recall'] + 0.000001)
    # all_metrics_Task['f1'] = (G23_O_metrics_Task['f1'] + G23_A_metrics_Task['f1'] + G4_A_metrics_Task['f1'] +
    #                           GBM_metrics_Task['f1']) / 4
    # all_metrics_Task['f1_micro'] =  GBM_metrics_Task['f1'] * label_all_Task.count(0) / len(label_all_Task) + \
    # G4_A_metrics_Task['f1'] * label_all_Task.count(1) / len(label_all_Task) + \
    # G23_A_metrics_Task['f1'] * label_all_Task.count(2) / len(label_all_Task) + \
    # G23_O_metrics_Task['f1'] * label_all_Task.count(3) / len(label_all_Task)
    import sklearn
    
    f1 =sklearn.metrics.f1_score(label_all_Task, pred_all_Task, labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn')
    #print(f1)
    all_metrics_Task['f1_micro'] = f1
    if task != 'Itest':
        roc_auc = calculate_auc(predicted_all_Task, label_all_Task, num_classes=4)
        all_metrics_Task['AUC'] = roc_auc["macro"]
        all_metrics_Task['AUC_micro'] = roc_auc["micro"]
        GBM_metrics_Task['AUC'] = roc_auc[0]
        G4_A_metrics_Task['AUC'] = roc_auc[1]
        G23_A_metrics_Task['AUC'] = roc_auc[2]
        G23_O_metrics_Task['AUC'] = roc_auc[3]
    else:
        roc_auc = calculate_auc(predicted_all_Task, label_all_Task, num_classes=2)
        all_metrics_Task['AUC'] = roc_auc["macro"]
        all_metrics_Task['AUC_micro'] = roc_auc["micro"]
        GBM_metrics_Task['AUC'] = roc_auc[0]
        G4_A_metrics_Task['AUC'] = roc_auc[1]
        G23_A_metrics_Task['AUC'] = 0
        G23_O_metrics_Task['AUC'] = 0

    # Calculate weighted AUC
    if task != 'Itest':
        weights = [np.sum(np.array(label_all_Task) == i) for i in range(4)]
        total_samples = np.sum(weights)
        all_metrics_Task['AUC_weight'] = sum([roc_auc[i] * weights[i] for i in range(4)]) / (total_samples + 0.000001)
    else:
        weights = [np.sum(np.array(label_all_Task) == i) for i in range(2)]
        total_samples = np.sum(weights)
        all_metrics_Task['AUC_weight'] = sum([roc_auc[i] * weights[i] for i in range(2)]) / (total_samples + 0.000001)

    list_Task = (
    Acc_Task, 0, all_metrics_Task['f1_micro'], all_metrics_Task['sen_micro'], all_metrics_Task['spec_micro'],
    all_metrics_Task['AUC_micro'], all_metrics_Task['pre_micro'])

        ################################################   Diag

    Acc_Diag = correct_Diag / count_Diag
    # Sensitivity
    G23_O_metrics_Diag['sen'] = (G23_O_metrics_Diag['tp']) / (
                G23_O_metrics_Diag['tp'] + G23_O_metrics_Diag['fn'] + 0.000001)
    G23_A_metrics_Diag['sen'] = (G23_A_metrics_Diag['tp']) / (
                G23_A_metrics_Diag['tp'] + G23_A_metrics_Diag['fn'] + 0.000001)
    G4_A_metrics_Diag['sen'] = (G4_A_metrics_Diag['tp']) / (
                G4_A_metrics_Diag['tp'] + G4_A_metrics_Diag['fn'] + 0.000001)
    GBM_metrics_Diag['sen'] = (GBM_metrics_Diag['tp']) / (GBM_metrics_Diag['tp'] + GBM_metrics_Diag['fn'] + 0.000001)
    all_metrics_Diag['sen'] = (G23_O_metrics_Diag['sen'] + G23_A_metrics_Diag['sen'] + G4_A_metrics_Diag['sen'] +
                               GBM_metrics_Diag['sen']) / 4

    label_all_Diag = list(label_all_Diag)
    all_metrics_Diag['sen_micro'] =  GBM_metrics_Diag['sen'] * label_all_Diag.count(0) / len(label_all_Diag) + \
    G4_A_metrics_Diag['sen'] * label_all_Diag.count(1) / len(label_all_Diag) + \
    G23_A_metrics_Diag['sen'] * label_all_Diag.count(2) / len(label_all_Diag) + \
    G23_O_metrics_Diag['sen'] * label_all_Diag.count(3) / len(label_all_Diag)
    #print("sen_micro ",all_metrics_Diag['sen_micro'])

    # Specificity
    G23_O_metrics_Diag['spec'] = (G23_O_metrics_Diag['tn']) / (
                G23_O_metrics_Diag['tn'] + G23_O_metrics_Diag['fp'] + 0.000001)
    G23_A_metrics_Diag['spec'] = (G23_A_metrics_Diag['tn']) / (
                G23_A_metrics_Diag['tn'] + G23_A_metrics_Diag['fp'] + 0.000001)
    G4_A_metrics_Diag['spec'] = (G4_A_metrics_Diag['tn']) / (
                G4_A_metrics_Diag['tn'] + G4_A_metrics_Diag['fp'] + 0.000001)
    GBM_metrics_Diag['spec'] = (GBM_metrics_Diag['tn']) / (GBM_metrics_Diag['tn'] + GBM_metrics_Diag['fp'] + 0.000001)
    all_metrics_Diag['spec'] = (G23_O_metrics_Diag['spec'] + G23_A_metrics_Diag['spec'] + G4_A_metrics_Diag['spec'] +
                                GBM_metrics_Diag['spec']) / 4
    
    all_metrics_Diag['spec_micro'] =  GBM_metrics_Diag['spec'] * label_all_Diag.count(0) / len(label_all_Diag) + \
    G4_A_metrics_Diag['spec'] * label_all_Diag.count(1) / len(label_all_Diag) + \
    G23_A_metrics_Diag['spec'] * label_all_Diag.count(2) / len(label_all_Diag) + \
    G23_O_metrics_Diag['spec'] * label_all_Diag.count(3) / len(label_all_Diag)

    # Precision
    G23_O_metrics_Diag['pre'] = (G23_O_metrics_Diag['tp']) / (
                G23_O_metrics_Diag['tp'] + G23_O_metrics_Diag['fp'] + 0.000001)
    G23_A_metrics_Diag['pre'] = (G23_A_metrics_Diag['tp']) / (
                G23_A_metrics_Diag['tp'] + G23_A_metrics_Diag['fp'] + 0.000001)
    G4_A_metrics_Diag['pre'] = (G4_A_metrics_Diag['tp']) / (
                G4_A_metrics_Diag['tp'] + G4_A_metrics_Diag['fp'] + 0.000001)
    GBM_metrics_Diag['pre'] = (GBM_metrics_Diag['tp']) / (GBM_metrics_Diag['tp'] + GBM_metrics_Diag['fp'] + 0.000001)
    all_metrics_Diag['pre'] = (G23_O_metrics_Diag['pre'] + G23_A_metrics_Diag['pre'] + G4_A_metrics_Diag['pre'] +
                               GBM_metrics_Diag['pre']) / 4
    all_metrics_Diag['pre_micro'] =  GBM_metrics_Diag['pre'] * label_all_Diag.count(0) / len(label_all_Diag) + \
    G4_A_metrics_Diag['pre'] * label_all_Diag.count(1) / len(label_all_Diag) + \
    G23_A_metrics_Diag['pre'] * label_all_Diag.count(2) / len(label_all_Diag) + \
    G23_O_metrics_Diag['pre'] * label_all_Diag.count(3) / len(label_all_Diag)

    # Recall
    G23_O_metrics_Diag['recall'] = (G23_O_metrics_Diag['tp']) / (
                G23_O_metrics_Diag['tp'] + G23_O_metrics_Diag['fn'] + 0.000001)
    G23_A_metrics_Diag['recall'] = (G23_A_metrics_Diag['tp']) / (
                G23_A_metrics_Diag['tp'] + G23_A_metrics_Diag['fn'] + 0.000001)
    G4_A_metrics_Diag['recall'] = (G4_A_metrics_Diag['tp']) / (
                G4_A_metrics_Diag['tp'] + G4_A_metrics_Diag['fn'] + 0.000001)
    GBM_metrics_Diag['recall'] = (GBM_metrics_Diag['tp']) / (GBM_metrics_Diag['tp'] + GBM_metrics_Diag['fn'] + 0.000001)
    all_metrics_Diag['recall'] = (G23_O_metrics_Diag['recall'] + G23_A_metrics_Diag['recall'] + G4_A_metrics_Diag[
        'recall'] + GBM_metrics_Diag['recall']) / 4
    all_metrics_Diag['recall_micro'] =  GBM_metrics_Diag['recall'] * label_all_Diag.count(0) / len(label_all_Diag) + \
    G4_A_metrics_Diag['recall'] * label_all_Diag.count(1) / len(label_all_Diag) + \
    G23_A_metrics_Diag['recall'] * label_all_Diag.count(2) / len(label_all_Diag) + \
    G23_O_metrics_Diag['recall'] * label_all_Diag.count(3) / len(label_all_Diag)

    # F1 Score
    G23_O_metrics_Diag['f1'] = 2 * (G23_O_metrics_Diag['pre'] * G23_O_metrics_Diag['recall']) / (
                G23_O_metrics_Diag['pre'] + G23_O_metrics_Diag['recall'] + 0.000001)
    G23_A_metrics_Diag['f1'] = 2 * (G23_A_metrics_Diag['pre'] * G23_A_metrics_Diag['recall']) / (
                G23_A_metrics_Diag['pre'] + G23_A_metrics_Diag['recall'] + 0.000001)
    G4_A_metrics_Diag['f1'] = 2 * (G4_A_metrics_Diag['pre'] * G4_A_metrics_Diag['recall']) / (
                G4_A_metrics_Diag['pre'] + G4_A_metrics_Diag['recall'] + 0.000001)
    GBM_metrics_Diag['f1'] = 2 * (GBM_metrics_Diag['pre'] * GBM_metrics_Diag['recall']) / (
                GBM_metrics_Diag['pre'] + GBM_metrics_Diag['recall'] + 0.000001)
    # all_metrics_Diag['f1'] = (G23_O_metrics_Diag['f1'] + G23_A_metrics_Diag['f1'] + G4_A_metrics_Diag['f1'] +
    #                           GBM_metrics_Diag['f1']) / 4
    # all_metrics_Diag['f1_micro'] =  GBM_metrics_Diag['f1'] * label_all_Diag.count(0) / len(label_all_Diag) + \
    # G4_A_metrics_Diag['f1'] * label_all_Diag.count(1) / len(label_all_Diag) + \
    # G23_A_metrics_Diag['f1'] * label_all_Diag.count(2) / len(label_all_Diag) + \
    # G23_O_metrics_Diag['f1'] * label_all_Diag.count(3) / len(label_all_Diag)
    import sklearn
    
    f1 =sklearn.metrics.f1_score(label_all_Diag, pred_all_Diag, labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn')
    #print(f1)
    all_metrics_Diag['f1_micro'] = f1
    if task != 'Itest':
        roc_auc = calculate_auc(predicted_all_Diag, label_all_Diag, num_classes=4)
        all_metrics_Diag['AUC'] = roc_auc["macro"]
        all_metrics_Diag['AUC_micro'] = roc_auc["micro"]
        GBM_metrics_Diag['AUC'] = roc_auc[0]
        G4_A_metrics_Diag['AUC'] = roc_auc[1]
        G23_A_metrics_Diag['AUC'] = roc_auc[2]
        G23_O_metrics_Diag['AUC'] = roc_auc[3]
    else:
        roc_auc = calculate_auc(predicted_all_Diag, label_all_Diag, num_classes=2)
        all_metrics_Diag['AUC'] = roc_auc["macro"]
        all_metrics_Diag['AUC_micro'] = roc_auc["micro"]
        GBM_metrics_Diag['AUC'] = roc_auc[0]
        G4_A_metrics_Diag['AUC'] = roc_auc[1]
        G23_A_metrics_Diag['AUC'] = 0
        G23_O_metrics_Diag['AUC'] = 0

    # Calculate weighted AUC
    if task != 'Itest':
        weights = [np.sum(np.array(label_all_Diag) == i) for i in range(4)]
        total_samples = np.sum(weights)
        all_metrics_Diag['AUC_weight'] = sum([roc_auc[i] * weights[i] for i in range(4)]) / (total_samples + 0.000001)
    else:
        weights = [np.sum(np.array(label_all_Diag) == i) for i in range(2)]
        total_samples = np.sum(weights)
        all_metrics_Diag['AUC_weight'] = sum([roc_auc[i] * weights[i] for i in range(2)]) / (total_samples + 0.000001)

    list_Diag = (
    Acc_Diag, 0, all_metrics_Diag['f1_micro'], all_metrics_Diag['sen_micro'], all_metrics_Diag['spec_micro'],
    all_metrics_Diag['AUC_micro'], all_metrics_Diag['pre_micro'])

    return list_IDH, list_1p19q, list_CDKN, list_His_2class, list_Diag

def make_one_hot(data1):
    return (np.arange(6)==data1[:,None]).astype(np.int16)


def calculate_tp():
    a=1

import torch

def Diag_process(label,pred_IDH,pred_1p19q,pred_CDKN,pred_His_2class):
    correct_Diag=0

    label_Diag = label[:, 5].tolist()#(BS)


    _, pred_IDH = torch.max(pred_IDH.data, 1)
    _, pred_1p19q = torch.max(pred_1p19q.data, 1)
    _, pred_CDKN = torch.max(pred_CDKN.data, 1)
    #_, pred_His_2class = torch.max(pred_His_2class.data, 1)
    _, pred_His_2class = torch.max(pred_His_2class.data, 1)

    pred_IDH = pred_IDH.tolist()
    pred_1p19q = pred_1p19q.tolist()
    pred_CDKN = pred_CDKN.tolist()
    pred_His_2class = pred_His_2class.tolist()

    """
    label 2021={ 0:'G2_O', 1:'G3_O', 2:'G2_A', 3:'G3_A', 4:'G4_A', 5:'GBM'}
    label 2021={ 0:'GBM', 1:'G4_A', 2:'G2/3_A', 3:'G2/3_O'}
    """
    for j in range(label.detach().cpu().numpy().shape[0]):
        if pred_IDH[j]==0:
            if label_Diag[j]==0:
                correct_Diag+=1
        if pred_IDH[j] == 1 and pred_1p19q[j] == 1:
            if label_Diag[j]==3:
                correct_Diag += 1
        if pred_IDH[j] == 1 and pred_1p19q[j] == 0:
            if pred_CDKN[j]==1 or pred_His_2class[j]==1:
                if label_Diag[j]==1:
                    correct_Diag += 1
            else:
                if label_Diag[j] == 2:
                    correct_Diag += 1



    return correct_Diag

def Diag_predict(pred_IDH,pred_1p19q,pred_CDKN,pred_His_2class):
    _, pred_IDH = torch.max(pred_IDH.data, 1)
    _, pred_1p19q = torch.max(pred_1p19q.data, 1)
    _, pred_CDKN = torch.max(pred_CDKN.data, 1)
    # _, pred_His = torch.max(pred_His.data, 1)
    _, pred_His_2class = torch.max(pred_His_2class.data, 1)

    pred_IDH = pred_IDH.tolist()
    pred_1p19q = pred_1p19q.tolist()
    pred_CDKN = pred_CDKN.tolist()
    # pred_His = pred_His.tolist()
    pred_His_2class = pred_His_2class.tolist()

    pred_Diag=[]

    for j in range(len(pred_IDH)):
        if pred_IDH[j]==0:
            pred_Diag.append(0)
        if pred_IDH[j] == 1 and pred_1p19q[j] == 1:
            pred_Diag.append(3)
        if pred_IDH[j] == 1 and pred_1p19q[j] == 0:
            if pred_CDKN[j]==1 or pred_His_2class[j]==1:
                pred_Diag.append(1)
            else :
                pred_Diag.append(2)
    return pred_Diag


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',
                 ignore_lb=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_lb = ignore_lb

    def forward(self, logits, label, transform='softmax'):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        ignore = label.data.cpu() == self.ignore_lb
        n_valid = (ignore == 0).sum()
        label[ignore] = 0

        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        mask = torch.ones_like(logits)
        mask[[a, torch.arange(mask.size(1)), *b]] = 0

        # compute loss
        if transform == 'softmax':
            probs = F.softmax(logits, dim=1)
        else:
            probs = torch.sigmoid(logits)
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        pt = torch.where(lb_one_hot == 1, probs, 1 - probs)
        alpha = self.alpha * lb_one_hot + (1 - self.alpha) * (1 - lb_one_hot)
        loss = -alpha * ((1 - pt) ** self.gamma) * torch.log(pt + 1e-12)
        loss[mask == 0] = 0
        if self.reduction == 'mean':
            loss = loss.sum(dim=1).sum() / n_valid
        return loss
# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2, alpha=0.75, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average
#
#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)
#
#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())
#
#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)
#
#         loss = -1 * (1-pt)**self.gamma * logpt
#         k=loss.mean()
#         return k
from model import *

def get_model(opt,if_end2end=False):
    gpuID = opt['gpus']
    Mine_model_init = Mine_init(opt,if_end2end=if_end2end).cuda(gpuID[0])
    Mine_model_IDH = Mine_IDH(opt).cuda(gpuID[0])
    Mine_model_1p19q = Mine_1p19q(opt).cuda(gpuID[0])
    Mine_model_CDKN = Mine_CDKN(opt).cuda(gpuID[0])
    Mine_model_Graph = Label_correlation_Graph(opt).cuda(gpuID[0])
    Mine_model_His = Mine_His(opt).cuda(gpuID[0])
    Mine_model_Cls = Cls_His_Grade(opt).cuda(gpuID[0])
    Mine_model_Task = Mine_Task(opt).cuda(gpuID[0])

    init_weights(Mine_model_init, init_type='xavier', init_gain=1)
    init_weights(Mine_model_IDH, init_type='xavier', init_gain=1)
    init_weights(Mine_model_1p19q, init_type='xavier', init_gain=1)
    init_weights(Mine_model_CDKN, init_type='xavier', init_gain=1)
    init_weights(Mine_model_His, init_type='xavier', init_gain=1)
    init_weights(Mine_model_Task, init_type='xavier', init_gain=1)  

    # Mine_model_init.to(device)
    # Mine_model_IDH.to(device)
    # Mine_model_1p19q.to(device)
    # Mine_model_CDKN.to(device)
    # Mine_model_Graph.to(device)
    # Mine_model_His.to(device)
    # Mine_model_Cls.to(device)
    Mine_model_init = torch.nn.DataParallel(Mine_model_init, device_ids=gpuID)
    Mine_model_IDH = torch.nn.DataParallel(Mine_model_IDH, device_ids=gpuID)
    Mine_model_1p19q = torch.nn.DataParallel(Mine_model_1p19q, device_ids=gpuID)
    Mine_model_CDKN = torch.nn.DataParallel(Mine_model_CDKN, device_ids=gpuID)
    Mine_model_Graph = torch.nn.DataParallel(Mine_model_Graph, device_ids=gpuID)
    Mine_model_His = torch.nn.DataParallel(Mine_model_His, device_ids=gpuID)
    Mine_model_Cls = torch.nn.DataParallel(Mine_model_Cls, device_ids=gpuID)
    Mine_model_Task = torch.nn.DataParallel(Mine_model_Task, device_ids=gpuID)

    opt_init = torch.optim.Adam(Mine_model_init.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_IDH = torch.optim.Adam(Mine_model_IDH.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_1p19q = torch.optim.Adam(Mine_model_1p19q.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_CDKN = torch.optim.Adam(Mine_model_CDKN.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_Graph = torch.optim.Adam(Mine_model_Graph.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_His = torch.optim.Adam(Mine_model_His.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    # opt_Grade = torch.optim.Adam(Mine_model_Grade.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_Cls = torch.optim.Adam(Mine_model_Cls.parameters(), opt['Network']['lr'], weight_decay=0.00001)
    opt_Task = torch.optim.Adam(Mine_model_Task.parameters(), opt['Network']['lr'], weight_decay=0.00001)

    ###############  fp16 #######################
    if opt['fp16']:
        from apex import amp
        Mine_model_init, opt_init = amp.initialize(models=Mine_model_init,optimizers=opt_init,opt_level="O1")
        Mine_model_IDH, opt_IDH = amp.initialize(models=Mine_model_IDH, optimizers=opt_IDH,opt_level="O1")
        Mine_model_1p19q, opt_1p19q = amp.initialize(models=Mine_model_1p19q, optimizers=opt_1p19q,opt_level="O1")
        Mine_model_CDKN, opt_CDKN = amp.initialize(models=Mine_model_CDKN, optimizers=opt_CDKN,opt_level="O1")
        Mine_model_Graph, opt_Graph = amp.initialize(models=Mine_model_Graph, optimizers=opt_Graph, opt_level="O1")
        Mine_model_His, opt_His = amp.initialize(models=Mine_model_His, optimizers=opt_His,opt_level="O1")
        # Mine_model_Grade, opt_Grade = amp.initialize(models=Mine_model_Grade, optimizers=opt_Grade,opt_level="O1")
        Mine_model_Cls, opt_Cls = amp.initialize(models=Mine_model_Cls, optimizers=opt_Cls, opt_level="O1")
        Mine_model_Task, opt_Task = amp.initialize(models=Mine_model_Task, optimizers=opt_Task, opt_level="O1")


    return  Mine_model_init,Mine_model_IDH,Mine_model_1p19q,Mine_model_CDKN,Mine_model_Graph,Mine_model_His,Mine_model_Cls,Mine_model_Task\
        ,opt_init,opt_IDH,opt_1p19q,opt_CDKN,opt_Graph,opt_His,opt_Cls,opt_Task



if __name__ == "__main__":

    loss=FocalLoss()
    # pred=torch.from_numpy(np.asarray([[-0.2,-0.5]])).float()
    # label=torch.from_numpy(np.asarray([0])).long()
    # my_loss=loss(pred,label)
