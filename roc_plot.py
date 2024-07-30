import os
import numpy as np
import torch
import torch.nn as nn
import platform
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from itertools import cycle
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
import scipy
import os
from sklearn.metrics import roc_auc_score
import pandas as pd

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

def ROC(df):
    n_classes = 9
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    l = np.zeros((len(df), n_classes))
    p = np.zeros((len(df), n_classes))
    for i in range(len(df)):
        l[i, df.iloc[i, 0]] = 1

    for i in range(n_classes):
        label = df['label'].tolist()
        score = df['score' + str(i)].tolist()
        p[:, i] = score
        fpr[i], tpr[i], _ = roc_curve(label, score, pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    weighted_auc = roc_auc_score(l, p, average='macro')

    return fpr['macro'],tpr['macro'], weighted_auc


if __name__ == "__main__":

    name='1p19q'
    print('\033[1;35;0m字体变色，但无背景色 \033[0m')
    np.random.seed(2)
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    COLOR_LIST = [ [139 / 255, 20 / 255, 8 / 255],[188 / 255, 189 / 255, 34 / 255],
                  [52 / 255, 193 / 255, 52 / 255],
                  [150 / 255, 150 / 255, 190 / 255], [139 / 255, 101 / 255, 8 / 255],
                  [68 / 255, 114 / 255, 236 / 255],
                  [100 / 255, 114 / 255, 196 / 255], [214 / 255 + 0.1, 39 / 255 + 0.2, 40 / 255 + 0.2],
                  [52 / 255, 163 / 255, 152 / 255]]

    LINE_WIDTH_LIST = [3, 3, 3, 3, 3, 3, 3,3,3]


    i = 0
    plt.figure(figsize=[10.5, 10])


    LABEL_LIST =['Ours', 'CLAM', 'TransMIL', 'ResNet-18',
                 'DenseNet-121', 'VGG-16', 'W/O Graph', 'W/O LC loss',
                 'W/O DCC']
    EXCEL_LIST = ['plot/Mine_'+name+'.xlsx', 'plot/CLAM_'+name+'_fea.xlsx', 'plot/TransMIL_'+name+'_fea.xlsx',
                  'plot/Basic_'+name+'_img_res.xlsx', 'plot/Basic_'+name+'_img_dense.xlsx', 'plot/Basic_'+name+'_img_VGG.xlsx',
                  'plot/Mine_Graph_'+name+'.xlsx', 'plot/Mine_Graphloss_'+name+'.xlsx', 'plot/Mine_DCC_'+name+'.xlsx', ]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()


    for i in range(9):
        df = pd.read_excel(EXCEL_LIST[i])
        label = df['label'].tolist()
        score = df['score'].tolist()
        fpr[i], tpr[i], _ = roc_curve(label, score)
        # roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i],
             # label=LABEL_LIST[i],
             linewidth= LINE_WIDTH_LIST[i] , color=np.array(COLOR_LIST[i]))
        plt.plot(1-df['spec'].tolist()[0], df['sen'].tolist()[0], marker="o", markersize=15, markerfacecolor=np.array(COLOR_LIST[i]), markeredgecolor=np.array(COLOR_LIST[i]))
        print(df['sen'].tolist()[0])
        print(df['spec'].tolist()[0])
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.grid(color=[0.85, 0.85, 0.85])

    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=24, weight='semibold')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=24, weight='semibold')

    font_axis_name = {'fontsize': 34, 'weight': 'bold'}
    plt.xlabel('1-Specificity',font_axis_name)
    plt.ylabel('Sensitivity',font_axis_name)
    plt.xlim((0, 0.5))
    plt.ylim((0.5, 1))
    plt.legend(framealpha=1, fontsize=30, loc='lower right')
    plt.tight_layout()

    plt.savefig("plot/"+name+".tiff")

    plt.show()
