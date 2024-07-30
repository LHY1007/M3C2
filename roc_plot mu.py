import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, roc_auc_score
import torch
import ast
import re
import numpy as np
import pandas as pd
import re
from scipy.special import softmax
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.special import expit  # Sigmoid function
def ROC(df):
    n_classes = 4
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    l = np.zeros((len(df), n_classes))
    p = np.zeros((len(df), n_classes))
    for i in range(len(df)):
        l[i, df.iloc[i, 0]] = 1
    label = df['label'].tolist()
    scores = df['score'].tolist()
    converted_data = [list(map(float, re.findall(r'-?\d+\.\d+', item))) for item in scores]
    score = np.array(converted_data)

    for i in range(n_classes):

        p[:, i] = score[:, i]
        fpr[i], tpr[i], _ = roc_curve(label, score[:, i], pos_label=i)
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
    name='IN_Diag'
    # print('\033[1;35;0m字体变色，但无背景色 \033[0m')
    np.random.seed(2)
    plt.rcParams["font.family"] = "ARIAL"
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    COLOR_LIST = [[139 / 255, 20 / 255, 8 / 255],
                [188 / 255, 189 / 255, 34 / 255],
                [52 / 255, 193 / 255, 52 / 255],
                [150 / 255, 150 / 255, 190 / 255], 
                [139 / 255, 101 / 255, 8 / 255],
                [68 / 255, 114 / 255, 236 / 255],
                [100 / 255, 114 / 255, 196 / 255], 
                [214 / 255 + 0.1, 39 / 255 + 0.2, 40 / 255 + 0.2],
                [52 / 255, 163 / 255, 152 / 255],
                [139 / 255 * 1.1, 20 / 255 * 1.1, 8 / 255 * 1.1],
                [188 / 255 * 0.9, 189 / 255 * 0.9, 34 / 255 * 0.9],
                [52 / 255 * 1.1, 193 / 255 * 1.1, 52 / 255 * 1.1],
                [150 / 255 * 0.9, 150 / 255 * 0.9, 190 / 255 * 0.9],
                [139 / 255 * 1.1, 101 / 255 * 1.1, 8 / 255 * 1.1]]

    LINE_WIDTH_LIST = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

    i = 0
    plt.figure(figsize=[10.5, 10])

    LABEL_LIST =['Ours', 'Wang et al.','ABMIL', 'TransMIL', 'CLAM',
                 'Charm', 'Deepglioma', 'MCAT', 'CMTA',
                 'AlexNet','DenseNet','InceptionNet','ResNet-50','VGG-18']
    EXCEL_LIST = ['plot/Mine_'+name+'.xlsx', 'plot/MICCAI_'+name+'.xlsx','plot/ABMIL_'+name+'.xlsx','plot/TransMIL_'+name+'.xlsx',\
                  'plot/Charm_'+name+'.xlsx','plot/Deepglioma_'+name+'.xlsx','plot/MCAT_'+name+'.xlsx','plot/CMTA_'+name+'.xlsx',\
                  'plot/CLAM_'+name+'.xlsx','plot/AlexNet_'+name+'.xlsx','plot/DenseNet_'+name+'.xlsx','plot/InceptionNet_'+name+'.xlsx',\
                    'plot/ResNet-50_'+name+'.xlsx','plot/VGG-18'+name+'.xlsx']

    LABEL_LIST = ['Ours']
    EXCEL_LIST = ['plot/Mine_'+name+'.xlsx']


    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    num = len(LABEL_LIST)


    for i in range(num):
        df = pd.read_excel(EXCEL_LIST[i])
        label = df['label'].tolist()
        score = df['score'].tolist()

        fpr[i], tpr[i], _ = ROC(df)
        plt.plot(fpr[i], tpr[i],
                 label=LABEL_LIST[i],  # 添加 label 参数
                 linewidth= LINE_WIDTH_LIST[i] , color=np.array(COLOR_LIST[i]))
        plt.plot(1-df['spec'].tolist()[0], df['sen'].tolist()[0], marker="o", markersize=15, markerfacecolor=np.array(COLOR_LIST[i]), markeredgecolor=np.array(COLOR_LIST[i]))
        print(df['sen'].tolist()[0])
        print(df['spec'].tolist()[0])
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.grid(color=[0.85, 0.85, 0.85])

    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=24, weight='semibold')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=24, weight='semibold')

    font_axis_name = {'fontsize': 34, 'weight': 'bold'}
    plt.xlabel('1-Specificity', font_axis_name)
    plt.ylabel('Sensitivity', font_axis_name)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.legend(framealpha=1, fontsize=30, loc='lower right')
    plt.tight_layout()

    plt.savefig("plot/"+name+".tiff")
    plt.show()
