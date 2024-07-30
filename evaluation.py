import os
os.environ['QTQPAPLATFORM']='offscreen'
import torch
import torchvision
# from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image
#import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, savename, title='Confusion Matrix', classes=['G2_O', 'G3_O', 'G2_A', 'G3_A', 'G4_A', 'GBM']):
    plt.figure(figsize=(11, 11), dpi=100)
    np.set_printoptions(precision=2)  # 输出小数点的个数0:'G2_O', 1:'G3_O', 2:'G2_A', 3:'G3_A', 4:'G4_A', 5:'GBM'

    # 在混淆矩阵中每格的概率值
    # classes = ['P_MN', 'S_MN', 'P_IgAN', 'S_IgAN', 'LN', 'DN', 'ANCA', 'MPGN']
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    thresh = cm.max() / 2.


    for x_val, y_val in zip(x.flatten(), y.flatten()):
      c = cm[y_val][x_val]
      if c > 0.001:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='white' if cm[x_val, y_val] > thresh else 'black',
                 fontsize=20, va='center', ha='center')

    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=36, pad=20)

    # plt.matshow(cm, cmap=plt.cm.Blues)  # 背景颜色

    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    # plt.xticks(xlocations, classes, rotation=90)
    # plt.yticks(xlocations, classes)
    plt.xticks(xlocations, classes, size=16)
    plt.yticks(xlocations, classes, size=16)
    plt.ylabel('Actual label', fontsize=22, labelpad=12)
    plt.xlabel('Predict label', fontsize=22, labelpad=12)

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    # plt.show()
    plt.close('all')