import os
import torch
import torchvision
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image
from evaluation import *
import cv2



class Saver():
  def __init__(self, opt):
    self.logDir = opt['logDir']
    self.n_ep_save = opt['n_ep_save']
    self.writer = SummaryWriter(logdir=self.logDir)

  def write_scalars(self, ep, lossdict):
    # Todo Save images
    for loss_key, loss_value in lossdict.items():
        self.writer.add_scalar(loss_key, loss_value, ep)


  def write_maps(self, ep, map_dict):
    for name,map in map_dict.items():
      if len(map.shape)==2:
        map = map[np.newaxis,...]
      if map.shape[0] == 1:
        map = np.concatenate((map, map, map), axis=0)
      self.writer.add_image('map/'+name, map, ep)


  def write_log(self, ep, lossdict, Name):
    logpath = os.path.join(self.logDir, Name + '.log')
    title = 'epochs,'
    vals = '%d,'%(ep)
    for loss_key, loss_value in lossdict.items():
      title = title + loss_key + ','
      vals = vals + '%4f,'% (loss_value)
    title = title[:-1] + '\n'
    vals = vals[:-1] + '\n'
    if ep==self.n_ep_save-1:
      saveFile = open(logpath, "w")
      saveFile.write(title)
      saveFile.write(vals)
    else:
      saveFile = open(logpath, "a")
      saveFile.write(vals)
    saveFile.close()




  def write_imagegroup(self, ep, images, basename, key):
    # images: tensor Bx3xHxW or Bx1xHxW or BxHxW
    if len(images.shape) == 3:
      images = torch.unsqueeze(images, 1)
      images = torch.cat([images, images, images], 1)
    elif images.shape[1] == 1:
      images = torch.cat([images, images, images], 1)
    image_dis = torchvision.utils.make_grid(images, nrow=7)
    self.writer.add_image('map/' + key, image_dis, ep)
    image_dis2 = image_dis

    ndarr = image_dis2.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    savename = os.path.join(self.logDir, key + '_' + basename + '_' +str(ep) + '.png')
    if key == 'SAmap':
      ndarr = cv2.applyColorMap(ndarr, cv2.COLORMAP_JET)
    cv2.imwrite(savename, ndarr)

    return ndarr

  def write_cm_maps(self, ep, cm, class_list, savename='cm.png'):
    savename = os.path.join(self.logDir, savename)
    plot_confusion_matrix(cm, savename, title='Confusion Matrix',
                          classes=class_list)
    cmimg = cv2.imread(savename)
    cmimg = np.transpose(cmimg, (2, 0, 1))
    self.writer.add_image('map/cm', cmimg, ep)

