import numpy as np
import h5py
import argparse, time

import gc
# root=r'/mnt/disk10T/fuyibing/wxf_data/TCGA/brain/npy/'
root=r'/mnt/disk10T/fyb/wxf_data/TCGA/brain/npy/'

import SimpleITK as sitk

#####itk
img=np.zeros(shape=(1200,3,224,224), dtype=np.uint8)
out = sitk.GetImageFromArray(img)
sitk.WriteImage(out, root+'simpleitk_save.nii.gz')


#####npy
img=np.zeros(shape=(1200,3,224,224), dtype=np.uint8)
np.save(root+'img.npy', img)

#####h5
imgData=np.zeros(shape=(1200,3,224,224), dtype=np.uint8)
with h5py.File(root+'test.h5','w') as f:
    f['data'] = imgData

k=h5py.File(root+'test.h5')['data'][:]
a=1