
# root=r'D:\PhD\Project_WSI\Others_code\GBM_WSSM-master\OneDrive-2023-03-15/'
# dirs=os.listdir(root)
# save_root=r'D:\PhD\Project_WSI\Others_code\GBM_WSSM-master/GBM_TCGA_SEG/'
# for i in range(len(dirs)):
#     files=os.listdir(root+dirs[i]+'/')
#     for j in range(len(files)):
#         shutil.copy(root+dirs[i]+'/'+files[j],save_root+files[j])
import os
# from openslide import OpenSlide
import h5py
import numpy as np
import cv2
from skimage import io,transform
root=r'D:\PhD\Project_WSI\Others_code\GBM_WSSM-master\GBM_Test_Images/'
imgs=os.listdir(root)
for i in range(len(imgs)):
    img_temp = io.imread(root+imgs[i])
    img_temp = cv2.resize(img_temp, (1024, 1024))
    io.imsave(root+imgs[i][0:-4]+'resize.jpg',img_temp)


# def get_filename(root_path,file_path):
#     return_file = []
#     files=os.listdir(root_path+file_path)
#     for i in range(len(files)):
#         get_path = os.path.join(root_path, file_path,files[i])
#         if get_path.endswith('.svs') or get_path.endswith('.partial'):
#             return_file.append(get_path)
#     return return_file
#
# root=r'/mnt/disk10T/fuyibing/wxf_data/TCGA/brain/ori_wsi/ffpe_GBM/'
# slide_path = []
# WSI_path_list = os.listdir(root)
# for i in range(len(WSI_path_list)):
#     get_file = get_filename(root, WSI_path_list[i])
#     slide_path.append(get_file[0])
#
#
# for i in range(len(slide_path)):
#     wsi_obj = OpenSlide(slide_path[i])
#
#     pro = dict(wsi_obj.properties)
#     MPP=np.float(pro['aperio.MPP'])
#     wsi_w=wsi_obj.dimensions[0]
#     wsi_h=wsi_obj.dimensions[1]
#
#
#
#     with h5py.File('vis_results/set0/'+slide_path[i].split('/')[-1].split('.')[0]+'.h5', 'w') as f:
#         f['wsi_w'] = wsi_w
#         f['wsi_h'] = wsi_h
#         f['MPP'] = MPP
#     print(i)







