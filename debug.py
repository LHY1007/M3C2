import os
import numpy as np
from skimage import io
import math
####### find out backward propogation difference of img and fea in TransMIL

# seed=100
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)
# parser = argparse.ArgumentParser()
# parser.add_argument('--opt', type=str, default='config/miccai.yml')
# args = parser.parse_args()
# with open(args.opt) as f:
#     opt = yaml.load(f, Loader=SafeLoader)
# gpuID = opt['gpus']
# TransMIL = model.TransMIL(opt)
# model.init_weights(TransMIL, init_type='xavier', init_gain=1)
# assert opt['name'].split('_')[0]=='TransMIL'
# device = torch.device('cuda:{}'.format(gpuID[0])) if gpuID else torch.device('cpu')
# if opt['name'].split('_')[2] == 'img':
#     Res_pretrain= net.Res50_pretrain()
#     Res_pretrain.to(device)
#
# TransMIL.to(device)
# TransMIL_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, TransMIL.parameters()), 0.01, weight_decay=0.00001)
# trainDataset = dataset.Our_Dataset(phase='Train',opt=opt)
#
# img,label,img_path=trainDataset._getitem__(0)
#
# label = torch.from_numpy(np.asarray([label.detach().numpy()])).long()
#
# if torch.cuda.is_available():
#     img = img.cuda(gpuID[0])
#     label = label.cuda(gpuID[0])
# TransMIL.train()
# TransMIL.zero_grad()
# print(img_path)
# if opt['name'].split('_')[2] == 'img':
#     Res_pretrain.train()
#     img=Res_pretrain(img)
#
# results_dict = TransMIL(img)
# pred=results_dict['logits']
# loss_all = TransMIL.calculateLoss(pred,label)
# loss_all.backward()
# TransMIL_opt.step()

# coords_all = h5py.File('temp/TCGA-DU-6404-01Z-00-DX1.93c15688-f5b2-40bc-85eb-ff2661e16d4e.h5')['coords'][:]
"""
Use only one object per slide
"""
excel_label_wsi = pd.read_excel('./merge_who.xlsx',sheet_name='wsi_level',header=0)
excel_wsi =excel_label_wsi.values
WSI_used_names=list(excel_wsi[:,1])
np.random.seed(1)
random.seed(1)
random.shuffle(WSI_used_names)

root_reading_list=r'/mnt/disk10T/fyb/wxf_data/TCGA/brain/reading_list_extract224/'
files_reading_list=os.listdir(root_reading_list)

for i in range(excel_wsi.shape[0]):

    # print(WSI_used_names[i])
    reading_list = np.load(root_reading_list + WSI_used_names[i]+'.npy')
    # reading_list = np.load(root_reading_list + 'TCGA-E1-A7Z4-01Z-00-DX2' + '.npy')
    Num_cluster = 0
    point_num = reading_list.shape[0]
    points_center_corrd = reading_list

    FLAT_w = int(points_center_corrd[0].split('_')[0])
    FLAT_h = int(points_center_corrd[0].split('_')[1])
    claster_corrds = {'0': []}
    for k in range(point_num):
        w_point = int(points_center_corrd[k].split('_')[0])
        h_point = int(points_center_corrd[k].split('_')[1])
        if w_point == FLAT_w:
            claster_corrds[str(Num_cluster)].append([w_point, h_point])
            continue
        # FLAT_w=w_point
        if w_point > FLAT_w and (w_point-FLAT_w)<=6:

            FLAT_w = w_point
            claster_corrds[str(Num_cluster)].append([w_point, h_point])
            continue
        if w_point < FLAT_w or  (w_point-FLAT_w)>6:
            FLAT_w = w_point
            Num_cluster += 1
            claster_corrds[str(Num_cluster)] = []
            claster_corrds[str(Num_cluster)].append([w_point, h_point])
    del_value=[]
    for key,value in enumerate(claster_corrds):
        claster_corrds[value]=np.asarray(claster_corrds[value])
        if claster_corrds[value].shape[0] < 150:
            del_value.append(value)
    for nn in range(len(del_value)):
        del claster_corrds[del_value[nn]]
    patch_length=[]
    value_name=[]
    for key, value in enumerate(claster_corrds):
        patch_length.append(claster_corrds[value].shape[0])
        value_name.append(value)
    save_dict=[]
    sort_0=np.argsort(np.asarray(patch_length))
    for nn in range(len(value_name)):
        save_dict.append(claster_corrds[value_name[sort_0[len(value_name)-nn-1]]])

    np.save('/mnt/disk10T/fyb/wxf_data/TCGA/brain/read_details/' + WSI_used_names[i]+ '.npy', save_dict)
    print(i)
    a=1
a=[[[1,2],[11,21]],[[12,23],[14,24]],[[15,25],[16,26]],[[17,27],[18,28]]]
a=np.asarray(a)
a=a.reshape(2,2,2,2)
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from torchvision.transforms import Compose
import transform.transforms_group as our_transform
def train_transform(degree=180):
    return Compose([
        our_transform.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),
    ])
My_transform=train_transform()
root=r'D:\PhD\Project_WSI\data\a/'
files=os.listdir(root)
imgs=[]
max_num=1200
read_details=np.load(r'D:\PhD\Project_WSI\data\TCGA-HT-8104-01A-01-TS1.npy',allow_pickle=True)[0]
for i in range(len(files)):
    imgs.append(io.imread(root + '/' + str(read_details[i][0]) + '_' + str(read_details[i][1]) + '.jpg'))
imgs = np.asarray(imgs)#(num_patches,224,224,3)



imgs=imgs.reshape(-1,224,3) #(num_patches*224,224,3)
imgs = Image.fromarray(imgs.astype('uint8')).convert('RGB')
imgs=My_transform(imgs)
imgs=np.array(imgs)#(num_patches*224,224,3)
imgs=imgs.reshape(-1,224,224,3)#(num_patches,224,224,3)

N_adj=int(math.sqrt(len(files)))
imgs=imgs[0:N_adj*N_adj]
imgs=imgs.reshape(N_adj,N_adj,224,224,3) #(Na,Na,224,224,3)
imgs=np.transpose(imgs,(0,2,1,3,4)) #(Na,224,Na,224,3)
imgs=imgs.reshape(N_adj*224,N_adj*224,3)#(Na*224,Na*224,3)

plt.imshow(imgs)
plt.show()









