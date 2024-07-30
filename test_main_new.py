# from apex import amp
import numpy as np

from utils import *
import dataset_mine
from net import init_weights,get_scheduler,WarmupCosineSchedule
from matplotlib import pyplot as plt
import matplotlib as mpl
import cmaps
import h5py
def train(opt):
    root=r'/home/hanyu/LHY/miccai7.22/best_model/Best3_0720-0755-0003.pth'

    opt['gpus']=[6,7]
    gpuID = opt['gpus']
    valDataset = dataset_mine.Our_Dataset_vis(phase='Test', opt=opt)
    valLoader = DataLoader(valDataset, batch_size=opt['Val_batchSize'],
                             num_workers=opt['nThreads'] if (sysstr == "Linux") else 1, shuffle=False)

    ############## initialize #######################

    last_ep = 0
    total_it = 0
    saver = Saver(opt)
    print('%d epochs and %d iterations has been trained' % (last_ep, total_it))
    alleps = opt['n_ep'] - last_ep
    curep=0
    if 1:
        ## IDH
        model_init = Mine_init(opt).cuda(gpuID[0])
        model_IDH = Mine_IDH(opt).cuda(gpuID[0])
        model_1p19q = Mine_1p19q(opt).cuda(gpuID[0])
        model_CDKN = Mine_CDKN(opt).cuda(gpuID[0])
        model_Graph = Label_correlation_Graph(opt).cuda(gpuID[0])
        model_His = Mine_His(opt).cuda(gpuID[0])
        model_Cls = Cls_His_Grade(opt).cuda(gpuID[0])
        model_Task = Mine_Task(opt).cuda(gpuID[0])

        model_init = torch.nn.DataParallel(model_init, device_ids=gpuID)
        model_IDH = torch.nn.DataParallel(model_IDH, device_ids=gpuID)
        model_1p19q = torch.nn.DataParallel(model_1p19q, device_ids=gpuID)
        model_CDKN = torch.nn.DataParallel(model_CDKN, device_ids=gpuID)
        model_Graph = torch.nn.DataParallel(model_Graph, device_ids=gpuID)
        model_His = torch.nn.DataParallel(model_His, device_ids=gpuID)
        model_Cls = torch.nn.DataParallel(model_Cls, device_ids=gpuID)
        model_Task = torch.nn.DataParallel(model_Task, device_ids=gpuID)

        ckptdir = os.path.join(root)
        checkpoint = torch.load(ckptdir)
        related_params = {k: v for k, v in checkpoint['init'].items()}
        model_init.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['IDH'].items()}
        model_IDH.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['1p19q'].items()}
        model_1p19q.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['CDKN'].items()}
        model_CDKN.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['Graph'].items()}
        model_Graph.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['His'].items()}
        model_His.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['Cls'].items()}
        model_Cls.load_state_dict(related_params,strict=False)
        related_params = {k: v for k, v in checkpoint['Task'].items()}
        model_Task.load_state_dict(related_params,strict=False)

        model_init.eval()
        model_IDH.eval()
        model_1p19q.eval()
        model_CDKN.eval()
        model_Graph.eval()
        model_His.eval()
        model_Cls.eval()
        model_Task.eval()
        model = [model_init, model_IDH, model_1p19q, model_CDKN, model_Graph, model_His, model_Cls,model_Task]



    print("----------Val-------------")
    # validation_test_vis(opt, model,valLoader, gpuID)
    vis_reconstruct(opt, model,valLoader, gpuID)



def vis_reconstruct(opt, model,dataloader, gpuID):
    CPTAC_label = pd.read_excel(opt['CPTAC_label_path'], header=0)
    IvYGAP_label = pd.read_excel(opt['IvYGAP_label_path'], sheet_name='Sheet1', header=0)
    TCGA_label = pd.read_excel(opt['TCGA_label_path'], sheet_name='Sheet1', header=0)
    combined_labels = pd.concat([TCGA_label, CPTAC_label], ignore_index=True)
    excel_wsi = combined_labels.values

    PATIENT_LIST=excel_wsi[:,0]
    np.random.seed(opt['seed'])
    random.seed(opt['seed'])
    PATIENT_LIST=list(PATIENT_LIST)
    # IvYGAP_label
    IvYGAP_label = IvYGAP_label.values

    PATIENT_LIST=np.unique(PATIENT_LIST)
    np.random.shuffle(PATIENT_LIST)
    NUM_PATIENT_ALL=len(PATIENT_LIST) # 952
    TRAIN_PATIENT_LIST=PATIENT_LIST[0:int(NUM_PATIENT_ALL * 0.8)]
    VAL_PATIENT_LIST = PATIENT_LIST[int(NUM_PATIENT_ALL * 0.9):]
    TEST_PATIENT_LIST = PATIENT_LIST[int(NUM_PATIENT_ALL * 0.80):int(NUM_PATIENT_ALL * 0.90)]
    TEST_LIST = []
    I_TEST_LIST = []

    for i in range(excel_wsi.shape[0]):# 2612

        if excel_wsi[:,0][i] in PATIENT_LIST:
            TEST_LIST.append(excel_wsi[i,:])
    TEST_LIST = np.asarray(TEST_LIST)

    for i in range(TEST_LIST.shape[0]):
        root = '/Res50_feature_2500_fixdim0_norm'

        patient_id = TEST_LIST[i, 0]


        if patient_id[0].startswith('T'):
            base_path = opt['dataDir'] + 'TCGA'
        elif patient_id[0].startswith('W'):
            base_path = opt['dataDir'] + 'IvYGAP'
        elif patient_id[0].startswith('C'):
            base_path = opt['dataDir'] + 'CPTAC'
        else:
            raise ValueError("Unknown data source")

        patch_20 = h5py.File(base_path + root + '_20x/' + TEST_LIST[i, 1] + '.h5')['Res_feature'][:]
        patch_10 = h5py.File(base_path + root + '/' + TEST_LIST[i, 1] + '.h5')['Res_feature'][:]
        img20 = torch.from_numpy(np.array(patch_20[0])).float()
        img10 = torch.from_numpy(np.array(patch_10[0])).float()

        read_details = np.load(opt['dataDir'] + 'read_details/' + TEST_LIST[i, 1] + '.npy', allow_pickle=True)[0]
        WSI_name=TEST_LIST[i, 1]


        if torch.cuda.is_available():
            img20 = img20.cuda(gpuID[0])
            img10 = img10.cuda(gpuID[0])
            # label = label.cuda(gpuID[0])
        # #  # # IDH

        init_feature_his, init_feature_mark, _, _, _, _ = model[0](img20,img10)  # (BS,2500,1024)
        hidden_states, encoded_IDH = model[1](init_feature_mark)
        hidden_states, encoded_1p19q = model[2](hidden_states)
        encoded_CDKN = model[3](hidden_states)
        
        results_dict,weight_IDH_wt,weight_IDH_mut,weight_1p19q_codel,weight_CDKN_HOMDEL,encoded_IDH0,encoded_1p19q0,encoded_CDKN0,Mark_output = \
            model[4](encoded_IDH, encoded_1p19q, encoded_CDKN)
        pred_IDH = results_dict['logits_IDH']
        pred_1p19q = results_dict['logits_1p19q']
        pred_CDKN = results_dict['logits_CDKN']

        hidden_states, encoded_His = model[5](init_feature_his)
        results_dict, weight_His_GBM, weight_His_GBM_Cls2,weight_His_O,His_output = model[6](encoded_His)
        pred_His_2class = results_dict['logits_His_2class']
        pred_His = results_dict['logits_His']


        # wsi_w = h5py.File('vis_results/set0/' + WSI_name + '.h5')['wsi_w'][()]
        # wsi_h = h5py.File('vis_results/set0/' + WSI_name + '.h5')['wsi_h'][()]
        # MPP = h5py.File('vis_results/set0/' + WSI_name + '.h5')['MPP'][()]

        weight_IDH_wt = norm(np.array(weight_IDH_wt.tolist()),read_details.shape[0])
        weight_His_GBM = norm(np.array(weight_His_GBM.tolist()),read_details.shape[0])
        weight_IDH_mut = norm(np.array(weight_IDH_mut.tolist()),read_details.shape[0])
        weight_1p19q_codel = norm(np.array(weight_1p19q_codel.tolist()),read_details.shape[0])
        weight_CDKN_HOMDEL = norm(np.array(weight_CDKN_HOMDEL.tolist()),read_details.shape[0])



        ################################################################



        #

        # relative_MPP = MPP / 0.5
        # PATCH_SIZE_revise = np.int(512 / relative_MPP)

        # wsi_w = np.int(wsi_w * (224 / PATCH_SIZE_revise)) + 1
        # wsi_h = np.int(wsi_h * (224 / PATCH_SIZE_revise)) + 1
        wsi_h=224
        wsi_w=224
        wsi_reconstruct = np.ones(shape=(wsi_h, wsi_w, 3), dtype=np.uint8) * 255
        wsi_reconstruct_nmp = np.ones(shape=(wsi_h, wsi_w, 3), dtype=np.uint8) * 255
        wsi_reconstruct_mut = np.ones(shape=(wsi_h, wsi_w, 3), dtype=np.uint8) * 255
        wsi_reconstruct_pq = np.ones(shape=(wsi_h, wsi_w, 3), dtype=np.uint8) * 255
        wsi_reconstruct_cdkn = np.ones(shape=(wsi_h, wsi_w, 3), dtype=np.uint8) * 255

        for j in range(len(weight_IDH_wt)):
            width_index = np.int(read_details[j][0])
            height_index = np.int(read_details[j][1])
            wsi_reconstruct[height_index * 224:(height_index + 1) * 224, width_index * 224:(width_index + 1) * 224,
            :] = weight_IDH_wt[j]
            wsi_reconstruct_nmp[height_index * 224:(height_index + 1) * 224, width_index * 224:(width_index + 1) * 224,
            :] = weight_His_GBM[j]
            wsi_reconstruct_mut[height_index * 224:(height_index + 1) * 224, width_index * 224:(width_index + 1) * 224,
            :] = weight_IDH_mut[j]
            wsi_reconstruct_pq[height_index * 224:(height_index + 1) * 224, width_index * 224:(width_index + 1) * 224,
            :] = weight_1p19q_codel[j]
            wsi_reconstruct_cdkn[height_index * 224:(height_index + 1) * 224, width_index * 224:(width_index + 1) * 224,
            :] = weight_CDKN_HOMDEL[j]

            #

        wsi_reconstruct = Image.fromarray(wsi_reconstruct)
        wsi_reconstruct = wsi_reconstruct.resize((2000, int(wsi_h / wsi_w * 2000)))
        wsi_reconstruct.save('vis_results/set2/' + WSI_name + '_vis_IDHwt.jpg')

        wsi_reconstruct_nmp = Image.fromarray(wsi_reconstruct_nmp)
        wsi_reconstruct_nmp = wsi_reconstruct_nmp.resize((2000, int(wsi_h / wsi_w * 2000)))
        wsi_reconstruct_nmp.save('vis_results/set2/' + WSI_name + '_vis_nmp.jpg')

        wsi_reconstruct_mut = Image.fromarray(wsi_reconstruct_mut)
        wsi_reconstruct_mut = wsi_reconstruct_mut.resize((2000, int(wsi_h / wsi_w * 2000)))
        wsi_reconstruct_mut.save('vis_results/set2/' + WSI_name + '_vis_IDHmut.jpg')

        wsi_reconstruct_pq = Image.fromarray(wsi_reconstruct_pq)
        wsi_reconstruct_pq = wsi_reconstruct_pq.resize((2000, int(wsi_h / wsi_w * 2000)))
        wsi_reconstruct_pq.save('vis_results/set2/' + WSI_name + '_vis_pq.jpg')

        wsi_reconstruct_cdkn = Image.fromarray(wsi_reconstruct_cdkn)
        wsi_reconstruct_cdkn = wsi_reconstruct_cdkn.resize((2000, int(wsi_h / wsi_w * 2000)))
        wsi_reconstruct_cdkn.save('vis_results/set2/' + WSI_name + '_vis_cdkn.jpg')


        count += 1
        print(i)





def validation_test_vis(opt,model, dataloader,gpuID):

    CPTAC_label = pd.read_excel(opt['CPTAC_label_path'], header=0)
    IvYGAP_label = pd.read_excel(opt['IvYGAP_label_path'], sheet_name='Sheet1', header=0)
    TCGA_label = pd.read_excel(opt['TCGA_label_path'], sheet_name='Sheet1', header=0)
    combined_labels = pd.concat([TCGA_label, CPTAC_label], ignore_index=True)
    excel_label_wsi = combined_labels.values
    
    excel_wsi = list(excel_label_wsi.values)
    excel_wsi_new=[]
    test_bar = tqdm(dataloader)
    bs = 1
    count = 0
    for packs in test_bar:
        img20,img10, label= packs[0]



        if torch.cuda.is_available():
            img20 = img20.cuda(gpuID[0])
            img10 = img10.cuda(gpuID[0])
            # label = label.cuda(gpuID[0])
        # #  # # IDH

        init_feature_his, init_feature_mark, _, _, _, _ = model[0](img20,img10)  # (BS,2500,1024)
        hidden_states, encoded_IDH = model[1](init_feature_mark)
        hidden_states, encoded_1p19q = model[2](hidden_states)
        encoded_CDKN = model[3](hidden_states)
        
        results_dict,weight_IDH_wt,weight_IDH_mut,weight_1p19q_codel,weight_CDKN_HOMDEL,encoded_IDH0,encoded_1p19q0,encoded_CDKN0,Mark_output = \
            model[4](encoded_IDH, encoded_1p19q, encoded_CDKN)
        pred_IDH = results_dict['logits_IDH']
        pred_1p19q = results_dict['logits_1p19q']
        pred_CDKN = results_dict['logits_CDKN']

        hidden_states, encoded_His = model[5](init_feature_his)
        results_dict, weight_His_GBM, weight_His_GBM_Cls2,weight_His_O,His_output = model[6](encoded_His)
        pred_His_2class = results_dict['logits_His_2class']
        pred_His = results_dict['logits_His']
        his_mark = model[7](His_output.float(), Mark_output.float())
        weight_IDH_wt=weight_IDH_wt.tolist()

        _, pred_His = torch.max(pred_His.data, 1)
        pred_His = pred_His.tolist()
        _, pred_IDH = torch.max(pred_IDH.data, 1)
        pred_IDH = pred_IDH.tolist()
        _, pred_1p19q = torch.max(pred_1p19q.data, 1)
        pred_1p19q = pred_1p19q.tolist()
        _, pred_CDKN = torch.max(pred_CDKN.data, 1)
        pred_CDKN = pred_CDKN.tolist()
        _, pred_His_2class = torch.max(pred_His_2class.data, 1)
        pred_His_2class = pred_His_2class.tolist()
        _, pred_Task = torch.max(his_mark.data, 1)
        pred_Task = pred_Task.tolist()


        if pred_His[0]==3 and pred_IDH[0]==0 and pred_1p19q[0]==0 and (pred_CDKN[0]==label[:, 2].tolist()[0]):
            excel_wsi_new.append(excel_wsi[count])
        count += 1
    excel_wsi_new=np.array(excel_wsi_new)
    df = pd.DataFrame(excel_wsi_new, columns=list(excel_label_wsi))
    df.to_excel("vis/Test_TPall.xlsx",index=False)


def norm(weight,num_patch):
    N_biorepet = int(2500 / num_patch)
    weight_0=weight[0:num_patch]
    weight_color=[]
    if N_biorepet>1:
        for j in range(N_biorepet-1):
            weight_0+=weight[(j+1)*num_patch:(j+2)*num_patch]

    min_w=np.min(weight_0)
    max_w = np.max(weight_0)
    weight_0=(weight_0-min_w)/(max_w-min_w)

    cmap = cmaps.MPL_viridis  # 引用NCL的colormap
    newcolors = cmap(np.linspace(0, 1, 256))*255
    newcolors = np.trunc(newcolors)
    newcolors = newcolors.astype(int)
    ref_array=np.zeros(shape=[256])

    for k in range(256):
        ref_array[k]=k/256

    for k in range(weight_0.shape[0]):
        delta=np.abs(ref_array-weight_0[k]*weight_0[k])
        delta=list(delta)
        min_del=min(delta)
        min_del_index=delta.index(min_del)
        weight_color.append(newcolors[min_del_index][0:3])



    return weight_color



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='config/mine.yml')
    args = parser.parse_args()
    with open(args.opt) as f:
        opt = yaml.load(f, Loader=SafeLoader)

    setup_seed(opt['seed'])
    sysstr = platform.system()
    opt['logDir'] = os.path.join(opt['logDir'], 'Mine')
    if not os.path.exists(opt['logDir']):
        os.makedirs(opt['logDir'])
    train(opt)




    a=1





























