# from apex import amp
from utils import *
import dataset_mine
from net import init_weights,get_scheduler,WarmupCosineSchedule

def train(opt):
    root_IDH=r'./models/Mine_pretrain2_exp_45_test10_0302-0538/Mine_model-0008.pth'
    root_1p19q = r'./models/Mine_pretrain2_exp_45_test10_0302-0538/Mine_model-0008.pth'
    root_CDKN = r'./models/Mine_pretrain2_exp_45_test10_0302-0538/Mine_model-0008.pth'
    root_His = r'./models/Mine_pretrain2_exp_45_test10_0302-0538/Mine_model-0008.pth'


    gpuID = opt['gpus']
    valDataset = dataset_mine.Our_Dataset(phase='Val', opt=opt)
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
        IDH_model_init = Mine_init(opt).cuda(gpuID[0])
        IDH_model_IDH = Mine_IDH(opt).cuda(gpuID[0])
        IDH_model_1p19q = Mine_1p19q(opt).cuda(gpuID[0])
        IDH_model_CDKN = Mine_CDKN(opt).cuda(gpuID[0])
        IDH_model_Graph = Label_correlation_Graph(opt).cuda(gpuID[0])
        IDH_model_init = torch.nn.DataParallel(IDH_model_init, device_ids=gpuID)
        IDH_model_IDH = torch.nn.DataParallel(IDH_model_IDH, device_ids=gpuID)
        IDH_model_1p19q = torch.nn.DataParallel(IDH_model_1p19q, device_ids=gpuID)
        IDH_model_CDKN = torch.nn.DataParallel(IDH_model_CDKN, device_ids=gpuID)
        IDH_model_Graph = torch.nn.DataParallel(IDH_model_Graph, device_ids=gpuID)


        ckptdir = os.path.join(root_IDH)
        checkpoint = torch.load(ckptdir)
        related_params = {k: v for k, v in checkpoint['init'].items()}
        IDH_model_init.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['IDH'].items()}
        IDH_model_IDH.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['1p19q'].items()}
        IDH_model_1p19q.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['CDKN'].items()}
        IDH_model_CDKN.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['Graph'].items()}
        IDH_model_Graph.load_state_dict(related_params,strict=False)

        IDH_model_init.eval()
        IDH_model_IDH.eval()
        IDH_model_1p19q.eval()
        IDH_model_CDKN.eval()
        IDH_model_Graph.eval()
        IDH_model=[IDH_model_init,IDH_model_IDH,IDH_model_1p19q,IDH_model_CDKN,IDH_model_Graph]

        ## 1p19q
        p19q_model_init = Mine_init(opt).cuda(gpuID[0])
        p19q_model_IDH = Mine_IDH(opt).cuda(gpuID[0])
        p19q_model_1p19q = Mine_1p19q(opt).cuda(gpuID[0])
        p19q_model_CDKN = Mine_CDKN(opt).cuda(gpuID[0])
        p19q_model_Graph = Label_correlation_Graph(opt).cuda(gpuID[0])
        p19q_model_init = torch.nn.DataParallel(p19q_model_init, device_ids=gpuID)
        p19q_model_IDH = torch.nn.DataParallel(p19q_model_IDH, device_ids=gpuID)
        p19q_model_1p19q = torch.nn.DataParallel(p19q_model_1p19q, device_ids=gpuID)
        p19q_model_CDKN = torch.nn.DataParallel(p19q_model_CDKN, device_ids=gpuID)
        p19q_model_Graph = torch.nn.DataParallel(p19q_model_Graph, device_ids=gpuID)

        ckptdir = os.path.join(root_1p19q)
        checkpoint = torch.load(ckptdir)
        related_params = {k: v for k, v in checkpoint['init'].items()}
        p19q_model_init.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['IDH'].items()}
        p19q_model_IDH.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['1p19q'].items()}
        p19q_model_1p19q.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['CDKN'].items()}
        p19q_model_CDKN.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['Graph'].items()}
        p19q_model_Graph.load_state_dict(related_params,strict=False)

        p19q_model_init.eval()
        p19q_model_IDH.eval()
        p19q_model_1p19q.eval()
        p19q_model_CDKN.eval()
        p19q_model_Graph.eval()
        p19q_model = [p19q_model_init, p19q_model_IDH, p19q_model_1p19q, p19q_model_CDKN, p19q_model_Graph]

        ## CDKN
        CDKN_model_init = Mine_init(opt).cuda(gpuID[0])
        CDKN_model_IDH = Mine_IDH(opt).cuda(gpuID[0])
        CDKN_model_1p19q = Mine_1p19q(opt).cuda(gpuID[0])
        CDKN_model_CDKN = Mine_CDKN(opt).cuda(gpuID[0])
        CDKN_model_Graph = Label_correlation_Graph(opt).cuda(gpuID[0])
        CDKN_model_init = torch.nn.DataParallel(CDKN_model_init, device_ids=gpuID)
        CDKN_model_IDH = torch.nn.DataParallel(CDKN_model_IDH, device_ids=gpuID)
        CDKN_model_1p19q = torch.nn.DataParallel(CDKN_model_1p19q, device_ids=gpuID)
        CDKN_model_CDKN = torch.nn.DataParallel(CDKN_model_CDKN, device_ids=gpuID)
        CDKN_model_Graph = torch.nn.DataParallel(CDKN_model_Graph, device_ids=gpuID)

        ckptdir = os.path.join(root_CDKN)
        checkpoint = torch.load(ckptdir)
        related_params = {k: v for k, v in checkpoint['init'].items()}
        CDKN_model_init.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['IDH'].items()}
        CDKN_model_IDH.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['1p19q'].items()}
        CDKN_model_1p19q.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['CDKN'].items()}
        CDKN_model_CDKN.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['Graph'].items()}
        CDKN_model_Graph.load_state_dict(related_params,strict=False)

        CDKN_model_init.eval()
        CDKN_model_IDH.eval()
        CDKN_model_1p19q.eval()
        CDKN_model_CDKN.eval()
        CDKN_model_Graph.eval()
        CDKN_model = [CDKN_model_init, CDKN_model_IDH, CDKN_model_1p19q, CDKN_model_CDKN, CDKN_model_Graph]

        ## His
        His_model_init = Mine_init(opt).cuda(gpuID[0])
        His_model_His = Mine_His(opt).cuda(gpuID[0])
        His_model_Cls = Cls_His_Grade(opt).cuda(gpuID[0])
        His_model_init = torch.nn.DataParallel(His_model_init, device_ids=gpuID)
        His_model_His = torch.nn.DataParallel(His_model_His, device_ids=gpuID)
        His_model_Cls = torch.nn.DataParallel(His_model_Cls, device_ids=gpuID)



        ckptdir = os.path.join(root_His)
        checkpoint = torch.load(ckptdir)
        related_params = {k: v for k, v in checkpoint['init'].items()}
        His_model_init.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['His'].items()}
        His_model_His.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['Cls'].items()}
        His_model_Cls.load_state_dict(related_params)

        His_model_init.eval()
        His_model_His.eval()
        His_model_Cls.eval()
        His_model = [His_model_init, His_model_His, His_model_Cls]

    print("----------Val-------------")
    list_WSI_IDH,list_WSI_1p19q,list_WSI_CDKN,list_WSI_His,list_WSI_His_2class,list_WSI_Diag = validation_All_sepe(opt, IDH_model, p19q_model, CDKN_model,His_model,valLoader, gpuID)
    print('val in epoch: %d/%d, acc_IDH:%.3f,acc_1p19q:%.3f,acc_CDKN:%.3f,acc_His:%.3f,acc_His_2class:%.3f, acc_Diag:%.3f' % (
        0 + 1, alleps, list_WSI_IDH[0], list_WSI_1p19q[0], list_WSI_CDKN[0], list_WSI_His[0], list_WSI_His_2class[0], list_WSI_Diag[0]))

    # val_dict_IDH = {'val/acc_IDH': list_WSI_IDH[0], 'val/sen_IDH': list_WSI_IDH[3], 'val/spec_IDH': list_WSI_IDH[4],
    #             'val/auc_IDH': list_WSI_IDH[5], 'val/f1_IDH': list_WSI_IDH[2], 'val/prec_IDH': list_WSI_IDH[6],                        }
    # val_dict_1p19q = {'val/acc_1p19q': list_WSI_1p19q[0], 'val/sen_1p19q': list_WSI_1p19q[3],'val/spec_1p19q': list_WSI_1p19q[4],
    #             'val/auc_1p19q': list_WSI_1p19q[5], 'val/f1_1p19q': list_WSI_1p19q[2],'val/prec_1p19q': list_WSI_1p19q[6],}
    # val_dict_CDKN = {'val/acc_CDKN': list_WSI_CDKN[0], 'val/sen_CDKN': list_WSI_CDKN[3],'val/spec_CDKN': list_WSI_CDKN[4],
    #             'val/auc_CDKN': list_WSI_CDKN[5], 'val/f1_CDKN': list_WSI_CDKN[2],'val/prec_CDKN': list_WSI_CDKN[6],}
    # val_dict_His = {'val/acc_His': list_WSI_His[0], 'val/sen_His': list_WSI_His[3], 'val/spec_His': list_WSI_His[4],
    #             'val/auc_His': list_WSI_His[5], 'val/f1_His': list_WSI_His[2], 'val/prec_His': list_WSI_His[6],}
    # val_dict_His_2class = {'val/acc_His_2class': list_WSI_His_2class[0], 'val/sen_His_2class': list_WSI_His_2class[3],'val/spec_His_2classe': list_WSI_His_2class[4],
    #             'val/auc_His_2class': list_WSI_His_2class[5], 'val/f1_His_2class': list_WSI_His_2class[2],'val/prec_His_2class': list_WSI_His_2class[6],}
    # val_dict_Diag = {'val/acc_Diag': list_WSI_Diag[0], 'val/sen_Diag': list_WSI_Diag[3],'val/spec_Diag': list_WSI_Diag[4],
    #             'val/auc_Diag': list_WSI_Diag[5], 'val/f1_Diag': list_WSI_Diag[2],'val/prec_Diag': list_WSI_Diag[6], }
    # saver.write_log(curep, val_dict_IDH, 'val_IDH')
    # saver.write_log(curep, val_dict_1p19q, 'val_1p19q')
    # saver.write_log(curep, val_dict_CDKN, 'val_CDKN')
    # saver.write_log(curep, val_dict_His, 'val_His')
    # saver.write_log(curep, val_dict_His_2class, 'val_His_2class')
    # saver.write_log(curep, val_dict_Diag, 'val_Diag')

def validation_test_speed(opt,IDH_model, p19q_model, CDKN_model,His_model, dataloader,gpuID):
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
        G23_O_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0,'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        G23_A_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0,'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        G4_A_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0,'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        GBM_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0,'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        all_metrics = {'sen': 0, 'spec': 0,'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}


        count_His = 0
        correct_His = 0
        A_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        O_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        GBM_His_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        all_metrics_His = {'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}

    test_bar = tqdm(dataloader)
    bs=1
    count=0
    for packs in test_bar:
        img = packs[0]
        label = packs[1]
        count += 1

        if torch.cuda.is_available():
            img = img.cuda(gpuID[0])
            label = label.cuda(gpuID[0])
        # #  # # IDH

        init_feature = IDH_model[0](img)  # (BS,2500,1024)
        hidden_states, encoded_IDH = IDH_model[1](init_feature)
        hidden_states, encoded_1p19q = IDH_model[2](hidden_states)
        encoded_CDKN = IDH_model[3](hidden_states)
        results_dict,_,__,I_,I__,I___ = IDH_model[4](encoded_IDH, encoded_1p19q, encoded_CDKN)
        pred_IDH_ori = results_dict['logits_IDH']




        # #  # # 1p19q

        init_feature = p19q_model[0](img)  # (BS,2500,1024)
        hidden_states, encoded_IDH = p19q_model[1](init_feature)
        hidden_states, encoded_1p19q = p19q_model[2](hidden_states)
        encoded_CDKN = p19q_model[3](hidden_states)
        results_dict, _, __, I_, I__, I___ = p19q_model[4](encoded_IDH, encoded_1p19q, encoded_CDKN)
        pred_1p19q_ori = results_dict['logits_1p19q']

        # #  # # CDKN
        init_feature = CDKN_model[0](img)  # (BS,2500,1024)
        hidden_states, encoded_IDH = CDKN_model[1](init_feature)
        hidden_states, encoded_1p19q = CDKN_model[2](hidden_states)
        encoded_CDKN = CDKN_model[3](hidden_states)
        results_dict, _, __, I_, I__, I___ = CDKN_model[4](encoded_IDH, encoded_1p19q, encoded_CDKN)
        pred_CDKN_ori = results_dict['logits_CDKN']

        # #  # # His

        init_feature = His_model[0](img)  # (BS,2500,1024)
        hidden_states, encoded_His = His_model[1](init_feature)
        results_dict, _, __ = His_model[2](encoded_His)
        pred_His_ori = results_dict['logits_His']
        pred_His_2class_ori = results_dict['logits_His_2class']






        _, pred_IDH = torch.max(pred_IDH_ori.data, 1)
        pred_IDH = pred_IDH.tolist()
        gt_IDH = label[:, 0].tolist()

        _, pred_1p19q = torch.max(pred_1p19q_ori.data, 1)
        pred_1p19q = pred_1p19q.tolist()
        gt_1p19q = label[:, 1].tolist()

        _, pred_CDKN = torch.max(pred_CDKN_ori.data, 1)
        pred_CDKN = pred_CDKN.tolist()
        gt_CDKN = label[:, 2].tolist()

        _, pred_His_2class = torch.max(pred_His_2class_ori.data, 1)
        pred_His_2class = pred_His_2class.tolist()
        gt_His_2class = label[:, 6].tolist()

        _, pred_His = torch.max(pred_His_ori[:, 1:].data, 1)
        pred_His = pred_His.tolist()
        gt_His = label[:, 3].tolist()


        gt_Diag = label[:, 5].tolist()
        pred_Diag=Diag_predict(pred_IDH_ori,pred_1p19q_ori,pred_CDKN_ori,pred_His_2class_ori)


def validation_All_sepe(opt,IDH_model, p19q_model, CDKN_model,His_model, dataloader,gpuID):

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
        G23_O_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0,'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        G23_A_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0,'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        G4_A_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0,'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        GBM_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0,'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        all_metrics = {'sen': 0, 'spec': 0,'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}


        count_His = 0
        correct_His = 0
        A_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        O_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        GBM_His_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        all_metrics_His = {'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}

    test_bar = tqdm(dataloader)
    bs=1
    count=0
    for packs in test_bar:
        img = packs[0]
        label = packs[1]
        count += 1

        if torch.cuda.is_available():
            img = img.cuda(gpuID[0])
            label = label.cuda(gpuID[0])
        # #  # # IDH

        init_feature = IDH_model[0](img)  # (BS,2500,1024)
        hidden_states, encoded_IDH = IDH_model[1](init_feature)
        hidden_states, encoded_1p19q = IDH_model[2](hidden_states)
        encoded_CDKN = IDH_model[3](hidden_states)
        results_dict,_,__,I_,I__,I___ = IDH_model[4](encoded_IDH, encoded_1p19q, encoded_CDKN)
        pred_IDH_ori = results_dict['logits_IDH']




        # #  # # 1p19q

        init_feature = p19q_model[0](img)  # (BS,2500,1024)
        hidden_states, encoded_IDH = p19q_model[1](init_feature)
        hidden_states, encoded_1p19q = p19q_model[2](hidden_states)
        encoded_CDKN = p19q_model[3](hidden_states)
        results_dict, _, __, I_, I__, I___ = p19q_model[4](encoded_IDH, encoded_1p19q, encoded_CDKN)
        pred_1p19q_ori = results_dict['logits_1p19q']

        # #  # # CDKN
        init_feature = CDKN_model[0](img)  # (BS,2500,1024)
        hidden_states, encoded_IDH = CDKN_model[1](init_feature)
        hidden_states, encoded_1p19q = CDKN_model[2](hidden_states)
        encoded_CDKN = CDKN_model[3](hidden_states)
        results_dict, _, __, I_, I__, I___ = CDKN_model[4](encoded_IDH, encoded_1p19q, encoded_CDKN)
        pred_CDKN_ori = results_dict['logits_CDKN']

        # #  # # His

        init_feature = His_model[0](img)  # (BS,2500,1024)
        hidden_states, encoded_His = His_model[1](init_feature)
        results_dict, _, __ = His_model[2](encoded_His)
        pred_His_ori = results_dict['logits_His']
        pred_His_2class_ori = results_dict['logits_His_2class']






        _, pred_IDH = torch.max(pred_IDH_ori.data, 1)
        pred_IDH = pred_IDH.tolist()
        gt_IDH = label[:, 0].tolist()

        _, pred_1p19q = torch.max(pred_1p19q_ori.data, 1)
        pred_1p19q = pred_1p19q.tolist()
        gt_1p19q = label[:, 1].tolist()

        _, pred_CDKN = torch.max(pred_CDKN_ori.data, 1)
        pred_CDKN = pred_CDKN.tolist()
        gt_CDKN = label[:, 2].tolist()

        _, pred_His_2class = torch.max(pred_His_2class_ori.data, 1)
        pred_His_2class = pred_His_2class.tolist()
        gt_His_2class = label[:, 6].tolist()

        _, pred_His = torch.max(pred_His_ori[:, 1:].data, 1)
        pred_His = pred_His.tolist()
        gt_His = label[:, 3].tolist()


        gt_Diag = label[:, 5].tolist()
        pred_Diag=Diag_predict(pred_IDH_ori,pred_1p19q_ori,pred_CDKN_ori,pred_His_2class_ori)

        for j in range(bs):
            ######   IDH
            label_all_IDH.append(gt_IDH[j])
            predicted_all_IDH.append(pred_IDH_ori.detach().cpu().numpy()[j][1])
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
            predicted_all_1p19q.append(pred_1p19q_ori.detach().cpu().numpy()[j][1])
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
            predicted_all_CDKN.append(pred_CDKN_ori.detach().cpu().numpy()[j][1])
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
            predicted_all_His_2class.append(pred_His_2class_ori.detach().cpu().numpy()[j][1])
            if gt_His_2class[j] == 0 and pred_His_2class[j] == 0:
                tn_His_2class += 1
            if gt_His_2class[j] == 0 and pred_His_2class[j] == 1:
                fp_His_2class += 1
            if gt_His_2class[j] == 1 and pred_His_2class[j] == 0:
                fn_His_2class += 1
            if gt_His_2class[j] == 1 and pred_His_2class[j] == 1:
                tp_His_2class += 1

            ##################   Grade
            # if not gt_Grade[j] == 2:
            #     label_all_Grade.append(gt_Grade[j])
            #     predicted_all_Grade.append(F.softmax(pred_Grade_ori[:, 0:2], dim=1).detach().cpu().numpy()[j][1])
            #     if gt_Grade[j] == 0 and pred_Grade[j] == 0:
            #         tn_Grade += 1
            #     if gt_Grade[j] == 0 and pred_Grade[j] == 1:
            #         fp_Grade += 1
            #     if gt_Grade[j] == 1 and pred_Grade[j] == 0:
            #         fn_Grade += 1
            #     if gt_Grade[j] == 1 and pred_Grade[j] == 1:
            #         tp_Grade += 1

            ##################   His
            if not gt_His[j] == 0:
                # A
                if pred_His[j] == 0:
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
                if pred_His[j] == 1:
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
                if pred_His[j] == 2:
                    if gt_His[j] == 3:
                        GBM_His_metrics['tn'] += 1
                    else:
                        GBM_His_metrics['fn'] += 1
                else:
                    if not gt_His[j] == 3:
                        GBM_His_metrics['tp'] += 1
                    else:
                        GBM_His_metrics['fp'] += 1
            if not gt_His[j] == 0:
                count_His += 1
            if gt_His[j] == 1 and pred_His[j] == 0:
                correct_His += 1
            if gt_His[j] == 2 and pred_His[j] == 1:
                correct_His += 1
            if gt_His[j] == 3 and pred_His[j] == 2:
                correct_His += 1
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


            count_Diag+=1
            correct_Diag += Diag_process(label, pred_IDH_ori, pred_1p19q_ori, pred_CDKN_ori, pred_His_2class_ori)

    ##########################################   IDH
    Acc_IDH = (tp_IDH + tn_IDH) / (tp_IDH + tn_IDH + fp_IDH + fn_IDH)
    Sen_IDH = (tp_IDH) / (tp_IDH + fn_IDH+0.000001)  # recall
    Spec_IDH = (tn_IDH) / (tn_IDH + fp_IDH+0.000001)
    precision_IDH = (tp_IDH) / (tp_IDH + fp_IDH+0.000001)
    recall_IDH = Sen_IDH
    f1_score_IDH = (2 * precision_IDH * recall_IDH) / (precision_IDH + recall_IDH+0.000001)
    AUC_IDH = metrics.roc_auc_score(y_true=np.array(label_all_IDH), y_score=np.array(predicted_all_IDH))
    list_IDH=(Acc_IDH, None, f1_score_IDH, Sen_IDH,Spec_IDH,AUC_IDH,precision_IDH)

    ##########################################   1p19q
    Acc_1p19q = (tp_1p19q + tn_1p19q) / (tp_1p19q + tn_1p19q + fp_1p19q + fn_1p19q)
    Sen_1p19q = (tp_1p19q) / (tp_1p19q + fn_1p19q+0.000001)  # recall
    Spec_1p19q = (tn_1p19q) / (tn_1p19q + fp_1p19q+0.000001)
    precision_1p19q = (tp_1p19q) / (tp_1p19q + fp_1p19q+0.000001)
    recall_1p19q = Sen_1p19q
    f1_score_1p19q = (2 * precision_1p19q * recall_1p19q) / (precision_1p19q + recall_1p19q+0.000001)
    AUC_1p19q = metrics.roc_auc_score(y_true=np.array(label_all_1p19q), y_score=np.array(predicted_all_1p19q))
    list_1p19q = (Acc_1p19q, None, f1_score_1p19q, Sen_1p19q, Spec_1p19q, AUC_1p19q, precision_1p19q)
    ##########################################   CDKN
    Acc_CDKN = (tp_CDKN + tn_CDKN) / (tp_CDKN + tn_CDKN + fp_CDKN + fn_CDKN)
    Sen_CDKN = (tp_CDKN) / (tp_CDKN + fn_CDKN+0.000001)  # recall
    Spec_CDKN = (tn_CDKN) / (tn_CDKN + fp_CDKN+0.000001)
    precision_CDKN = (tp_CDKN) / (tp_CDKN + fp_CDKN+0.000001)
    recall_CDKN = Sen_CDKN
    f1_score_CDKN = (2 * precision_CDKN * recall_CDKN) / (precision_CDKN + recall_CDKN+0.000001)
    AUC_CDKN = metrics.roc_auc_score(y_true=np.array(label_all_CDKN), y_score=np.array(predicted_all_CDKN))
    list_CDKN = (Acc_CDKN, None, f1_score_CDKN, Sen_CDKN, Spec_CDKN, AUC_CDKN, precision_CDKN)
    ##########################################   His_2class
    Acc_His_2class = (tp_His_2class + tn_His_2class) / (tp_His_2class + tn_His_2class + fp_His_2class + fn_His_2class)
    Sen_His_2class = (tp_His_2class) / (tp_His_2class + fn_His_2class + 0.000001)  # recall
    Spec_His_2class = (tn_His_2class) / (tn_His_2class + fp_His_2class + 0.000001)
    precision_His_2class = (tp_His_2class) / (tp_His_2class + fp_His_2class + 0.000001)
    recall_His_2class = Sen_His_2class
    f1_score_His_2class = (2 * precision_His_2class * recall_His_2class) / (precision_His_2class + recall_His_2class + 0.000001)
    AUC_His_2class = metrics.roc_auc_score(y_true=np.array(label_all_His_2class),y_score=np.array(predicted_all_His_2class))
    list_His_2class = (Acc_His_2class, None, f1_score_His_2class, Sen_His_2class, Spec_His_2class, AUC_His_2class, precision_His_2class)
    ##########################################   Grade
    # Acc_Grade = (tp_Grade + tn_Grade) / (tp_Grade + tn_Grade + fp_Grade + fn_Grade)
    # Sen_Grade = (tp_Grade) / (tp_Grade + fn_Grade + 0.000001)  # recall
    # Spec_Grade = (tn_Grade) / (tn_Grade + fp_Grade + 0.000001)
    # precision_Grade = (tp_Grade) / (tp_Grade + fp_Grade + 0.000001)
    # recall_Grade = Sen_Grade
    # f1_score_Grade = (2 * precision_Grade * recall_Grade) / (precision_Grade + recall_Grade + 0.000001)
    # AUC_Grade = metrics.roc_auc_score(y_true=np.array(label_all_Grade), y_score=np.array(predicted_all_Grade))
    # list_Grade = (Acc_Grade, None, f1_score_Grade, Sen_Grade, Spec_Grade, AUC_Grade, precision_Grade)
    ##########################################   His
    Acc_His = correct_His / count_His
    #  Sensitivity
    A_metrics['sen'] = (A_metrics['tp']) / (A_metrics['tp'] + A_metrics['fn'] + 0.000001)
    O_metrics['sen'] = (O_metrics['tp']) / (O_metrics['tp'] + O_metrics['fn'] + 0.000001)
    GBM_His_metrics['sen'] = (GBM_His_metrics['tp']) / (GBM_His_metrics['tp'] + GBM_His_metrics['fn'] + 0.000001)
    all_metrics_His['sen'] = (A_metrics['sen']  + O_metrics['sen'] + GBM_His_metrics['sen']) / 3
    #  Spec
    A_metrics['spec'] = (A_metrics['tn']) / (A_metrics['tn'] + A_metrics['fp'] + 0.000001)
    O_metrics['spec'] = (O_metrics['tn']) / (O_metrics['tn'] + O_metrics['fp'] + 0.000001)
    GBM_His_metrics['spec'] = (GBM_His_metrics['tn']) / (GBM_His_metrics['tn'] + GBM_His_metrics['fp'] + 0.000001)
    all_metrics_His['spec'] = (A_metrics['spec'] + O_metrics['spec'] +GBM_His_metrics['spec']) / 3
    #  Precision
    A_metrics['pre'] = (A_metrics['tp']) / (A_metrics['tp'] + A_metrics['fp'] + 0.000001)
    O_metrics['pre'] = (O_metrics['tp']) / (O_metrics['tp'] + O_metrics['fp'] + 0.000001)
    GBM_His_metrics['pre'] = (GBM_His_metrics['tp']) / (GBM_His_metrics['tp'] + GBM_His_metrics['fp'] + 0.000001)
    all_metrics_His['pre'] = (A_metrics['pre']  + O_metrics['pre'] +GBM_His_metrics['pre']) / 3
    #  Recall
    A_metrics['recall'] = (A_metrics['tp']) / (A_metrics['tp'] + A_metrics['fn'] + 0.000001)
    O_metrics['recall'] = (O_metrics['tp']) / (O_metrics['tp'] + O_metrics['fn'] + 0.000001)
    GBM_His_metrics['recall'] = (GBM_His_metrics['tp']) / (GBM_His_metrics['tp'] + GBM_His_metrics['fn'] + 0.000001)
    all_metrics_His['recall'] = (A_metrics['recall']+ O_metrics['recall'] + GBM_His_metrics['recall']) / 3
    #  F1
    A_metrics['f1'] = (2 * A_metrics['pre'] * A_metrics['recall']) / (
            A_metrics['pre'] + A_metrics['recall'] + 0.000001)
    O_metrics['f1'] = (2 * O_metrics['pre'] * O_metrics['recall']) / (
            O_metrics['pre'] + O_metrics['recall'] + 0.000001)
    GBM_His_metrics['f1'] = (2 * GBM_His_metrics['pre'] * GBM_His_metrics['recall']) / (
                GBM_His_metrics['pre'] + GBM_His_metrics['recall'] + 0.000001)
    all_metrics_His['f1'] = (A_metrics['f1']  + O_metrics['f1'] +
                         GBM_His_metrics['f1']) / 3
    list_His = (Acc_His, 0, all_metrics_His['f1'], all_metrics_His['sen'], all_metrics_His['spec'], all_metrics_His['AUC'], all_metrics_His['pre'])

    ################################################   Diag
    Acc_Diag=correct_Diag/count_Diag
    #  Sensitivity
    G23_O_metrics['sen']= (G23_O_metrics['tp'])/(G23_O_metrics['tp']+G23_O_metrics['fn']+0.000001)
    G23_A_metrics['sen']= (G23_A_metrics['tp'])/(G23_A_metrics['tp']+G23_A_metrics['fn']+0.000001)
    G4_A_metrics['sen'] = (G4_A_metrics['tp']) / (G4_A_metrics['tp'] + G4_A_metrics['fn']+0.000001)
    GBM_metrics['sen'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fn']+0.000001)
    all_metrics['sen']=(G23_O_metrics['sen']+G23_A_metrics['sen']+G4_A_metrics['sen']+GBM_metrics['sen'])/4
    #  Spec
    G23_O_metrics['spec'] = (G23_O_metrics['tn']) / (G23_O_metrics['tn'] + G23_O_metrics['fp']+0.000001)
    G23_A_metrics['spec'] = (G23_A_metrics['tn']) / (G23_A_metrics['tn'] + G23_A_metrics['fp']+0.000001)
    G4_A_metrics['spec'] = (G4_A_metrics['tn']) / (G4_A_metrics['tn'] + G4_A_metrics['fp']+0.000001)
    GBM_metrics['spec'] = (GBM_metrics['tn']) / (GBM_metrics['tn'] + GBM_metrics['fp']+0.000001)
    all_metrics['spec'] = (G23_O_metrics['spec'] + G23_A_metrics['spec'] + G4_A_metrics['spec'] + GBM_metrics['spec']) / 4
    #  Precision
    G23_O_metrics['pre'] = (G23_O_metrics['tp']) / (G23_O_metrics['tp'] + G23_O_metrics['fp']+0.000001)
    G23_A_metrics['pre'] = (G23_A_metrics['tp']) / (G23_A_metrics['tp'] + G23_A_metrics['fp']+0.000001)
    G4_A_metrics['pre'] = (G4_A_metrics['tp']) / (G4_A_metrics['tp'] + G4_A_metrics['fp']+0.000001)
    GBM_metrics['pre'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fp']+0.000001)
    all_metrics['pre'] = (G23_O_metrics['pre'] + G23_A_metrics['pre'] + G4_A_metrics['pre'] + GBM_metrics['pre']) / 4
    #  Recall
    G23_O_metrics['recall'] = (G23_O_metrics['tp']) / (G23_O_metrics['tp'] + G23_O_metrics['fn']+0.000001)
    G23_A_metrics['recall'] = (G23_A_metrics['tp']) / (G23_A_metrics['tp'] + G23_A_metrics['fn']+0.000001)
    G4_A_metrics['recall'] = (G4_A_metrics['tp']) / (G4_A_metrics['tp'] + G4_A_metrics['fn']+0.000001)
    GBM_metrics['recall'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fn']+0.000001)
    all_metrics['recall'] = (G23_O_metrics['recall'] + G23_A_metrics['recall'] + G4_A_metrics['recall'] + GBM_metrics['recall']) / 4

    #  F1
    G23_O_metrics['f1'] = 2*(G23_O_metrics['pre']*G23_O_metrics['recall']) / (G23_O_metrics['pre']+G23_O_metrics['recall']+0.000001)
    G23_A_metrics['f1'] = 2*(G23_A_metrics['pre']*G23_A_metrics['recall']) / (G23_A_metrics['pre']+G23_A_metrics['recall']+0.000001)
    G4_A_metrics['f1'] = 2*(G4_A_metrics['pre']*G4_A_metrics['recall']) / (G4_A_metrics['pre']+G4_A_metrics['recall']+0.000001)
    GBM_metrics['f1'] = 2*(GBM_metrics['pre']*GBM_metrics['recall']) / (GBM_metrics['pre']+GBM_metrics['recall']+0.000001)
    all_metrics['f1'] = (G23_O_metrics['f1'] + G23_A_metrics['f1'] + G4_A_metrics['f1'] + GBM_metrics['f1']) / 4

    list_Diag = (Acc_Diag, 0, all_metrics['f1'], all_metrics['sen'], all_metrics['spec'], all_metrics['AUC'], all_metrics['pre'])

    return list_IDH,list_1p19q,list_CDKN,list_His,list_His_2class,list_Diag



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





























