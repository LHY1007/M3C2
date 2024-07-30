# from apex import amp
from utils import *
import dataset_mine
from net import init_weights, get_scheduler, WarmupCosineSchedule
import sklearn


def train(opt):
    root_IDH = r'/home/hanyu/LHY/miccai7.26-vis miccai/best_model/MICCAI-0012.pth'
    root_1p19q = root_IDH
    root_CDKN = root_IDH
    root_His = root_IDH
    opt['gpus'] = [0,1]
    gpuID = opt['gpus']
    opt['batchSize'] = 1

    TestDataset = dataset_mine.Our_Dataset(phase='Test', opt=opt)
    TestLoader = DataLoader(TestDataset, batch_size=opt['Val_batchSize'],
                            num_workers=opt['nThreads'] if (sysstr == "Linux") else 1, shuffle=False)
    ItestDataset = dataset_mine.Our_Dataset(phase='ITest', opt=opt)
    ItestLoader = DataLoader(ItestDataset, batch_size=opt['Val_batchSize'],
                             num_workers=opt['nThreads'] if (sysstr == "Linux") else 1, shuffle=False)

    ############## initialize #######################

    last_ep = 0
    total_it = 0
    saver = Saver(opt)
    print('%d epochs and %d iterations has been trained' % (last_ep, total_it))
    alleps = opt['n_ep'] - last_ep
    curep = 0
    if 1:
        ## IDH
        IDH_model_init = Mine_init1(opt).cuda(gpuID[0])#

        IDH_model_IDH = Mine_IDH(opt).cuda(gpuID[0])
        IDH_model_1p19q = Mine_1p19q(opt).cuda(gpuID[0])
        IDH_model_CDKN = Mine_CDKN(opt).cuda(gpuID[0])
        IDH_model_Graph = Label_correlation_Graph(opt).cuda(gpuID[0])
        IDH_model_init = torch.nn.DataParallel(IDH_model_init, device_ids=gpuID)
        IDH_model_IDH = torch.nn.DataParallel(IDH_model_IDH, device_ids=gpuID)
        IDH_model_1p19q = torch.nn.DataParallel(IDH_model_1p19q, device_ids=gpuID)
        IDH_model_CDKN = torch.nn.DataParallel(IDH_model_CDKN, device_ids=gpuID)
        IDH_model_Graph = torch.nn.DataParallel(IDH_model_Graph, device_ids=gpuID)

        ckptdir = os.path.join(root_IDH)#
        checkpoint = torch.load(ckptdir)#
        related_params = {k: v for k, v in checkpoint['init'].items()}#

        IDH_model_init.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['IDH'].items()}
        IDH_model_IDH.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['1p19q'].items()}
        IDH_model_1p19q.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['CDKN'].items()}
        IDH_model_CDKN.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['Graph'].items()}
        IDH_model_Graph.load_state_dict(related_params, strict=False)

        IDH_model_init.eval()#
        IDH_model_IDH.eval()
        IDH_model_1p19q.eval()
        IDH_model_CDKN.eval()
        IDH_model_Graph.eval()
        IDH_model = [IDH_model_init, IDH_model_IDH, IDH_model_1p19q, IDH_model_CDKN, IDH_model_Graph]

        ## 1p19q
        p19q_model_init = Mine_init1(opt).cuda(gpuID[0])#
        p19q_model_IDH = Mine_IDH(opt).cuda(gpuID[0])
        p19q_model_1p19q = Mine_1p19q(opt).cuda(gpuID[0])
        p19q_model_CDKN = Mine_CDKN(opt).cuda(gpuID[0])
        p19q_model_Graph = Label_correlation_Graph(opt).cuda(gpuID[0])
        p19q_model_init = torch.nn.DataParallel(p19q_model_init, device_ids=gpuID)
        p19q_model_IDH = torch.nn.DataParallel(p19q_model_IDH, device_ids=gpuID)
        p19q_model_1p19q = torch.nn.DataParallel(p19q_model_1p19q, device_ids=gpuID)
        p19q_model_CDKN = torch.nn.DataParallel(p19q_model_CDKN, device_ids=gpuID)
        p19q_model_Graph = torch.nn.DataParallel(p19q_model_Graph, device_ids=gpuID)

        ckptdir = os.path.join(root_1p19q)#
        checkpoint = torch.load(ckptdir)#
        related_params = {k: v for k, v in checkpoint['init'].items()}#
        p19q_model_init.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['IDH'].items()}
        p19q_model_IDH.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['1p19q'].items()}
        p19q_model_1p19q.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['CDKN'].items()}
        p19q_model_CDKN.load_state_dict(related_params)
        related_params = {k: v for k, v in checkpoint['Graph'].items()}
        p19q_model_Graph.load_state_dict(related_params, strict=False)

        p19q_model_init.eval()#
        p19q_model_IDH.eval()
        p19q_model_1p19q.eval()
        p19q_model_CDKN.eval()
        p19q_model_Graph.eval()
        p19q_model = [p19q_model_init, p19q_model_IDH, p19q_model_1p19q, p19q_model_CDKN, p19q_model_Graph]

        ## CDKN
        CDKN_model_init = Mine_init1(opt).cuda(gpuID[0])
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
        CDKN_model_Graph.load_state_dict(related_params, strict=False)

        CDKN_model_init.eval()
        CDKN_model_IDH.eval()
        CDKN_model_1p19q.eval()
        CDKN_model_CDKN.eval()
        CDKN_model_Graph.eval()
        CDKN_model = [CDKN_model_init, CDKN_model_IDH, CDKN_model_1p19q, CDKN_model_CDKN, CDKN_model_Graph]

        ## His
        His_model_init = Mine_init1(opt).cuda(gpuID[0])
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
    opt['name'] = 'MICCAI_IN'
    list_WSI_IDH, list_WSI_1p19q, list_WSI_CDKN, list_WSI_His_2class, list_WSI_Diag = validation_All_sepe(
        opt, IDH_model, p19q_model, CDKN_model, His_model, TestLoader, gpuID, task='')
    print(
        'val in epoch: %d/%d, acc_IDH:%.3f,acc_1p19q:%.3f,acc_CDKN:%.3f,acc_His_2class:%.3f,acc_Diag:%.3f' % (
            0 + 1, alleps, list_WSI_IDH[0], list_WSI_1p19q[0], list_WSI_CDKN[0], list_WSI_His_2class[0],
            list_WSI_Diag[0]))

    # print("----------ITEST-------------")
    # opt['name'] = 'MICCAI_EX'
    # list_WSI_IDH, list_WSI_1p19q, list_WSI_CDKN, list_WSI_His_2class, list_WSI_Diag = validation_All_sepe(
    #     opt, IDH_model, p19q_model, CDKN_model, His_model, ItestLoader, gpuID, task='Itest')
    # print(
    #     'val in epoch: %d/%d, acc_IDH:%.3f,acc_1p19q:%.3f,acc_CDKN:%.3f,acc_His_2class:%.3f,acc_Diag:%.3f' % (
    #         0 + 1, alleps, list_WSI_IDH[0], list_WSI_1p19q[0], list_WSI_CDKN[0], list_WSI_His_2class[0],
    #         list_WSI_Diag[0]))


def validation_All_sepe(opt, IDH_model, p19q_model, CDKN_model, His_model, dataloader, gpuID, task=''):
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
        fp_His_2class = 0
        fn_His_2class = 0
        label_all_His_2class = []
        predicted_all_His_2class = []

        count_Diag = 0
        correct_Diag = 0
        G23_O_metrics_Diag = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'sen_weight': 0, 'sen_micro': 0, 'spec': 0,
                              'spec_weight': 0, 'spec_micro': 0, 'pre': 0, 'pre_weight': 0, 'pre_micro': 0, 'recall': 0,
                              'recall_weight': 0, 'recall_micro': 0, 'f1': 0, 'f1_weight': 0, 'f1_micro': 0, 'AUC': 0}
        G23_A_metrics_Diag = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'sen_weight': 0, 'sen_micro': 0, 'spec': 0,
                              'spec_weight': 0, 'spec_micro': 0, 'pre': 0, 'pre_weight': 0, 'pre_micro': 0, 'recall': 0,
                              'recall_weight': 0, 'recall_micro': 0, 'f1': 0, 'f1_weight': 0, 'f1_micro': 0, 'AUC': 0}
        G4_A_metrics_Diag = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'sen_weight': 0, 'sen_micro': 0, 'spec': 0,
                             'spec_weight': 0, 'spec_micro': 0, 'pre': 0, 'pre_weight': 0, 'pre_micro': 0, 'recall': 0,
                             'recall_weight': 0, 'recall_micro': 0, 'f1': 0, 'f1_weight': 0, 'f1_micro': 0, 'AUC': 0}
        GBM_metrics_Diag = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'sen_weight': 0, 'sen_micro': 0, 'spec': 0,
                            'spec_weight': 0, 'spec_micro': 0, 'pre': 0, 'pre_weight': 0, 'pre_micro': 0, 'recall': 0,
                            'recall_weight': 0, 'recall_micro': 0, 'f1': 0, 'f1_weight': 0, 'f1_micro': 0, 'AUC': 0}
        all_metrics_Diag = {'sen': 0, 'sen_weight': 0, 'sen_micro': 0, 'spec': 0, 'spec_weight': 0, 'spec_micro': 0,
                            'pre': 0, 'pre_weight': 0, 'pre_micro': 0, 'recall': 0, 'recall_weight': 0,
                            'recall_micro': 0, 'f1': 0, 'f1_weight': 0, 'f1_micro': 0, 'AUC': 0}
        label_all_Diag = []
        predicted_all_Diag = []
        pred_all_Diag = []

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
    test_bar = tqdm(dataloader)
    bs = 1
    count = 0
    for packs in test_bar:
        img20, img10, label = packs

        if torch.cuda.is_available():
            img20 = img20.cuda(gpuID[0])
            img10 = img10.cuda(gpuID[0])

            # label = label.cuda(gpuID[0])
        # #  # # IDH
        '''
        pred_IDH_ori, _, _, _, _ = IDH_model(img20, img10, label)
        # #  # # 1p19q
        _, pred_1p19q_ori, _, _, _ = p19q_model(img20, img10, label)
        # #  # # CDKN
        _, _, pred_CDKN_ori, _, _ = CDKN_model(img20, img10, label)
        # #  # # His
        _, _, _, pred_His_2class_ori, _ = His_model(img20, img10, label)
        # #  # # Task
        _, _, _, _, pred_Task_ori = Task_model(img20, img10, label)
        '''
        #init_feature_his, init_feature_mark, _, _, _, _ = IDH_model[0](img20, img10)  # (BS,2500,1024)#####################
        init_feature=IDH_model[0](img20,img10) # (BS,2500,1024)
        hidden_states, encoded_IDH = IDH_model[1](init_feature)
        hidden_states, encoded_1p19q = IDH_model[2](hidden_states)
        encoded_CDKN = IDH_model[3](hidden_states)
        results_dict, _, _, _, _, _, _, _, Mark_output = IDH_model[4](encoded_IDH, encoded_1p19q, encoded_CDKN)
        pred_IDH_ori = results_dict['logits_IDH']

        # #  # # 1p19q

        #init_feature_his, init_feature_mark, _, _, _, _ = p19q_model[0](img20, img10)  # (BS,2500,1024)#####################
        init_feature=p19q_model[0](img20,img10) # (BS,2500,1024)
        hidden_states, encoded_IDH = p19q_model[1](init_feature)
        hidden_states, encoded_1p19q = p19q_model[2](hidden_states)
        encoded_CDKN = p19q_model[3](hidden_states)
        results_dict, _, _, _, _, _, _, _, Mark_output = p19q_model[4](encoded_IDH, encoded_1p19q, encoded_CDKN)
        pred_1p19q_ori = results_dict['logits_1p19q']

        # #  # # CDKN
        #init_feature_his, init_feature_mark, _, _, _, _ = CDKN_model[0](img20, img10)  # (BS,2500,1024)#####################
        init_feature=CDKN_model[0](img20,img10) # (BS,2500,1024)
        hidden_states, encoded_IDH = CDKN_model[1](init_feature)
        hidden_states, encoded_1p19q = CDKN_model[2](hidden_states)
        encoded_CDKN = CDKN_model[3](hidden_states)
        results_dict, _, _, _, _, _, _, _, Mark_output = CDKN_model[4](encoded_IDH, encoded_1p19q, encoded_CDKN)
        pred_CDKN_ori = results_dict['logits_CDKN']

        # #  # # His
        #init_feature_his, init_feature_mark, _, _, _, _ = His_model[0](img20, img10)  # (BS,2500,1024)####################
        init_feature=His_model[0](img20,img10) # (BS,2500,1024)
        hidden_states, encoded_His = His_model[1](init_feature)
        results_dict, _, __, __, His_output = His_model[2](encoded_His)
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
        '''
        _, pred_Task = torch.max(pred_Task_ori.data, 1)
        pred_Task = pred_Task.tolist()
        gt_Task = label[:, 5].tolist()
        '''

        gt_Diag = label[:, 5].tolist()
        pred_Diag = Diag_predict(pred_IDH_ori, pred_1p19q_ori, pred_CDKN_ori, pred_His_2class_ori)
        pred_Diag_ori = torch.zeros(1, 4).cuda(gpuID[0])

        pred_IDH_ori_softmax = F.softmax(pred_IDH_ori, dim=1)
        pred_1p19q_ori_softmax = F.softmax(pred_1p19q_ori, dim=1)
        pred_CDKN_ori_softmax = F.softmax(pred_CDKN_ori, dim=1)
        pred_His_2class_ori_softmax = F.softmax(pred_His_2class_ori, dim=1)


        pred_Diag_ori[0, 0] = pred_IDH_ori_softmax[0, 0]
        pred_Diag_ori[0, 1] = pred_IDH_ori_softmax[0, 1] * pred_1p19q_ori_softmax[0, 0] * (
                    1 - pred_CDKN_ori_softmax[0, 0] * (1 - pred_His_2class_ori_softmax[0, 1]))
        pred_Diag_ori[0, 2] = pred_IDH_ori_softmax[0, 1] * pred_1p19q_ori_softmax[0, 0] * pred_CDKN_ori_softmax[
            0, 0] * (1 - pred_His_2class_ori_softmax[0, 1])
        pred_Diag_ori[0, 3] = pred_IDH_ori_softmax[0, 1] * pred_1p19q_ori_softmax[0, 1]

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
#######################################################
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
            #############Task
            '''
            label_all_Task.append(gt_Task[j])
            pred_all_Task.append(pred_Task[j])
            pred_Task_ori_softmax = F.softmax(pred_Task_ori, dim=-1)
            predicted_all_Task.append(pred_Task_ori_softmax.detach().cpu().numpy()[0])
            if gt_Task[j] == 0:
                if pred_Task[j] == 0:
                    correct_Task+=1
                    GBM_metrics_Task['tp'] += 1
                else:
                    GBM_metrics_Task['fn'] += 1
            else:
                if not pred_Task[j] == 0:
                    GBM_metrics_Task['tn'] += 1
                else:
                    GBM_metrics_Task['fp'] += 1
            # G234_A
            if gt_Task[j] == 1:
                if pred_Task[j] == 1:
                    correct_Task+=1
                    G4_A_metrics_Task['tp'] += 1
                else:
                    G4_A_metrics_Task['fn'] += 1
            else:
                if not pred_Task[j] == 1:
                    G4_A_metrics_Task['tn'] += 1
                else:
                    G4_A_metrics_Task['fp'] += 1
            if gt_Task[j] == 2:
                if pred_Task[j] == 2:
                    correct_Task+=1
                    G23_A_metrics_Task['tp'] += 1
                else:
                    G23_A_metrics_Task['fn'] += 1
            else:
                if not pred_Task[j] == 2:
                    G23_A_metrics_Task['tn'] += 1
                else:
                    G23_A_metrics_Task['fp'] += 1
            # G23_O
            if gt_Task[j] == 3:
                if pred_Task[j] == 3:
                    correct_Task+=1
                    G23_O_metrics_Task['tp'] += 1
                else:
                    G23_O_metrics_Task['fn'] += 1
            else:
                if not pred_Task[j] == 3:
                    G23_O_metrics_Task['tn'] += 1
                else:
                    G23_O_metrics_Task['fp'] += 1
            
            count_Task+=1
            '''
    if task != 'Itest':
        ##########################################   IDH
        Acc_IDH = (tp_IDH + tn_IDH) / (tp_IDH + tn_IDH + fp_IDH + fn_IDH)
        Sen_IDH = (tp_IDH) / (tp_IDH + fn_IDH + 0.000001)  # recall
        Spec_IDH = (tn_IDH) / (tn_IDH + fp_IDH + 0.000001)
        precision_IDH = (tp_IDH) / (tp_IDH + fp_IDH + 0.000001)
        recall_IDH = Sen_IDH
        f1_score_IDH = (2 * precision_IDH * recall_IDH) / (precision_IDH + recall_IDH + 0.000001)
        AUC_IDH = metrics.roc_auc_score(y_true=np.array(label_all_IDH), y_score=np.array(predicted_all_IDH))
        dataframe = pd.DataFrame({'label': label_all_IDH, 'score': predicted_all_IDH, 'sen': Sen_IDH, 'spec': Spec_IDH})
        dataframe.to_excel('plot/' + opt['name'] + '_IDH' + '.xlsx', index=False)
        list_IDH = (Acc_IDH, None, f1_score_IDH, Sen_IDH, Spec_IDH, AUC_IDH, precision_IDH)

        ##########################################   1p19q
        Acc_1p19q = (tp_1p19q + tn_1p19q) / (tp_1p19q + tn_1p19q + fp_1p19q + fn_1p19q)
        Sen_1p19q = (tp_1p19q) / (tp_1p19q + fn_1p19q + 0.000001)  # recall
        Spec_1p19q = (tn_1p19q) / (tn_1p19q + fp_1p19q + 0.000001)
        precision_1p19q = (tp_1p19q) / (tp_1p19q + fp_1p19q + 0.000001)
        recall_1p19q = Sen_1p19q
        f1_score_1p19q = (2 * precision_1p19q * recall_1p19q) / (precision_1p19q + recall_1p19q + 0.000001)
        AUC_1p19q = metrics.roc_auc_score(y_true=np.array(label_all_1p19q), y_score=np.array(predicted_all_1p19q))
        dataframe = pd.DataFrame(
            {'label': label_all_1p19q, 'score': predicted_all_1p19q, 'sen': Sen_1p19q, 'spec': Spec_1p19q})
        dataframe.to_excel('plot/' + opt['name'] + '_1p19q' + '.xlsx', index=False)

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
            dataframe = pd.DataFrame(
                {'label': label_all_CDKN, 'score': predicted_all_CDKN, 'sen': Sen_CDKN, 'spec': Spec_CDKN})
            dataframe.to_excel('plot/' + opt['name'] + '_CDKN' + '.xlsx', index=False)

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
        dataframe = pd.DataFrame(
            {'label': label_all_His_2class, 'score': predicted_all_His_2class, 'sen': Sen_His_2class,
            'spec': Spec_His_2class})
        dataframe.to_excel('plot/' + opt['name'] + '_His_2class' + '.xlsx', index=False)
        list_His_2class = (
        Acc_His_2class, None, f1_score_His_2class, Sen_His_2class, Spec_His_2class, AUC_His_2class, precision_His_2class)
    ################################################   Diag
    else:
        list_IDH = (
        0, None, 0, 0, 0, 0, 0)
        list_1p19q = (
        0, None, 0, 0, 0, 0, 0)
        list_CDKN = (
        0, None, 0, 0, 0, 0, 0)
        list_His_2class = (
        0, None, 0, 0, 0, 0, 0)
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
    all_metrics_Diag['f1'] = (G23_O_metrics_Diag['f1'] + G23_A_metrics_Diag['f1'] + G4_A_metrics_Diag['f1'] +
                              GBM_metrics_Diag['f1']) / 4


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
        all_metrics_Diag['AUC_weight'] = sum([roc_auc[i] * weights[i] for i in range(4)]) / total_samples + 0.000001
    else:
        weights = [np.sum(np.array(label_all_Diag) == i) for i in range(2)]
        total_samples = np.sum(weights)
        all_metrics_Diag['AUC_weight'] = sum([roc_auc[i] * weights[i] for i in range(2)]) / total_samples + 0.000001
    dataframe = pd.DataFrame(
        {'label': label_all_Diag, 'score': predicted_all_Diag, 'sen': all_metrics_Diag['sen_micro'],
         'spec': all_metrics_Diag['spec_micro']})
    dataframe.to_excel('plot/' + opt['name'] + '_Diag' + '.xlsx', index=False)
    # print(all_metrics_Diag['AUC_micro'])
    import sklearn
    
    f1 =sklearn.metrics.f1_score(label_all_Diag, pred_all_Diag, labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn')
    #print(f1)
    all_metrics_Diag['f1_micro'] = f1
    list_Diag = (
    Acc_Diag, 0, all_metrics_Diag['f1_micro'], all_metrics_Diag['sen_micro'], all_metrics_Diag['spec_micro'],
    all_metrics_Diag['AUC_micro'], all_metrics_Diag['pre_micro'])
    ###############Task
    # Acc_Task = correct_Task / count_Task
    # # Sensitivity
    # G23_O_metrics_Task['sen'] = (G23_O_metrics_Task['tp']) / (
    #             G23_O_metrics_Task['tp'] + G23_O_metrics_Task['fn'] + 0.000001)
    # G23_A_metrics_Task['sen'] = (G23_A_metrics_Task['tp']) / (
    #             G23_A_metrics_Task['tp'] + G23_A_metrics_Task['fn'] + 0.000001)
    # G4_A_metrics_Task['sen'] = (G4_A_metrics_Task['tp']) / (
    #             G4_A_metrics_Task['tp'] + G4_A_metrics_Task['fn'] + 0.000001)
    # GBM_metrics_Task['sen'] = (GBM_metrics_Task['tp']) / (GBM_metrics_Task['tp'] + GBM_metrics_Task['fn'] + 0.000001)
    # all_metrics_Task['sen'] = (G23_O_metrics_Task['sen'] + G23_A_metrics_Task['sen'] + G4_A_metrics_Task['sen'] +
    #                            GBM_metrics_Task['sen']) / 4

    # label_all_Task = list(label_all_Task)
    # all_metrics_Task['sen_micro'] =  GBM_metrics_Task['sen'] * label_all_Task.count(0) / len(label_all_Task) + \
    # G4_A_metrics_Task['sen'] * label_all_Task.count(1) / len(label_all_Task) + \
    # G23_A_metrics_Task['sen'] * label_all_Task.count(2) / len(label_all_Task) + \
    # G23_O_metrics_Task['sen'] * label_all_Task.count(3) / len(label_all_Task)
    # #print("sen_micro ",all_metrics_Task['sen_micro'])

    # # Specificity
    # G23_O_metrics_Task['spec'] = (G23_O_metrics_Task['tn']) / (
    #             G23_O_metrics_Task['tn'] + G23_O_metrics_Task['fp'] + 0.000001)
    # G23_A_metrics_Task['spec'] = (G23_A_metrics_Task['tn']) / (
    #             G23_A_metrics_Task['tn'] + G23_A_metrics_Task['fp'] + 0.000001)
    # G4_A_metrics_Task['spec'] = (G4_A_metrics_Task['tn']) / (
    #             G4_A_metrics_Task['tn'] + G4_A_metrics_Task['fp'] + 0.000001)
    # GBM_metrics_Task['spec'] = (GBM_metrics_Task['tn']) / (GBM_metrics_Task['tn'] + GBM_metrics_Task['fp'] + 0.000001)
    # all_metrics_Task['spec'] = (G23_O_metrics_Task['spec'] + G23_A_metrics_Task['spec'] + G4_A_metrics_Task['spec'] +
    #                             GBM_metrics_Task['spec']) / 4
    
    # all_metrics_Task['spec_micro'] =  GBM_metrics_Task['spec'] * label_all_Task.count(0) / len(label_all_Task) + \
    # G4_A_metrics_Task['spec'] * label_all_Task.count(1) / len(label_all_Task) + \
    # G23_A_metrics_Task['spec'] * label_all_Task.count(2) / len(label_all_Task) + \
    # G23_O_metrics_Task['spec'] * label_all_Task.count(3) / len(label_all_Task)

    # # Precision
    # G23_O_metrics_Task['pre'] = (G23_O_metrics_Task['tp']) / (
    #             G23_O_metrics_Task['tp'] + G23_O_metrics_Task['fp'] + 0.000001)
    # G23_A_metrics_Task['pre'] = (G23_A_metrics_Task['tp']) / (
    #             G23_A_metrics_Task['tp'] + G23_A_metrics_Task['fp'] + 0.000001)
    # G4_A_metrics_Task['pre'] = (G4_A_metrics_Task['tp']) / (
    #             G4_A_metrics_Task['tp'] + G4_A_metrics_Task['fp'] + 0.000001)
    # GBM_metrics_Task['pre'] = (GBM_metrics_Task['tp']) / (GBM_metrics_Task['tp'] + GBM_metrics_Task['fp'] + 0.000001)
    # all_metrics_Task['pre'] = (G23_O_metrics_Task['pre'] + G23_A_metrics_Task['pre'] + G4_A_metrics_Task['pre'] +
    #                            GBM_metrics_Task['pre']) / 4
    # all_metrics_Task['pre_micro'] =  GBM_metrics_Task['pre'] * label_all_Task.count(0) / len(label_all_Task) + \
    # G4_A_metrics_Task['pre'] * label_all_Task.count(1) / len(label_all_Task) + \
    # G23_A_metrics_Task['pre'] * label_all_Task.count(2) / len(label_all_Task) + \
    # G23_O_metrics_Task['pre'] * label_all_Task.count(3) / len(label_all_Task)

    # # Recall
    # G23_O_metrics_Task['recall'] = (G23_O_metrics_Task['tp']) / (
    #             G23_O_metrics_Task['tp'] + G23_O_metrics_Task['fn'] + 0.000001)
    # G23_A_metrics_Task['recall'] = (G23_A_metrics_Task['tp']) / (
    #             G23_A_metrics_Task['tp'] + G23_A_metrics_Task['fn'] + 0.000001)
    # G4_A_metrics_Task['recall'] = (G4_A_metrics_Task['tp']) / (
    #             G4_A_metrics_Task['tp'] + G4_A_metrics_Task['fn'] + 0.000001)
    # GBM_metrics_Task['recall'] = (GBM_metrics_Task['tp']) / (GBM_metrics_Task['tp'] + GBM_metrics_Task['fn'] + 0.000001)
    # all_metrics_Task['recall'] = (G23_O_metrics_Task['recall'] + G23_A_metrics_Task['recall'] + G4_A_metrics_Task[
    #     'recall'] + GBM_metrics_Task['recall']) / 4
    # all_metrics_Task['recall_micro'] =  GBM_metrics_Task['recall'] * label_all_Task.count(0) / len(label_all_Task) + \
    # G4_A_metrics_Task['recall'] * label_all_Task.count(1) / len(label_all_Task) + \
    # G23_A_metrics_Task['recall'] * label_all_Task.count(2) / len(label_all_Task) + \
    # G23_O_metrics_Task['recall'] * label_all_Task.count(3) / len(label_all_Task)

    # # F1 Score
    # G23_O_metrics_Task['f1'] = 2 * (G23_O_metrics_Task['pre'] * G23_O_metrics_Task['recall']) / (
    #             G23_O_metrics_Task['pre'] + G23_O_metrics_Task['recall'] + 0.000001)
    # G23_A_metrics_Task['f1'] = 2 * (G23_A_metrics_Task['pre'] * G23_A_metrics_Task['recall']) / (
    #             G23_A_metrics_Task['pre'] + G23_A_metrics_Task['recall'] + 0.000001)
    # G4_A_metrics_Task['f1'] = 2 * (G4_A_metrics_Task['pre'] * G4_A_metrics_Task['recall']) / (
    #             G4_A_metrics_Task['pre'] + G4_A_metrics_Task['recall'] + 0.000001)
    # GBM_metrics_Task['f1'] = 2 * (GBM_metrics_Task['pre'] * GBM_metrics_Task['recall']) / (
    #             GBM_metrics_Task['pre'] + GBM_metrics_Task['recall'] + 0.000001)
    # # all_metrics_Task['f1'] = (G23_O_metrics_Task['f1'] + G23_A_metrics_Task['f1'] + G4_A_metrics_Task['f1'] +
    # #                           GBM_metrics_Task['f1']) / 4
    # # all_metrics_Task['f1_micro'] =  GBM_metrics_Task['f1'] * label_all_Task.count(0) / len(label_all_Task) + \
    # # G4_A_metrics_Task['f1'] * label_all_Task.count(1) / len(label_all_Task) + \
    # # G23_A_metrics_Task['f1'] * label_all_Task.count(2) / len(label_all_Task) + \
    # # G23_O_metrics_Task['f1'] * label_all_Task.count(3) / len(label_all_Task)
    # import sklearn
    
    # f1 =sklearn.metrics.f1_score(label_all_Task, pred_all_Task, labels=None, pos_label=1, average='micro', sample_weight=None, zero_division='warn')
    # #print(f1)
    # all_metrics_Task['f1_micro'] = f1
    # if task != 'Itest':
    #     roc_auc = calculate_auc(predicted_all_Task, label_all_Task, num_classes=4)
    #     all_metrics_Task['AUC'] = roc_auc["macro"]
    #     all_metrics_Task['AUC_micro'] = roc_auc["micro"]
    #     GBM_metrics_Task['AUC'] = roc_auc[0]
    #     G4_A_metrics_Task['AUC'] = roc_auc[1]
    #     G23_A_metrics_Task['AUC'] = roc_auc[2]
    #     G23_O_metrics_Task['AUC'] = roc_auc[3]
    # else:
    #     roc_auc = calculate_auc(predicted_all_Task, label_all_Task, num_classes=2)
    #     all_metrics_Task['AUC'] = roc_auc["macro"]
    #     all_metrics_Task['AUC_micro'] = roc_auc["micro"]
    #     GBM_metrics_Task['AUC'] = roc_auc[0]
    #     G4_A_metrics_Task['AUC'] = roc_auc[1]
    #     G23_A_metrics_Task['AUC'] = 0
    #     G23_O_metrics_Task['AUC'] = 0

    # # Calculate weighted AUC
    # if task != 'Itest':
    #     weights = [np.sum(np.array(label_all_Task) == i) for i in range(4)]
    #     total_samples = np.sum(weights)
    #     all_metrics_Task['AUC_weight'] = sum([roc_auc[i] * weights[i] for i in range(4)]) / (total_samples + 0.000001)
    # else:
    #     weights = [np.sum(np.array(label_all_Task) == i) for i in range(2)]
    #     total_samples = np.sum(weights)
    #     all_metrics_Task['AUC_weight'] = sum([roc_auc[i] * weights[i] for i in range(2)]) / (total_samples + 0.000001)

    # list_Task = (
    # Acc_Task, 0, all_metrics_Task['f1_micro'], all_metrics_Task['sen_micro'], all_metrics_Task['spec_micro'],
    # all_metrics_Task['AUC_micro'], all_metrics_Task['pre_micro'])
    # dataframe = pd.DataFrame(
    # {'label': label_all_Task, 'score': predicted_all_Task, 'sen': all_metrics_Task['sen_micro'],
    #     'spec': all_metrics_Task['spec_micro']})
    # dataframe.to_excel('plot/' + opt['name'] + '_Task' + '.xlsx', index=False)


    return list_IDH, list_1p19q, list_CDKN, list_His_2class, list_Diag


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

    a = 1





























