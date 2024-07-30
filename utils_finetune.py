import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch
import os
import torch.cuda
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from scipy import interp
# from utils import saliency_map_read, make_one_hot,imp_gene,gene_DiagSim_ori,\
#     saliency_comparison,saliency_map_read_stage2,Diag_full,Diag_Simple,gene_Diag_ori
import sys
def test_stage2(opt, Mine_model_init, Mine_model_His, Mine_model_Cls,Mine_model_molecular, Mine_model_Graph, dataloader, gpuID,external=False,flag_tiantan=False):
    Mine_model_init.eval()
    Mine_model_His.eval()
    Mine_model_Cls.eval()
    Mine_model_molecular.eval()
    Mine_model_Graph.eval()
    gpuID = opt['gpus']

    if 1:
        Acc_IDH=0
        Acc_1p19q=0
        Acc_CDKN=0
        Acc_IDH_patient=0
        Acc_1p19q_patient=0
        Acc_CDKN_patient=0
        count_IDH = 0
        count_1p19q = 0
        count_CDKN = 0
        count_Diag = 0
        count_Diag_Sim = 0
        correct_IDH = 0
        correct_1p19q = 0
        correct_CDKN = 0
        correct_Diag = 0
        correct_Diag_Sim = 0
        count_IDH_patient = 0
        count_1p19q_patient = 0
        count_CDKN_patient = 0
        count_Diag_patient = 0
        count_DiagSim_patient = 0
        correct_IDH_patient = 0
        correct_1p19q_patient = 0
        correct_CDKN_patient = 0
        correct_Diag_patient = 0
        correct_DiagSim_patient = 0
        IDH_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        p19q_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        CDKN_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        Diag_GBM = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        Diag_G4A = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        Diag_G3A = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        Diag_G2A = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        Diag_G3O = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        Diag_G2O = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        Diag_all = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        DiagSim_GBM = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        DiagSim_G4A = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        DiagSim_G23A = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        DiagSim_G23O = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        DiagSim_all = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}

        IDH_metrics_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        p19q_metrics_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        CDKN_metrics_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        Diag_GBM_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        Diag_G4A_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        Diag_G3A_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        Diag_G2A_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        Diag_G3O_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        Diag_G2O_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        Diag_all_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        DiagSim_GBM_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        DiagSim_G4A_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        DiagSim_G23A_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        DiagSim_G23O_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        DiagSim_all_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}

        A_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                     'AUC': 0}
        O_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                     'AUC': 0}
        GBM_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                       'AUC': 0}
        all_metrics_His = {'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        label_all_His = []
        predicted_all_His = []
        count_His = 0
        count_Grade = 0
        correct_His = 0

        label_all_IDH = []
        predicted_all_IDH = []
        label_all_1p19q = []
        predicted_all_1p19q = []
        label_all_CDKN = []
        predicted_all_CDKN = []
        label_all_Diag = []
        predicted_all_Diag = []
        label_all_DiagSim = []
        predicted_all_DiagSim = []

        label_all_IDH_patient = []
        predicted_all_IDH_patient = []
        label_all_1p19q_patient = []
        predicted_all_1p19q_patient = []
        label_all_CDKN_patient = []
        predicted_all_CDKN_patient = []
        label_all_Diag_patient = []
        predicted_all_Diag_patient = []
        label_all_DiagSim_patient = []
        predicted_all_DiagSim_patient = []

    test_bar = tqdm(dataloader)
    Patient_predori_Diag = {}
    Patient_predori_DiagSim = {}
    Patient_predori_IDH = {}
    Patient_predori_1p19q = {}
    Patient_predori_CDKN = {}
    for packs in test_bar:
        img = packs[0]
        label = packs[1]
        file_name = packs[2]
        patient_name = packs[3][0]
        if torch.cuda.is_available():
            img = img.cuda(gpuID[0])
            label = label.cuda(gpuID[0])
        label_his = label[:, 0]
        label_grade = label[:, 1]
        label_IDH = label[:, 2]
        label_1p19q = label[:, 3]
        label_CDKN = label[:, 4]
        label_Diag_simple = label[:, 5]
        label_Diag = label[:, 6]

        saliency_map_His, saliency_map_Grade = saliency_map_read_stage2(opt, file_name)
        saliency_map_His = torch.from_numpy(np.array(saliency_map_His)).float().cuda(gpuID[0])
        saliency_map_Grade = torch.from_numpy(np.array(saliency_map_Grade)).float().cuda(gpuID[0])
        init_feature = Mine_model_init(img)  # (BS,2500,1024)
        hidden_states_his, hidden_states_grade, encoded_His, encoded_Grade = Mine_model_His(init_feature,
                                                                                            saliency_map_His,
                                                                                            saliency_map_Grade)
        results_dict, saliency_A, saliency_O, saliency_GBM, saliency_G2, saliency_G3, saliency_G4 = Mine_model_Cls(
            encoded_His, encoded_Grade)

        ### WHO2007 prediction
        pred_His_ori = results_dict['logits_His']
        pred_Grade_ori = results_dict['logits_Grade']
        _, pred_His0 = torch.max(pred_His_ori.data, 1)
        pred_His = pred_His0.tolist()  # [BS] A  O GBM //0 1 2
        gt_His = label_his.tolist()
        _, pred_Grade0 = torch.max(pred_Grade_ori.data, 1)
        pred_Grade = pred_Grade0.tolist()  # [BS] A  O GBM //0 1 2
        gt_Grade = label_grade.tolist()
        pred_His_ori = F.softmax(pred_His_ori)
        pred_Grade_ori = F.softmax(pred_Grade_ori)

        ### molecular prediction
        encoded_IDH, encoded_1p19q, encoded_CDKN = Mine_model_molecular(init_feature)
        results_dict, saliency_IDH_wt, saliency_1p19q_codel, encoded_IDH0, encoded_1p19q0, encoded_CDKN0 = Mine_model_Graph(
            encoded_IDH, encoded_1p19q, encoded_CDKN)
        pred_IDH_ori = results_dict['logits_IDH']
        pred_1p19q_ori = results_dict['logits_1p19q']
        pred_CDKN_ori = results_dict['logits_CDKN']
        _, pred_IDH0 = torch.max(pred_IDH_ori.data, 1)
        pred_IDH = pred_IDH0.tolist()
        gt_IDH = label_IDH.tolist()
        _, pred_1p19q0 = torch.max(pred_1p19q_ori.data, 1)
        pred_1p19q = pred_1p19q0.tolist()
        gt_1p19q = label_1p19q.tolist()
        _, pred_CDKN0 = torch.max(pred_CDKN_ori.data, 1)
        pred_CDKN = pred_CDKN0.tolist()
        gt_CDKN = label_CDKN.tolist()
        pred_IDH_ori = F.softmax(pred_IDH_ori)
        pred_1p19q_ori = F.softmax(pred_1p19q_ori)
        pred_CDKN_ori = F.softmax(pred_CDKN_ori)

        ### Diag prediction
        gt_Diag = label_Diag.tolist()
        pred_Diag = Diag_full(pred_IDH0, pred_1p19q0, pred_CDKN0, pred_His0, pred_Grade0).tolist()
        pred_Diag_ori = gene_Diag_ori(pred_IDH_ori[0], pred_1p19q_ori[0], pred_CDKN_ori[0], pred_His_ori[0],
                                      pred_Grade_ori[0])

        ### Diag Simple prediction
        gt_DiagSim = label_Diag_simple.tolist()
        pred_DiagSim = Diag_Simple(pred_IDH0, pred_1p19q0, pred_CDKN0, pred_His0).tolist()
        pred_DiagSim_ori = gene_DiagSim_ori(pred_IDH_ori[0], pred_1p19q_ori[0], pred_CDKN_ori[0], pred_His_ori[0])

        ############################ WSI calculate tntp
        ##############subtype
        if gt_His[0] != 3:
            label_all_His.append(gt_His[0])
            predicted_all_His.append(pred_His_ori.detach().cpu().numpy()[0])
            count_His += 1
            if gt_His[0] == pred_His[0]:
                correct_His += 1
            if gt_His[0] == 0:
                if pred_His[0] == 0:
                    A_metrics['tp'] += 1
                else:
                    A_metrics['fn'] += 1
            else:
                if not pred_His[0] == 0:
                    A_metrics['tn'] += 1
                else:
                    A_metrics['fp'] += 1
                # O
            if gt_His[0] == 1:
                if pred_His[0] == 1:
                    O_metrics['tp'] += 1
                else:
                    O_metrics['fn'] += 1
            else:
                if not pred_His[0] == 1:
                    O_metrics['tn'] += 1
                else:
                    O_metrics['fp'] += 1
                # GBM
            if gt_His[0] == 2:
                if pred_His[0] == 2:
                    GBM_metrics['tp'] += 1
                else:
                    GBM_metrics['fn'] += 1
            else:
                if not pred_His[0] == 2:
                    GBM_metrics['tn'] += 1
                else:
                    GBM_metrics['fp'] += 1
        ##############IDH
        if gt_IDH[0] != 2:
            label_all_IDH.append(gt_IDH[0])
            predicted_all_IDH.append(pred_IDH_ori.detach().cpu().numpy()[0][1])
            if gt_IDH[0] == 0 and pred_IDH[0] == 0:
                IDH_metrics['tn'] += 1
            if gt_IDH[0] == 0 and pred_IDH[0] == 1:
                IDH_metrics['fp'] += 1
            if gt_IDH[0] == 1 and pred_IDH[0] == 0:
                IDH_metrics['fn'] += 1
            if gt_IDH[0] == 1 and pred_IDH[0] == 1:
                IDH_metrics['tp'] += 1
        ##############1p19q
        if gt_1p19q[0] != 2 :
            label_all_1p19q.append(gt_1p19q[0])
            predicted_all_1p19q.append(pred_1p19q_ori.detach().cpu().numpy()[0][1])
            if gt_1p19q[0] == 0 and pred_1p19q[0] == 0:
                p19q_metrics['tn'] += 1
            if gt_1p19q[0] == 0 and pred_1p19q[0] == 1:
                p19q_metrics['fp'] += 1
            if gt_1p19q[0] == 1 and pred_1p19q[0] == 0:
                p19q_metrics['fn'] += 1
            if gt_1p19q[0] == 1 and pred_1p19q[0] == 1:
                p19q_metrics['tp'] += 1
        ##############CDKN
        if gt_CDKN[0] != 2:
            label_all_CDKN.append(gt_CDKN[0])
            predicted_all_CDKN.append(pred_CDKN_ori.detach().cpu().numpy()[0][1])
            if gt_CDKN[0] == 0 and pred_CDKN[0] == 0:
                CDKN_metrics['tn'] += 1
            if gt_CDKN[0] == 0 and pred_CDKN[0] == 1:
                CDKN_metrics['fp'] += 1
            if gt_CDKN[0] == 1 and pred_CDKN[0] == 0:
                CDKN_metrics['fn'] += 1
            if gt_CDKN[0] == 1 and pred_CDKN[0] == 1:
                CDKN_metrics['tp'] += 1
        ##############Diag
        if gt_Diag[0] != 6:
            label_all_Diag.append(gt_Diag[0])
            predicted_all_Diag.append(pred_Diag_ori)
            count_Diag += 1
            if gt_Diag[0] == pred_Diag[0]:
                correct_Diag += 1
            # G4 GBM
            if gt_Diag[0] == 0:
                if pred_Diag[0] == 0:
                    Diag_GBM['tp'] += 1
                else:
                    Diag_GBM['fn'] += 1
            else:
                if not pred_Diag[0] == 0:
                    Diag_GBM['tn'] += 1
                else:
                    Diag_GBM['fp'] += 1
            # G4 A
            if gt_Diag[0] == 1:
                if pred_Diag[0] == 1:
                    Diag_G4A['tp'] += 1
                else:
                    Diag_G4A['fn'] += 1
            else:
                if not pred_Diag[0] == 1:
                    Diag_G4A['tn'] += 1
                else:
                    Diag_G4A['fp'] += 1
            # G3 A
            if gt_Diag[0] == 2:
                if pred_Diag[0] == 2:
                    Diag_G3A['tp'] += 1
                else:
                    Diag_G3A['fn'] += 1
            else:
                if not pred_Diag[0] == 2:
                    Diag_G3A['tn'] += 1
                else:
                    Diag_G3A['fp'] += 1
            # G2 A
            if gt_Diag[0] == 3:
                if pred_Diag[0] == 3:
                    Diag_G2A['tp'] += 1
                else:
                    Diag_G2A['fn'] += 1
            else:
                if not pred_Diag[0] == 3:
                    Diag_G2A['tn'] += 1
                else:
                    Diag_G2A['fp'] += 1
            # G3 O
            if gt_Diag[0] == 4:
                if pred_Diag[0] == 4:
                    Diag_G3O['tp'] += 1
                else:
                    Diag_G3O['fn'] += 1
            else:
                if not pred_Diag[0] == 4:
                    Diag_G3O['tn'] += 1
                else:
                    Diag_G3O['fp'] += 1
            # G2 O
            if gt_Diag[0] == 5:
                if pred_Diag[0] == 5:
                    Diag_G2O['tp'] += 1
                else:
                    Diag_G2O['fn'] += 1
            else:
                if not pred_Diag[0] == 5:
                    Diag_G2O['tn'] += 1
                else:
                    Diag_G2O['fp'] += 1

        ##############DiagSim
        if gt_DiagSim[0] != 4:
            label_all_DiagSim.append(gt_DiagSim[0])
            predicted_all_DiagSim.append(pred_DiagSim_ori)
            count_Diag_Sim += 1
            if gt_DiagSim[0] == pred_DiagSim[0]:
                correct_Diag_Sim += 1
            # G4 GBM
            if gt_DiagSim[0] == 0:
                if pred_DiagSim[0] == 0:
                    DiagSim_GBM['tp'] += 1
                else:
                    DiagSim_GBM['fn'] += 1
            else:
                if not pred_DiagSim[0] == 0:
                    DiagSim_GBM['tn'] += 1
                else:
                    DiagSim_GBM['fp'] += 1
            # G4 A
            if gt_DiagSim[0] == 1:
                if pred_DiagSim[0] == 1:
                    DiagSim_G4A['tp'] += 1
                else:
                    DiagSim_G4A['fn'] += 1
            else:
                if not pred_DiagSim[0] == 1:
                    DiagSim_G4A['tn'] += 1
                else:
                    DiagSim_G4A['fp'] += 1
            # G23 A
            if gt_DiagSim[0] == 2:
                if pred_DiagSim[0] == 2:
                    DiagSim_G23A['tp'] += 1
                else:
                    DiagSim_G23A['fn'] += 1
            else:
                if not pred_DiagSim[0] == 2:
                    DiagSim_G23A['tn'] += 1
                else:
                    DiagSim_G23A['fp'] += 1
            # G23 O
            if gt_DiagSim[0] == 3:
                if pred_DiagSim[0] == 3:
                    DiagSim_G23O['tp'] += 1
                else:
                    DiagSim_G23O['fn'] += 1
            else:
                if not pred_DiagSim[0] == 3:
                    DiagSim_G23O['tn'] += 1
                else:
                    DiagSim_G23O['fp'] += 1

        ################################patient
        if gt_Diag[0] != 6:
            if patient_name not in Patient_predori_Diag:
                Patient_predori_Diag[patient_name] = []
                Patient_predori_Diag[patient_name].append(pred_Diag_ori)
                label_all_Diag_patient.append(gt_Diag[0])
                count_Diag_patient += 1
            else:
                Patient_predori_Diag[patient_name].append(pred_Diag_ori)
        if gt_DiagSim[0] != 4:
            if patient_name not in Patient_predori_DiagSim:
                Patient_predori_DiagSim[patient_name] = []
                Patient_predori_DiagSim[patient_name].append(pred_DiagSim_ori)
                label_all_DiagSim_patient.append(gt_DiagSim[0])
                count_DiagSim_patient += 1
            else:
                Patient_predori_DiagSim[patient_name].append(pred_DiagSim_ori)
        if gt_IDH[0] != 2:
            if patient_name not in Patient_predori_IDH:
                Patient_predori_IDH[patient_name] = []
                Patient_predori_IDH[patient_name].append(pred_IDH_ori.detach().cpu().numpy()[0])
                label_all_IDH_patient.append(gt_IDH[0])
                count_IDH_patient += 1
            else:
                Patient_predori_IDH[patient_name].append(pred_IDH_ori.detach().cpu().numpy()[0])
        if gt_1p19q[0] != 2:
            if patient_name not in Patient_predori_1p19q:
                Patient_predori_1p19q[patient_name] = []
                Patient_predori_1p19q[patient_name].append(pred_1p19q_ori.detach().cpu().numpy()[0])
                label_all_1p19q_patient.append(gt_1p19q[0])
                count_1p19q_patient += 1
            else:
                Patient_predori_1p19q[patient_name].append(pred_1p19q_ori.detach().cpu().numpy()[0])
        if gt_CDKN[0] != 2:
            if patient_name not in Patient_predori_CDKN:
                Patient_predori_CDKN[patient_name] = []
                Patient_predori_CDKN[patient_name].append(pred_CDKN_ori.detach().cpu().numpy()[0])
                label_all_CDKN_patient.append(gt_CDKN[0])
                count_CDKN_patient += 1
            else:
                Patient_predori_CDKN[patient_name].append(pred_CDKN_ori.detach().cpu().numpy()[0])

    Acc_His = correct_His / count_His
    ####patient process
    for patient, pred_wsis in Patient_predori_Diag.items():
        predicted_all_Diag_patient.append(np.mean(np.array(Patient_predori_Diag[patient]), axis=0))
    for i in range(len(label_all_Diag_patient)):
        count_Diag_patient += 1
        pred_patient = np.argmax(predicted_all_Diag_patient[i])
        if label_all_Diag_patient[i] == pred_patient:
            correct_Diag_patient += 1
        # GBM
        if label_all_Diag_patient[i] == 0:
            if pred_patient == 0:
                Diag_GBM_patient['tp'] += 1
            else:
                Diag_GBM_patient['fn'] += 1
        else:
            if not pred_patient == 0:
                Diag_GBM_patient['tn'] += 1
            else:
                Diag_GBM_patient['fp'] += 1
        # G4A
        if label_all_Diag_patient[i] == 1:
            if pred_patient == 1:
                Diag_G4A_patient['tp'] += 1
            else:
                Diag_G4A_patient['fn'] += 1
        else:
            if not pred_patient == 1:
                Diag_G4A_patient['tn'] += 1
            else:
                Diag_G4A_patient['fp'] += 1
        # G3A
        if label_all_Diag_patient[i] == 2:
            if pred_patient == 2:
                Diag_G3A_patient['tp'] += 1
            else:
                Diag_G3A_patient['fn'] += 1
        else:
            if not pred_patient == 2:
                Diag_G3A_patient['tn'] += 1
            else:
                Diag_G3A_patient['fp'] += 1
        # G2A
        if label_all_Diag_patient[i] == 3:
            if pred_patient == 3:
                Diag_G2A_patient['tp'] += 1
            else:
                Diag_G2A_patient['fn'] += 1
        else:
            if not pred_patient == 3:
                Diag_G2A_patient['tn'] += 1
            else:
                Diag_G2A_patient['fp'] += 1
        # G3O
        if label_all_Diag_patient[i] == 4:
            if pred_patient == 4:
                Diag_G3O_patient['tp'] += 1
            else:
                Diag_G3O_patient['fn'] += 1
        else:
            if not pred_patient == 4:
                Diag_G3O_patient['tn'] += 1
            else:
                Diag_G3O_patient['fp'] += 1
        # G2O
        if label_all_Diag_patient[i] == 5:
            if pred_patient == 5:
                Diag_G2O_patient['tp'] += 1
            else:
                Diag_G2O_patient['fn'] += 1
        else:
            if not pred_patient == 5:
                Diag_G2O_patient['tn'] += 1
            else:
                Diag_G2O_patient['fp'] += 1
    for patient, pred_wsis in Patient_predori_DiagSim.items():
        predicted_all_DiagSim_patient.append(np.mean(np.array(Patient_predori_DiagSim[patient]), axis=0))
    for i in range(len(label_all_DiagSim_patient)):
        count_DiagSim_patient += 1
        pred_patient = np.argmax(predicted_all_DiagSim_patient[i])
        if label_all_DiagSim_patient[i] == pred_patient:
            correct_DiagSim_patient += 1
        # GBM
        if label_all_DiagSim_patient[i] == 0:
            if pred_patient == 0:
                DiagSim_GBM_patient['tp'] += 1
            else:
                DiagSim_GBM_patient['fn'] += 1
        else:
            if not pred_patient == 0:
                DiagSim_GBM_patient['tn'] += 1
            else:
                DiagSim_GBM_patient['fp'] += 1
        # G4A
        if label_all_DiagSim_patient[i] == 1:
            if pred_patient == 1:
                DiagSim_G4A_patient['tp'] += 1
            else:
                DiagSim_G4A_patient['fn'] += 1
        else:
            if not pred_patient == 1:
                DiagSim_G4A_patient['tn'] += 1
            else:
                DiagSim_G4A_patient['fp'] += 1
        # G23A
        if label_all_DiagSim_patient[i] == 2:
            if pred_patient == 2:
                DiagSim_G23A_patient['tp'] += 1
            else:
                DiagSim_G23A_patient['fn'] += 1
        else:
            if not pred_patient == 2:
                DiagSim_G23A_patient['tn'] += 1
            else:
                DiagSim_G23A_patient['fp'] += 1
        # G23O
        if label_all_DiagSim_patient[i] == 3:
            if pred_patient == 3:
                DiagSim_G23O_patient['tp'] += 1
            else:
                DiagSim_G23O_patient['fn'] += 1
        else:
            if not pred_patient == 3:
                DiagSim_G23O_patient['tn'] += 1
            else:
                DiagSim_G23O_patient['fp'] += 1
    for patient, pred_wsis in Patient_predori_IDH.items():
        predicted_all_IDH_patient.append(np.mean(np.array(Patient_predori_IDH[patient]), axis=0))
    for i in range(len(label_all_IDH_patient)):
        count_IDH_patient += 1
        pred_patient = np.argmax(predicted_all_IDH_patient[i])
        if label_all_IDH_patient[i] == pred_patient:
            correct_IDH_patient += 1

        if label_all_IDH_patient[i] == 0 and pred_patient == 0:
            IDH_metrics_patient['tn'] += 1
        if label_all_IDH_patient[i] == 0 and pred_patient == 1:
            IDH_metrics_patient['fp'] += 1
        if label_all_IDH_patient[i] == 1 and pred_patient == 0:
            IDH_metrics_patient['fn'] += 1
        if label_all_IDH_patient[i] == 1 and pred_patient == 1:
            IDH_metrics_patient['tp'] += 1
    for patient, pred_wsis in Patient_predori_1p19q.items():
        predicted_all_1p19q_patient.append(np.mean(np.array(Patient_predori_1p19q[patient]), axis=0))
    for i in range(len(label_all_1p19q_patient)):
        count_1p19q_patient += 1
        pred_patient = np.argmax(predicted_all_1p19q_patient[i])
        if label_all_1p19q_patient[i] == pred_patient:
            correct_1p19q_patient += 1
        if label_all_1p19q_patient[i] == 0 and pred_patient == 0:
            p19q_metrics_patient['tn'] += 1
        if label_all_1p19q_patient[i] == 0 and pred_patient == 1:
            p19q_metrics_patient['fp'] += 1
        if label_all_1p19q_patient[i] == 1 and pred_patient == 0:
            p19q_metrics_patient['fn'] += 1
        if label_all_1p19q_patient[i] == 1 and pred_patient == 1:
            p19q_metrics_patient['tp'] += 1
    for patient, pred_wsis in Patient_predori_CDKN.items():
        predicted_all_CDKN_patient.append(np.mean(np.array(Patient_predori_CDKN[patient]), axis=0))
    for i in range(len(label_all_CDKN_patient)):
        count_CDKN_patient += 1
        pred_patient = np.argmax(predicted_all_CDKN_patient[i])
        if label_all_CDKN_patient[i] == pred_patient:
            correct_CDKN_patient += 1

        if label_all_CDKN_patient[i] == 0 and pred_patient == 0:
            CDKN_metrics_patient['tn'] += 1
        if label_all_CDKN_patient[i] == 0 and pred_patient == 1:
            CDKN_metrics_patient['fp'] += 1
        if label_all_CDKN_patient[i] == 1 and pred_patient == 0:
            CDKN_metrics_patient['fn'] += 1
        if label_all_CDKN_patient[i] == 1 and pred_patient == 1:
            CDKN_metrics_patient['tp'] += 1

    if 1: #################################### wsi
        if (not flag_tiantan) and (not external):
            ##########  IDH
            Acc_IDH = (IDH_metrics['tp'] + IDH_metrics['tn']) / (
                        IDH_metrics['tp'] + IDH_metrics['tn'] + IDH_metrics['fp'] + IDH_metrics['fn']+ 0.000001)
            IDH_metrics['sen'] = (IDH_metrics['tp']) / (IDH_metrics['tp'] + IDH_metrics['fn'] + 0.000001)  # recall
            IDH_metrics['spec'] = (IDH_metrics['tn']) / (IDH_metrics['tn'] + IDH_metrics['fp'] + 0.000001)
            IDH_metrics['pre'] = (IDH_metrics['tp']) / (IDH_metrics['tp'] + IDH_metrics['fp'] + 0.000001)
            IDH_metrics['recall'] = IDH_metrics['sen']
            IDH_metrics['f1'] = (2 * IDH_metrics['pre'] * IDH_metrics['recall']) / (
                        IDH_metrics['pre'] + IDH_metrics['recall'] + 0.000001)
            IDH_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_IDH), y_score=np.array(predicted_all_IDH))
            ##########  1p19q
            Acc_1p19q = (p19q_metrics['tp'] + p19q_metrics['tn']) / (
                        p19q_metrics['tp'] + p19q_metrics['tn'] + p19q_metrics['fp'] + p19q_metrics['fn']+ 0.000001)
            p19q_metrics['sen'] = (p19q_metrics['tp']) / (p19q_metrics['tp'] + p19q_metrics['fn'] + 0.000001)  # recall
            p19q_metrics['spec'] = (p19q_metrics['tn']) / (p19q_metrics['tn'] + p19q_metrics['fp'] + 0.000001)
            p19q_metrics['pre'] = (p19q_metrics['tp']) / (p19q_metrics['tp'] + p19q_metrics['fp'] + 0.000001)
            p19q_metrics['recall'] = p19q_metrics['sen']
            p19q_metrics['f1'] = (2 * p19q_metrics['pre'] * p19q_metrics['recall']) / (
                        p19q_metrics['pre'] + p19q_metrics['recall'] + 0.000001)
            p19q_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_1p19q),
                                                        y_score=np.array(predicted_all_1p19q))
            ##########  CDKN

            Acc_CDKN = (CDKN_metrics['tp'] + CDKN_metrics['tn']) / (
                        CDKN_metrics['tp'] + CDKN_metrics['tn'] + CDKN_metrics['fp'] + CDKN_metrics['fn']+ 0.000001)
            CDKN_metrics['sen'] = (CDKN_metrics['tp']) / (CDKN_metrics['tp'] + CDKN_metrics['fn'] + 0.000001)  # recall
            CDKN_metrics['spec'] = (CDKN_metrics['tn']) / (CDKN_metrics['tn'] + CDKN_metrics['fp'] + 0.000001)
            CDKN_metrics['pre'] = (CDKN_metrics['tp']) / (CDKN_metrics['tp'] + CDKN_metrics['fp'] + 0.000001)
            CDKN_metrics['recall'] = CDKN_metrics['sen']
            CDKN_metrics['f1'] = (2 * CDKN_metrics['pre'] * CDKN_metrics['recall']) / (
                        CDKN_metrics['pre'] + CDKN_metrics['recall'] + 0.000001)
            CDKN_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_CDKN),
                                                        y_score=np.array(predicted_all_CDKN))
        ##########  Diag
        Acc_Diag = correct_Diag / count_Diag
        #  Sensitivity
        Diag_GBM['sen'] = (Diag_GBM['tp']) / (Diag_GBM['tp'] + Diag_GBM['fn'] + 0.000001)
        Diag_G4A['sen'] = (Diag_G4A['tp']) / (Diag_G4A['tp'] + Diag_G4A['fn'] + 0.000001)
        Diag_G3A['sen'] = (Diag_G3A['tp']) / (Diag_G3A['tp'] + Diag_G3A['fn'] + 0.000001)
        Diag_G2A['sen'] = (Diag_G2A['tp']) / (Diag_G2A['tp'] + Diag_G2A['fn'] + 0.000001)
        Diag_G3O['sen'] = (Diag_G3O['tp']) / (Diag_G3O['tp'] + Diag_G3O['fn'] + 0.000001)
        Diag_G2O['sen'] = (Diag_G2O['tp']) / (Diag_G2O['tp'] + Diag_G2O['fn'] + 0.000001)
        Diag_all['sen'] = Diag_GBM['sen'] * label_all_Diag.count(0) / len(label_all_Diag) + \
                             Diag_G4A['sen'] * label_all_Diag.count(1) / len(label_all_Diag) + \
                             Diag_G3A['sen'] * label_all_Diag.count(2) / len(label_all_Diag) + \
                             Diag_G2A['sen'] * label_all_Diag.count(3) / len(label_all_Diag) + \
                             Diag_G3O['sen'] * label_all_Diag.count(4) / len(label_all_Diag) + \
                             Diag_G2O['sen'] * label_all_Diag.count(5) / len(label_all_Diag)

        #  Spec
        Diag_GBM['spec'] = (Diag_GBM['tn']) / (Diag_GBM['tn'] + Diag_GBM['fp'] + 0.000001)
        Diag_G4A['spec'] = (Diag_G4A['tn']) / (Diag_G4A['tn'] + Diag_G4A['fp'] + 0.000001)
        Diag_G3A['spec'] = (Diag_G3A['tn']) / (Diag_G3A['tn'] + Diag_G3A['fp'] + 0.000001)
        Diag_G2A['spec'] = (Diag_G2A['tn']) / (Diag_G2A['tn'] + Diag_G2A['fp'] + 0.000001)
        Diag_G3O['spec'] = (Diag_G3O['tn']) / (Diag_G3O['tn'] + Diag_G3O['fp'] + 0.000001)
        Diag_G2O['spec'] = (Diag_G2O['tn']) / (Diag_G2O['tn'] + Diag_G2O['fp'] + 0.000001)
        Diag_all['spec'] = Diag_GBM['spec'] * label_all_Diag.count(0) / len(label_all_Diag) + \
                              Diag_G4A['spec'] * label_all_Diag.count(1) / len(label_all_Diag) + \
                              Diag_G3A['spec'] * label_all_Diag.count(2) / len(label_all_Diag) + \
                              Diag_G2A['spec'] * label_all_Diag.count(3) / len(label_all_Diag) + \
                              Diag_G3O['spec'] * label_all_Diag.count(4) / len(label_all_Diag) + \
                              Diag_G2O['spec'] * label_all_Diag.count(5) / len(label_all_Diag)
        #  Precision
        Diag_GBM['pre'] = (Diag_GBM['tp']) / (Diag_GBM['tp'] + Diag_GBM['fp'] + 0.000001)
        Diag_G4A['pre'] = (Diag_G4A['tp']) / (Diag_G4A['tp'] + Diag_G4A['fp'] + 0.000001)
        Diag_G3A['pre'] = (Diag_G3A['tp']) / (Diag_G3A['tp'] + Diag_G3A['fp'] + 0.000001)
        Diag_G2A['pre'] = (Diag_G2A['tp']) / (Diag_G2A['tp'] + Diag_G2A['fp'] + 0.000001)
        Diag_G3O['pre'] = (Diag_G3O['tp']) / (Diag_G3O['tp'] + Diag_G3O['fp'] + 0.000001)
        Diag_G2O['pre'] = (Diag_G2O['tp']) / (Diag_G2O['tp'] + Diag_G2O['fp'] + 0.000001)
        Diag_all['pre'] = Diag_GBM['pre'] * label_all_Diag.count(0) / len(label_all_Diag) + \
                             Diag_G4A['pre'] * label_all_Diag.count(1) / len(label_all_Diag) + \
                             Diag_G3A['pre'] * label_all_Diag.count(2) / len(label_all_Diag) + \
                             Diag_G2A['pre'] * label_all_Diag.count(3) / len(label_all_Diag) + \
                             Diag_G3O['pre'] * label_all_Diag.count(4) / len(label_all_Diag) + \
                             Diag_G2O['pre'] * label_all_Diag.count(5) / len(label_all_Diag)
        #  Recall
        Diag_GBM['recall'] = (Diag_GBM['tp']) / (Diag_GBM['tp'] + Diag_GBM['fn'] + 0.000001)
        Diag_G4A['recall'] = (Diag_G4A['tp']) / (Diag_G4A['tp'] + Diag_G4A['fn'] + 0.000001)
        Diag_G3A['recall'] = (Diag_G3A['tp']) / (Diag_G3A['tp'] + Diag_G3A['fn'] + 0.000001)
        Diag_G2A['recall'] = (Diag_G2A['tp']) / (Diag_G2A['tp'] + Diag_G2A['fn'] + 0.000001)
        Diag_G3O['recall'] = (Diag_G3O['tp']) / (Diag_G3O['tp'] + Diag_G3O['fn'] + 0.000001)
        Diag_G2O['recall'] = (Diag_G2O['tp']) / (Diag_G2O['tp'] + Diag_G2O['fn'] + 0.000001)
        Diag_all['recall'] = Diag_GBM['recall'] * label_all_Diag.count(0) / len(label_all_Diag) + \
                                Diag_G4A['recall'] * label_all_Diag.count(1) / len(label_all_Diag) + \
                                Diag_G3A['recall'] * label_all_Diag.count(2) / len(label_all_Diag) + \
                                Diag_G2A['recall'] * label_all_Diag.count(3) / len(label_all_Diag) + \
                                Diag_G3O['recall'] * label_all_Diag.count(4) / len(label_all_Diag) + \
                                Diag_G2O['recall'] * label_all_Diag.count(5) / len(label_all_Diag)
        #  F1
        Diag_GBM['f1'] = (2 * Diag_GBM['pre'] * Diag_GBM['recall']) / (Diag_GBM['pre'] + Diag_GBM['recall'] + 0.000001)
        Diag_G4A['f1'] = (2 * Diag_G4A['pre'] * Diag_G4A['recall']) / (Diag_G4A['pre'] + Diag_G4A['recall'] + 0.000001)
        Diag_G3A['f1'] = (2 * Diag_G3A['pre'] * Diag_G3A['recall']) / (Diag_G3A['pre'] + Diag_G3A['recall'] + 0.000001)
        Diag_G2A['f1'] = (2 * Diag_G2A['pre'] * Diag_G2A['recall']) / (Diag_G2A['pre'] + Diag_G2A['recall'] + 0.000001)
        Diag_G3O['f1'] = (2 * Diag_G3O['pre'] * Diag_G3O['recall']) / (Diag_G3O['pre'] + Diag_G3O['recall'] + 0.000001)
        Diag_G2O['f1'] = (2 * Diag_G2O['pre'] * Diag_G2O['recall']) / (Diag_G2O['pre'] + Diag_G2O['recall'] + 0.000001)
        Diag_all['f1'] =  Diag_GBM['f1'] * label_all_Diag.count(0) / len(label_all_Diag) + \
                            Diag_G4A['f1'] * label_all_Diag.count(1) / len(label_all_Diag) + \
                            Diag_G3A['f1'] * label_all_Diag.count(2) / len(label_all_Diag) + \
                            Diag_G2A['f1'] * label_all_Diag.count(3) / len(label_all_Diag) + \
                            Diag_G3O['f1'] * label_all_Diag.count(4) / len(label_all_Diag) + \
                            Diag_G2O['f1'] * label_all_Diag.count(5) / len(label_all_Diag)
        # AUC
        if not external:
            out_cls_all_softmax_Diag = np.array(predicted_all_Diag)
            label_all_np = np.array(label_all_Diag)
            label_all_onehot = make_one_hot(label_all_np)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(6):
                fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_Diag[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(6)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(6):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= 6
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            Diag_GBM['AUC'] = roc_auc[0]
            Diag_G4A['AUC'] = roc_auc[1]
            Diag_G3A['AUC'] = roc_auc[2]
            Diag_G2A['AUC'] = roc_auc[3]
            Diag_G3O['AUC'] = roc_auc[4]
            Diag_G2O['AUC'] = roc_auc[5]
            fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), out_cls_all_softmax_Diag.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            Diag_all['AUC'] = roc_auc["micro"]
        else:
            out_cls_all_softmax_Diag = np.array(predicted_all_Diag)
            label_all_np = np.array(label_all_Diag)
            label_all_onehot = make_one_hot(label_all_np,N=2)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(2):
                fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_Diag[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(2)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(2):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= 2
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            Diag_GBM['AUC'] = roc_auc[0]
            Diag_G4A['AUC'] = roc_auc[1]
            Diag_G3A['AUC'] = 0
            Diag_G2A['AUC'] = 0
            Diag_G3O['AUC'] = 0
            Diag_G2O['AUC'] = 0
            fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), out_cls_all_softmax_Diag[:,0:2].ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            Diag_all['AUC'] = roc_auc["micro"]
        ##########  DiagSim
        Acc_DiagSim = correct_Diag_Sim / count_Diag_Sim
        #  Sensitivity
        DiagSim_GBM['sen'] = (DiagSim_GBM['tp']) / (DiagSim_GBM['tp'] + DiagSim_GBM['fn'] + 0.000001)
        DiagSim_G4A['sen'] = (DiagSim_G4A['tp']) / (DiagSim_G4A['tp'] + DiagSim_G4A['fn'] + 0.000001)
        DiagSim_G23A['sen'] = (DiagSim_G23A['tp']) / (DiagSim_G23A['tp'] + DiagSim_G23A['fn'] + 0.000001)
        DiagSim_G23O['sen'] = (DiagSim_G23O['tp']) / (DiagSim_G23O['tp'] + DiagSim_G23O['fn'] + 0.000001)
        DiagSim_all['sen'] = DiagSim_GBM['sen'] * label_all_DiagSim.count(0) / len(label_all_DiagSim) + \
                            DiagSim_G4A['sen'] * label_all_DiagSim.count(1) / len(label_all_DiagSim) + \
                            DiagSim_G23A['sen'] * label_all_DiagSim.count(2) / len(label_all_DiagSim) + \
                            DiagSim_G23O['sen'] * label_all_DiagSim.count(3) / len(label_all_DiagSim)
        #  Spec
        DiagSim_GBM['spec'] = (DiagSim_GBM['tn']) / (DiagSim_GBM['tn'] + DiagSim_GBM['fp'] + 0.000001)
        DiagSim_G4A['spec'] = (DiagSim_G4A['tn']) / (DiagSim_G4A['tn'] + DiagSim_G4A['fp'] + 0.000001)
        DiagSim_G23A['spec'] = (DiagSim_G23A['tn']) / (DiagSim_G23A['tn'] + DiagSim_G23A['fp'] + 0.000001)
        DiagSim_G23O['spec'] = (DiagSim_G23O['tn']) / (DiagSim_G23O['tn'] + DiagSim_G23O['fp'] + 0.000001)
        DiagSim_all['spec'] =  DiagSim_GBM['spec'] * label_all_DiagSim.count(0) / len(label_all_DiagSim) + \
                            DiagSim_G4A['spec'] * label_all_DiagSim.count(1) / len(label_all_DiagSim) + \
                            DiagSim_G23A['spec'] * label_all_DiagSim.count(2) / len(label_all_DiagSim) + \
                            DiagSim_G23O['spec'] * label_all_DiagSim.count(3) / len(label_all_DiagSim)
        #  Precision
        DiagSim_GBM['pre'] = (DiagSim_GBM['tp']) / (DiagSim_GBM['tp'] + DiagSim_GBM['fp'] + 0.000001)
        DiagSim_G4A['pre'] = (DiagSim_G4A['tp']) / (DiagSim_G4A['tp'] + DiagSim_G4A['fp'] + 0.000001)
        DiagSim_G23A['pre'] = (DiagSim_G23A['tp']) / (DiagSim_G23A['tp'] + DiagSim_G23A['fp'] + 0.000001)
        DiagSim_G23O['pre'] = (DiagSim_G23O['tp']) / (DiagSim_G23O['tp'] + DiagSim_G23O['fp'] + 0.000001)
        DiagSim_all['pre'] =  DiagSim_GBM['pre'] * label_all_DiagSim.count(0) / len(label_all_DiagSim) + \
                            DiagSim_G4A['pre'] * label_all_DiagSim.count(1) / len(label_all_DiagSim) + \
                            DiagSim_G23A['pre'] * label_all_DiagSim.count(2) / len(label_all_DiagSim) + \
                            DiagSim_G23O['pre'] * label_all_DiagSim.count(3) / len(label_all_DiagSim)
        #  Recall
        DiagSim_GBM['recall'] = (DiagSim_GBM['tp']) / (DiagSim_GBM['tp'] + DiagSim_GBM['fn'] + 0.000001)
        DiagSim_G4A['recall'] = (DiagSim_G4A['tp']) / (DiagSim_G4A['tp'] + DiagSim_G4A['fn'] + 0.000001)
        DiagSim_G23A['recall'] = (DiagSim_G23A['tp']) / (DiagSim_G23A['tp'] + DiagSim_G23A['fn'] + 0.000001)
        DiagSim_G23O['recall'] = (DiagSim_G23O['tp']) / (DiagSim_G23O['tp'] + DiagSim_G23O['fn'] + 0.000001)
        DiagSim_all['recall'] =  DiagSim_GBM['recall'] * label_all_DiagSim.count(0) / len(label_all_DiagSim) + \
                            DiagSim_G4A['recall'] * label_all_DiagSim.count(1) / len(label_all_DiagSim) + \
                            DiagSim_G23A['recall'] * label_all_DiagSim.count(2) / len(label_all_DiagSim) + \
                            DiagSim_G23O['recall'] * label_all_DiagSim.count(3) / len(label_all_DiagSim)
        #  F1
        DiagSim_GBM['f1'] = (2 * DiagSim_GBM['pre'] * DiagSim_GBM['recall']) / (
                    DiagSim_GBM['pre'] + DiagSim_GBM['recall'] + 0.000001)
        DiagSim_G4A['f1'] = (2 * DiagSim_G4A['pre'] * DiagSim_G4A['recall']) / (
                    DiagSim_G4A['pre'] + DiagSim_G4A['recall'] + 0.000001)
        DiagSim_G23A['f1'] = (2 * DiagSim_G23A['pre'] * DiagSim_G23A['recall']) / (
                    DiagSim_G23A['pre'] + DiagSim_G23A['recall'] + 0.000001)
        DiagSim_G23O['f1'] = (2 * DiagSim_G23O['pre'] * DiagSim_G23O['recall']) / (
                    DiagSim_G23O['pre'] + DiagSim_G23O['recall'] + 0.000001)
        DiagSim_all['f1'] =  DiagSim_GBM['f1'] * label_all_DiagSim.count(0) / len(label_all_DiagSim) + \
                            DiagSim_G4A['f1'] * label_all_DiagSim.count(1) / len(label_all_DiagSim) + \
                            DiagSim_G23A['f1'] * label_all_DiagSim.count(2) / len(label_all_DiagSim) + \
                            DiagSim_G23O['f1'] * label_all_DiagSim.count(3) / len(label_all_DiagSim)
        # AUC
        if not external:
            out_cls_all_softmax_DiagSim = np.array(predicted_all_DiagSim)
            label_all_np = np.array(label_all_DiagSim)
            label_all_onehot = make_one_hot(label_all_np)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(4):
                fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_DiagSim[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(4):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= 4
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            DiagSim_GBM['AUC'] = roc_auc[0]
            DiagSim_G4A['AUC'] = roc_auc[1]
            DiagSim_G23A['AUC'] = roc_auc[2]
            DiagSim_G23O['AUC'] = roc_auc[3]

            fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), out_cls_all_softmax_DiagSim.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            DiagSim_all['AUC'] = roc_auc["micro"]
        else:
            out_cls_all_softmax_DiagSim = np.array(predicted_all_DiagSim)
            label_all_np = np.array(label_all_DiagSim)
            label_all_onehot = make_one_hot(label_all_np, N=2)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(2):
                fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_DiagSim[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(2)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(2):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= 2
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            DiagSim_GBM['AUC'] = roc_auc[0]
            DiagSim_G4A['AUC'] = roc_auc[1]
            DiagSim_G23A['AUC'] = 0
            DiagSim_G23O['AUC'] = 0

            fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(),
                                                      out_cls_all_softmax_DiagSim[:, 0:2].ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            DiagSim_all['AUC'] = roc_auc["micro"]

        list_Diag = (Acc_Diag, Diag_all['sen'], Diag_all['spec'], Diag_all['pre'], Diag_all['recall']
                , Diag_all['f1'], Diag_all['AUC'])

        list_Diag_GBM = (None, Diag_GBM['sen'], Diag_GBM['spec'], Diag_GBM['pre'], Diag_GBM['recall']
                    , Diag_GBM['f1'], Diag_GBM['AUC'])
        list_Diag_G4A = (None, Diag_G4A['sen'], Diag_G4A['spec'], Diag_G4A['pre'], Diag_G4A['recall']
                    , Diag_G4A['f1'], Diag_G4A['AUC'])
        list_Diag_G3A = (None, Diag_G3A['sen'], Diag_G3A['spec'], Diag_G3A['pre'], Diag_G3A['recall']
                    , Diag_G3A['f1'], Diag_G3A['AUC'])
        list_Diag_G2A = (None, Diag_G2A['sen'], Diag_G2A['spec'], Diag_G2A['pre'], Diag_G2A['recall']
                    , Diag_G2A['f1'], Diag_G2A['AUC'])
        list_Diag_G3O = (None, Diag_G3O['sen'], Diag_G3O['spec'], Diag_G3O['pre'], Diag_G3O['recall']
                    , Diag_G3O['f1'], Diag_G3O['AUC'])
        list_Diag_G2O = (None, Diag_G2O['sen'], Diag_G2O['spec'], Diag_G2O['pre'], Diag_G2O['recall']
                    , Diag_G2O['f1'], Diag_G2O['AUC'])

        list_DiagSim = (Acc_DiagSim, DiagSim_all['sen'], DiagSim_all['spec'], DiagSim_all['pre'], DiagSim_all['recall']
                     , DiagSim_all['f1'], DiagSim_all['AUC'])

        list_DiagSim_GBM = (None, DiagSim_GBM['sen'], DiagSim_GBM['spec'], DiagSim_GBM['pre'], DiagSim_GBM['recall']
                            , DiagSim_GBM['f1'], DiagSim_GBM['AUC'])
        list_DiagSim_G4A = (None, DiagSim_G4A['sen'], DiagSim_G4A['spec'], DiagSim_G4A['pre'], DiagSim_G4A['recall']
                            , DiagSim_G4A['f1'], DiagSim_G4A['AUC'])
        list_DiagSim_G23A = (None, DiagSim_G23A['sen'], DiagSim_G23A['spec'], DiagSim_G23A['pre'], DiagSim_G23A['recall']
                            , DiagSim_G23A['f1'], DiagSim_G23A['AUC'])
        list_DiagSim_G23O = (None, DiagSim_G23O['sen'], DiagSim_G23O['spec'], DiagSim_G23O['pre'], DiagSim_G23O['recall']
                            , DiagSim_G23O['f1'], DiagSim_G23O['AUC'])

        list_IDH = (Acc_IDH, IDH_metrics['sen'], IDH_metrics['spec'], IDH_metrics['pre'], IDH_metrics['recall']
                        , IDH_metrics['f1'], IDH_metrics['AUC'])
        list_1p19q = (Acc_1p19q, p19q_metrics['sen'], p19q_metrics['spec'], p19q_metrics['pre'], p19q_metrics['recall']
                    , p19q_metrics['f1'], p19q_metrics['AUC'])
        list_CDKN = (Acc_CDKN, CDKN_metrics['sen'], CDKN_metrics['spec'], CDKN_metrics['pre'], CDKN_metrics['recall']
                      , CDKN_metrics['f1'], CDKN_metrics['AUC'])

    if 1:  #################################### patient
        if (not flag_tiantan) and (not external):
            ##########  IDH
            Acc_IDH_patient = (IDH_metrics_patient['tp'] + IDH_metrics_patient['tn']) / (
                    IDH_metrics_patient['tp'] + IDH_metrics_patient['tn'] + IDH_metrics_patient['fp'] + IDH_metrics_patient[
                'fn']+ 0.000001)
            IDH_metrics_patient['sen'] = (IDH_metrics_patient['tp']) / (
                        IDH_metrics_patient['tp'] + IDH_metrics_patient['fn'] + 0.000001)  # recall
            IDH_metrics_patient['spec'] = (IDH_metrics_patient['tn']) / (
                        IDH_metrics_patient['tn'] + IDH_metrics_patient['fp'] + 0.000001)
            IDH_metrics_patient['pre'] = (IDH_metrics_patient['tp']) / (
                        IDH_metrics_patient['tp'] + IDH_metrics_patient['fp'] + 0.000001)
            IDH_metrics_patient['recall'] = IDH_metrics_patient['sen']
            IDH_metrics_patient['f1'] = (2 * IDH_metrics_patient['pre'] * IDH_metrics_patient['recall']) / (
                    IDH_metrics_patient['pre'] + IDH_metrics_patient['recall'] + 0.000001)
            IDH_metrics_patient['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_IDH_patient),
                                                               y_score=np.array(predicted_all_IDH_patient)[:,1])
            ##########  1p19q
            Acc_1p19q_patient = (p19q_metrics_patient['tp'] + p19q_metrics_patient['tn']) / (
                    p19q_metrics_patient['tp'] + p19q_metrics_patient['tn'] + p19q_metrics_patient['fp'] +
                    p19q_metrics_patient['fn']+ 0.000001)
            p19q_metrics_patient['sen'] = (p19q_metrics_patient['tp']) / (
                        p19q_metrics_patient['tp'] + p19q_metrics_patient['fn'] + 0.000001)  # recall
            p19q_metrics_patient['spec'] = (p19q_metrics_patient['tn']) / (
                        p19q_metrics_patient['tn'] + p19q_metrics_patient['fp'] + 0.000001)
            p19q_metrics_patient['pre'] = (p19q_metrics_patient['tp']) / (
                        p19q_metrics_patient['tp'] + p19q_metrics_patient['fp'] + 0.000001)
            p19q_metrics_patient['recall'] = p19q_metrics_patient['sen']
            p19q_metrics_patient['f1'] = (2 * p19q_metrics_patient['pre'] * p19q_metrics_patient['recall']) / (
                    p19q_metrics_patient['pre'] + p19q_metrics_patient['recall'] + 0.000001)
            p19q_metrics_patient['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_1p19q_patient),
                                                                y_score=np.array(predicted_all_1p19q_patient)[:,1])
            ##########  CDKN
            Acc_CDKN_patient = (CDKN_metrics_patient['tp'] + CDKN_metrics_patient['tn']) / (
                    CDKN_metrics_patient['tp'] + CDKN_metrics_patient['tn'] + CDKN_metrics_patient['fp'] +
                    CDKN_metrics_patient['fn']+ 0.000001)
            CDKN_metrics_patient['sen'] = (CDKN_metrics_patient['tp']) / (
                        CDKN_metrics_patient['tp'] + CDKN_metrics_patient['fn'] + 0.000001)  # recall
            CDKN_metrics_patient['spec'] = (CDKN_metrics_patient['tn']) / (
                        CDKN_metrics_patient['tn'] + CDKN_metrics_patient['fp'] + 0.000001)
            CDKN_metrics_patient['pre'] = (CDKN_metrics_patient['tp']) / (
                        CDKN_metrics_patient['tp'] + CDKN_metrics_patient['fp'] + 0.000001)
            CDKN_metrics_patient['recall'] = CDKN_metrics_patient['sen']
            CDKN_metrics_patient['f1'] = (2 * CDKN_metrics_patient['pre'] * CDKN_metrics_patient['recall']) / (
                    CDKN_metrics_patient['pre'] + CDKN_metrics_patient['recall'] + 0.000001)
            CDKN_metrics_patient['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_CDKN_patient),
                                                                y_score=np.array(predicted_all_CDKN_patient)[:,1])
        ##########  Diag
        Acc_Diag_patient = correct_Diag_patient / count_Diag_patient
        #  Sensitivity
        Diag_GBM_patient['sen'] = (Diag_GBM_patient['tp']) / (
                    Diag_GBM_patient['tp'] + Diag_GBM_patient['fn'] + 0.000001)
        Diag_G4A_patient['sen'] = (Diag_G4A_patient['tp']) / (
                    Diag_G4A_patient['tp'] + Diag_G4A_patient['fn'] + 0.000001)
        Diag_G3A_patient['sen'] = (Diag_G3A_patient['tp']) / (
                    Diag_G3A_patient['tp'] + Diag_G3A_patient['fn'] + 0.000001)
        Diag_G2A_patient['sen'] = (Diag_G2A_patient['tp']) / (
                    Diag_G2A_patient['tp'] + Diag_G2A_patient['fn'] + 0.000001)
        Diag_G3O_patient['sen'] = (Diag_G3O_patient['tp']) / (
                    Diag_G3O_patient['tp'] + Diag_G3O_patient['fn'] + 0.000001)
        Diag_G2O_patient['sen'] = (Diag_G2O_patient['tp']) / (
                    Diag_G2O_patient['tp'] + Diag_G2O_patient['fn'] + 0.000001)
        Diag_all_patient['sen'] = Diag_GBM_patient['sen'] * label_all_Diag_patient.count(0) / len(label_all_Diag_patient) + \
                          Diag_G4A_patient['sen'] * label_all_Diag_patient.count(1) / len(label_all_Diag_patient) + \
                          Diag_G3A_patient['sen'] * label_all_Diag_patient.count(2) / len(label_all_Diag_patient) + \
                          Diag_G2A_patient['sen'] * label_all_Diag_patient.count(3) / len(label_all_Diag_patient) + \
                          Diag_G3O_patient['sen'] * label_all_Diag_patient.count(4) / len(label_all_Diag_patient) + \
                          Diag_G2O_patient['sen'] * label_all_Diag_patient.count(5) / len(label_all_Diag_patient)

        #  Spec
        Diag_GBM_patient['spec'] = (Diag_GBM_patient['tn']) / (
                    Diag_GBM_patient['tn'] + Diag_GBM_patient['fp'] + 0.000001)
        Diag_G4A_patient['spec'] = (Diag_G4A_patient['tn']) / (
                    Diag_G4A_patient['tn'] + Diag_G4A_patient['fp'] + 0.000001)
        Diag_G3A_patient['spec'] = (Diag_G3A_patient['tn']) / (
                    Diag_G3A_patient['tn'] + Diag_G3A_patient['fp'] + 0.000001)
        Diag_G2A_patient['spec'] = (Diag_G2A_patient['tn']) / (
                    Diag_G2A_patient['tn'] + Diag_G2A_patient['fp'] + 0.000001)
        Diag_G3O_patient['spec'] = (Diag_G3O_patient['tn']) / (
                    Diag_G3O_patient['tn'] + Diag_G3O_patient['fp'] + 0.000001)
        Diag_G2O_patient['spec'] = (Diag_G2O_patient['tn']) / (
                    Diag_G2O_patient['tn'] + Diag_G2O_patient['fp'] + 0.000001)
        Diag_all_patient['spec'] = Diag_GBM_patient['spec'] * label_all_Diag_patient.count(0) / len(label_all_Diag_patient) + \
                           Diag_G4A_patient['spec'] * label_all_Diag_patient.count(1) / len(label_all_Diag_patient) + \
                           Diag_G3A_patient['spec'] * label_all_Diag_patient.count(2) / len(label_all_Diag_patient) + \
                           Diag_G2A_patient['spec'] * label_all_Diag_patient.count(3) / len(label_all_Diag_patient) + \
                           Diag_G3O_patient['spec'] * label_all_Diag_patient.count(4) / len(label_all_Diag_patient) + \
                           Diag_G2O_patient['spec'] * label_all_Diag_patient.count(5) / len(label_all_Diag_patient)
        #  Precision
        Diag_GBM_patient['pre'] = (Diag_GBM_patient['tp']) / (
                    Diag_GBM_patient['tp'] + Diag_GBM_patient['fp'] + 0.000001)
        Diag_G4A_patient['pre'] = (Diag_G4A_patient['tp']) / (
                    Diag_G4A_patient['tp'] + Diag_G4A_patient['fp'] + 0.000001)
        Diag_G3A_patient['pre'] = (Diag_G3A_patient['tp']) / (
                    Diag_G3A_patient['tp'] + Diag_G3A_patient['fp'] + 0.000001)
        Diag_G2A_patient['pre'] = (Diag_G2A_patient['tp']) / (
                    Diag_G2A_patient['tp'] + Diag_G2A_patient['fp'] + 0.000001)
        Diag_G3O_patient['pre'] = (Diag_G3O_patient['tp']) / (
                    Diag_G3O_patient['tp'] + Diag_G3O_patient['fp'] + 0.000001)
        Diag_G2O_patient['pre'] = (Diag_G2O_patient['tp']) / (
                    Diag_G2O_patient['tp'] + Diag_G2O_patient['fp'] + 0.000001)
        Diag_all_patient['pre'] = Diag_GBM_patient['pre'] * label_all_Diag_patient.count(0) / len(label_all_Diag_patient) + \
                          Diag_G4A_patient['pre'] * label_all_Diag_patient.count(1) / len(label_all_Diag_patient) + \
                          Diag_G3A_patient['pre'] * label_all_Diag_patient.count(2) / len(label_all_Diag_patient) + \
                          Diag_G2A_patient['pre'] * label_all_Diag_patient.count(3) / len(label_all_Diag_patient) + \
                          Diag_G3O_patient['pre'] * label_all_Diag_patient.count(4) / len(label_all_Diag_patient) + \
                          Diag_G2O_patient['pre'] * label_all_Diag_patient.count(5) / len(label_all_Diag_patient)
        #  Recall
        Diag_GBM_patient['recall'] = (Diag_GBM_patient['tp']) / (
                    Diag_GBM_patient['tp'] + Diag_GBM_patient['fn'] + 0.000001)
        Diag_G4A_patient['recall'] = (Diag_G4A_patient['tp']) / (
                    Diag_G4A_patient['tp'] + Diag_G4A_patient['fn'] + 0.000001)
        Diag_G3A_patient['recall'] = (Diag_G3A_patient['tp']) / (
                    Diag_G3A_patient['tp'] + Diag_G3A_patient['fn'] + 0.000001)
        Diag_G2A_patient['recall'] = (Diag_G2A_patient['tp']) / (
                    Diag_G2A_patient['tp'] + Diag_G2A_patient['fn'] + 0.000001)
        Diag_G3O_patient['recall'] = (Diag_G3O_patient['tp']) / (
                    Diag_G3O_patient['tp'] + Diag_G3O_patient['fn'] + 0.000001)
        Diag_G2O_patient['recall'] = (Diag_G2O_patient['tp']) / (
                    Diag_G2O_patient['tp'] + Diag_G2O_patient['fn'] + 0.000001)
        Diag_all_patient['recall'] = Diag_GBM_patient['recall'] * label_all_Diag_patient.count(0) / len(
            label_all_Diag_patient) + \
                             Diag_G4A_patient['recall'] * label_all_Diag_patient.count(1) / len(
            label_all_Diag_patient) + \
                             Diag_G3A_patient['recall'] * label_all_Diag_patient.count(2) / len(
            label_all_Diag_patient) + \
                             Diag_G2A_patient['recall'] * label_all_Diag_patient.count(3) / len(
            label_all_Diag_patient) + \
                             Diag_G3O_patient['recall'] * label_all_Diag_patient.count(4) / len(
            label_all_Diag_patient) + \
                             Diag_G2O_patient['recall'] * label_all_Diag_patient.count(5) / len(label_all_Diag_patient)
        #  F1
        Diag_GBM_patient['f1'] = (2 * Diag_GBM_patient['pre'] * Diag_GBM_patient['recall']) / (
                    Diag_GBM_patient['pre'] + Diag_GBM_patient['recall'] + 0.000001)
        Diag_G4A_patient['f1'] = (2 * Diag_G4A_patient['pre'] * Diag_G4A_patient['recall']) / (
                    Diag_G4A_patient['pre'] + Diag_G4A_patient['recall'] + 0.000001)
        Diag_G3A_patient['f1'] = (2 * Diag_G3A_patient['pre'] * Diag_G3A_patient['recall']) / (
                    Diag_G3A_patient['pre'] + Diag_G3A_patient['recall'] + 0.000001)
        Diag_G2A_patient['f1'] = (2 * Diag_G2A_patient['pre'] * Diag_G2A_patient['recall']) / (
                    Diag_G2A_patient['pre'] + Diag_G2A_patient['recall'] + 0.000001)
        Diag_G3O_patient['f1'] = (2 * Diag_G3O_patient['pre'] * Diag_G3O_patient['recall']) / (
                    Diag_G3O_patient['pre'] + Diag_G3O_patient['recall'] + 0.000001)
        Diag_G2O_patient['f1'] = (2 * Diag_G2O_patient['pre'] * Diag_G2O_patient['recall']) / (
                    Diag_G2O_patient['pre'] + Diag_G2O_patient['recall'] + 0.000001)
        Diag_all_patient['f1'] = Diag_GBM_patient['f1'] * label_all_Diag_patient.count(0) / len(label_all_Diag_patient) + \
                         Diag_G4A_patient['f1'] * label_all_Diag_patient.count(1) / len(label_all_Diag_patient) + \
                         Diag_G3A_patient['f1'] * label_all_Diag_patient.count(2) / len(label_all_Diag_patient) + \
                         Diag_G2A_patient['f1'] * label_all_Diag_patient.count(3) / len(label_all_Diag_patient) + \
                         Diag_G3O_patient['f1'] * label_all_Diag_patient.count(4) / len(label_all_Diag_patient) + \
                         Diag_G2O_patient['f1'] * label_all_Diag_patient.count(5) / len(label_all_Diag_patient)
        # AUC
        if not external:
            out_cls_all_softmax_Diag_patient = np.array(predicted_all_Diag_patient)
            label_all_np = np.array(label_all_Diag_patient)
            label_all_onehot = make_one_hot(label_all_np)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(6):
                fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_Diag_patient[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(6)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(6):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= 6
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            Diag_GBM_patient['AUC'] = roc_auc[0]
            Diag_G4A_patient['AUC'] = roc_auc[1]
            Diag_G3A_patient['AUC'] = roc_auc[2]
            Diag_G2A_patient['AUC'] = roc_auc[3]
            Diag_G3O_patient['AUC'] = roc_auc[4]
            Diag_G2O_patient['AUC'] = roc_auc[5]
            fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), out_cls_all_softmax_Diag_patient.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            Diag_all_patient['AUC'] = roc_auc["micro"]
        else:
            out_cls_all_softmax_Diag_patient = np.array(predicted_all_Diag_patient)
            label_all_np = np.array(label_all_Diag_patient)
            label_all_onehot = make_one_hot(label_all_np, N=2)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(2):
                fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_Diag_patient[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(2)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(2):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= 2
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            Diag_GBM_patient['AUC'] = roc_auc[0]
            Diag_G4A_patient['AUC'] = roc_auc[1]
            Diag_G3A_patient['AUC'] = 0
            Diag_G2A_patient['AUC'] = 0
            Diag_G3O_patient['AUC'] = 0
            Diag_G2O_patient['AUC'] = 0
            fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(),
                                                      out_cls_all_softmax_Diag_patient[:, 0:2].ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            Diag_all_patient['AUC'] = roc_auc["micro"]
        ##########  DiagSim
        Acc_DiagSim_patient = correct_DiagSim_patient / count_DiagSim_patient
        #  Sensitivity
        DiagSim_GBM_patient['sen'] = (DiagSim_GBM_patient['tp']) / (
                    DiagSim_GBM_patient['tp'] + DiagSim_GBM_patient['fn'] + 0.000001)
        DiagSim_G4A_patient['sen'] = (DiagSim_G4A_patient['tp']) / (
                    DiagSim_G4A_patient['tp'] + DiagSim_G4A_patient['fn'] + 0.000001)
        DiagSim_G23A_patient['sen'] = (DiagSim_G23A_patient['tp']) / (
                    DiagSim_G23A_patient['tp'] + DiagSim_G23A_patient['fn'] + 0.000001)
        DiagSim_G23O_patient['sen'] = (DiagSim_G23O_patient['tp']) / (
                    DiagSim_G23O_patient['tp'] + DiagSim_G23O_patient['fn'] + 0.000001)
        DiagSim_all_patient['sen'] = DiagSim_GBM_patient['sen'] * label_all_DiagSim_patient.count(0) / len(
            label_all_DiagSim_patient) + \
                                     DiagSim_G4A_patient['sen'] * label_all_DiagSim_patient.count(1) / len(
            label_all_DiagSim_patient) + \
                                     DiagSim_G23A_patient['sen'] * label_all_DiagSim_patient.count(2) / len(
            label_all_DiagSim_patient) + \
                                     DiagSim_G23O_patient['sen'] * label_all_DiagSim_patient.count(3) / len(
            label_all_DiagSim_patient)
        #  Spec
        DiagSim_GBM_patient['spec'] = (DiagSim_GBM_patient['tn']) / (
                    DiagSim_GBM_patient['tn'] + DiagSim_GBM_patient['fp'] + 0.000001)
        DiagSim_G4A_patient['spec'] = (DiagSim_G4A_patient['tn']) / (
                    DiagSim_G4A_patient['tn'] + DiagSim_G4A_patient['fp'] + 0.000001)
        DiagSim_G23A_patient['spec'] = (DiagSim_G23A_patient['tn']) / (
                    DiagSim_G23A_patient['tn'] + DiagSim_G23A_patient['fp'] + 0.000001)
        DiagSim_G23O_patient['spec'] = (DiagSim_G23O_patient['tn']) / (
                    DiagSim_G23O_patient['tn'] + DiagSim_G23O_patient['fp'] + 0.000001)
        DiagSim_all_patient['spec'] = DiagSim_GBM_patient['spec'] * label_all_DiagSim_patient.count(0) / len(
            label_all_DiagSim_patient) + \
                                      DiagSim_G4A_patient['spec'] * label_all_DiagSim_patient.count(1) / len(
            label_all_DiagSim_patient) + \
                                      DiagSim_G23A_patient['spec'] * label_all_DiagSim_patient.count(2) / len(
            label_all_DiagSim_patient) + \
                                      DiagSim_G23O_patient['spec'] * label_all_DiagSim_patient.count(3) / len(
            label_all_DiagSim_patient)
        #  Precision
        DiagSim_GBM_patient['pre'] = (DiagSim_GBM_patient['tp']) / (
                    DiagSim_GBM_patient['tp'] + DiagSim_GBM_patient['fp'] + 0.000001)
        DiagSim_G4A_patient['pre'] = (DiagSim_G4A_patient['tp']) / (
                    DiagSim_G4A_patient['tp'] + DiagSim_G4A_patient['fp'] + 0.000001)
        DiagSim_G23A_patient['pre'] = (DiagSim_G23A_patient['tp']) / (
                    DiagSim_G23A_patient['tp'] + DiagSim_G23A_patient['fp'] + 0.000001)
        DiagSim_G23O_patient['pre'] = (DiagSim_G23O_patient['tp']) / (
                    DiagSim_G23O_patient['tp'] + DiagSim_G23O_patient['fp'] + 0.000001)
        DiagSim_all_patient['pre'] = DiagSim_GBM_patient['pre'] * label_all_DiagSim_patient.count(0) / len(
            label_all_DiagSim_patient) + \
                                     DiagSim_G4A_patient['pre'] * label_all_DiagSim_patient.count(1) / len(
            label_all_DiagSim_patient) + \
                                     DiagSim_G23A_patient['pre'] * label_all_DiagSim_patient.count(2) / len(
            label_all_DiagSim_patient) + \
                                     DiagSim_G23O_patient['pre'] * label_all_DiagSim_patient.count(3) / len(
            label_all_DiagSim_patient)
        #  Recall
        DiagSim_GBM_patient['recall'] = (DiagSim_GBM_patient['tp']) / (
                    DiagSim_GBM_patient['tp'] + DiagSim_GBM_patient['fn'] + 0.000001)
        DiagSim_G4A_patient['recall'] = (DiagSim_G4A_patient['tp']) / (
                    DiagSim_G4A_patient['tp'] + DiagSim_G4A_patient['fn'] + 0.000001)
        DiagSim_G23A_patient['recall'] = (DiagSim_G23A_patient['tp']) / (
                    DiagSim_G23A_patient['tp'] + DiagSim_G23A_patient['fn'] + 0.000001)
        DiagSim_G23O_patient['recall'] = (DiagSim_G23O_patient['tp']) / (
                    DiagSim_G23O_patient['tp'] + DiagSim_G23O_patient['fn'] + 0.000001)
        DiagSim_all_patient['recall'] = DiagSim_GBM_patient['recall'] * label_all_DiagSim_patient.count(
            0) / len(label_all_DiagSim_patient) + \
                                        DiagSim_G4A_patient['recall'] * label_all_DiagSim_patient.count(
            1) / len(label_all_DiagSim_patient) + \
                                        DiagSim_G23A_patient['recall'] * label_all_DiagSim_patient.count(
            2) / len(label_all_DiagSim_patient) + \
                                        DiagSim_G23O_patient['recall'] * label_all_DiagSim_patient.count(
            3) / len(label_all_DiagSim_patient)
        #  F1
        DiagSim_GBM_patient['f1'] = (2 * DiagSim_GBM_patient['pre'] * DiagSim_GBM_patient['recall']) / (
                DiagSim_GBM_patient['pre'] + DiagSim_GBM_patient['recall'] + 0.000001)
        DiagSim_G4A_patient['f1'] = (2 * DiagSim_G4A_patient['pre'] * DiagSim_G4A_patient['recall']) / (
                DiagSim_G4A_patient['pre'] + DiagSim_G4A_patient['recall'] + 0.000001)
        DiagSim_G23A_patient['f1'] = (2 * DiagSim_G23A_patient['pre'] * DiagSim_G23A_patient['recall']) / (
                DiagSim_G23A_patient['pre'] + DiagSim_G23A_patient['recall'] + 0.000001)
        DiagSim_G23O_patient['f1'] = (2 * DiagSim_G23O_patient['pre'] * DiagSim_G23O_patient['recall']) / (
                DiagSim_G23O_patient['pre'] + DiagSim_G23O_patient['recall'] + 0.000001)
        DiagSim_all_patient['f1'] = DiagSim_GBM_patient['f1'] * label_all_DiagSim_patient.count(0) / len(
            label_all_DiagSim_patient) + \
                                    DiagSim_G4A_patient['f1'] * label_all_DiagSim_patient.count(1) / len(
            label_all_DiagSim_patient) + \
                                    DiagSim_G23A_patient['f1'] * label_all_DiagSim_patient.count(2) / len(
            label_all_DiagSim_patient) + \
                                    DiagSim_G23O_patient['f1'] * label_all_DiagSim_patient.count(3) / len(
            label_all_DiagSim_patient)
        # AUC
        if not external:
            out_cls_all_softmax_DiagSim_patient = np.array(predicted_all_DiagSim_patient)
            label_all_np = np.array(label_all_DiagSim_patient)
            label_all_onehot = make_one_hot(label_all_np)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(4):
                fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_DiagSim_patient[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(4):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= 4
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            DiagSim_GBM_patient['AUC'] = roc_auc[0]
            DiagSim_G4A_patient['AUC'] = roc_auc[1]
            DiagSim_G23A_patient['AUC'] = roc_auc[2]
            DiagSim_G23O_patient['AUC'] = roc_auc[3]

            fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), out_cls_all_softmax_DiagSim_patient.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            DiagSim_all_patient['AUC'] = roc_auc["micro"]
        else:
            out_cls_all_softmax_DiagSim_patient = np.array(predicted_all_DiagSim_patient)
            label_all_np = np.array(label_all_DiagSim_patient)
            label_all_onehot = make_one_hot(label_all_np, N=2)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(2):
                fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_DiagSim_patient[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(2)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(2):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= 2
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            DiagSim_GBM_patient['AUC'] = roc_auc[0]
            DiagSim_G4A_patient['AUC'] = roc_auc[1]
            DiagSim_G23A_patient['AUC'] = 0
            DiagSim_G23O_patient['AUC'] = 0

            fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(),
                                                      out_cls_all_softmax_DiagSim_patient[:, 0:2].ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            DiagSim_all_patient['AUC'] = roc_auc["micro"]
        list_Diag_patient = (Acc_Diag_patient, Diag_all_patient['sen'], Diag_all_patient['spec'], Diag_all_patient['pre'], Diag_all_patient['recall']
                     , Diag_all_patient['f1'], Diag_all_patient['AUC'])

        list_Diag_GBM_patient = (
        None, Diag_GBM_patient['sen'], Diag_GBM_patient['spec'], Diag_GBM_patient['pre'], Diag_GBM_patient['recall']
        , Diag_GBM_patient['f1'], Diag_GBM_patient['AUC'])
        list_Diag_G4A_patient = (
        None, Diag_G4A_patient['sen'], Diag_G4A_patient['spec'], Diag_G4A_patient['pre'], Diag_G4A_patient['recall']
        , Diag_G4A_patient['f1'], Diag_G4A_patient['AUC'])
        list_Diag_G3A_patient = (
        None, Diag_G3A_patient['sen'], Diag_G3A_patient['spec'], Diag_G3A_patient['pre'], Diag_G3A_patient['recall']
        , Diag_G3A_patient['f1'], Diag_G3A_patient['AUC'])
        list_Diag_G2A_patient = (
        None, Diag_G2A_patient['sen'], Diag_G2A_patient['spec'], Diag_G2A_patient['pre'], Diag_G2A_patient['recall']
        , Diag_G2A_patient['f1'], Diag_G2A_patient['AUC'])
        list_Diag_G3O_patient = (
        None, Diag_G3O_patient['sen'], Diag_G3O_patient['spec'], Diag_G3O_patient['pre'], Diag_G3O_patient['recall']
        , Diag_G3O_patient['f1'], Diag_G3O_patient['AUC'])
        list_Diag_G2O_patient = (
        None, Diag_G2O_patient['sen'], Diag_G2O_patient['spec'], Diag_G2O_patient['pre'], Diag_G2O_patient['recall']
        , Diag_G2O_patient['f1'], Diag_G2O_patient['AUC'])

        list_DiagSim_patient = (Acc_DiagSim_patient, DiagSim_all_patient['sen'], DiagSim_all_patient['spec'], DiagSim_all_patient['pre'], DiagSim_all_patient['recall']
                        , DiagSim_all_patient['f1'], DiagSim_all_patient['AUC'])

        list_DiagSim_GBM_patient = (
        None, DiagSim_GBM_patient['sen'], DiagSim_GBM_patient['spec'], DiagSim_GBM_patient['pre'],
        DiagSim_GBM_patient['recall']
        , DiagSim_GBM_patient['f1'], DiagSim_GBM_patient['AUC'])
        list_DiagSim_G4A_patient = (
        None, DiagSim_G4A_patient['sen'], DiagSim_G4A_patient['spec'], DiagSim_G4A_patient['pre'],
        DiagSim_G4A_patient['recall']
        , DiagSim_G4A_patient['f1'], DiagSim_G4A_patient['AUC'])
        list_DiagSim_G23A_patient = (
        None, DiagSim_G23A_patient['sen'], DiagSim_G23A_patient['spec'], DiagSim_G23A_patient['pre'], DiagSim_G23A_patient['recall']
        , DiagSim_G23A_patient['f1'], DiagSim_G23A_patient['AUC'])
        list_DiagSim_G23O_patient = (
        None, DiagSim_G23O_patient['sen'], DiagSim_G23O_patient['spec'], DiagSim_G23O_patient['pre'], DiagSim_G23O_patient['recall']
        , DiagSim_G23O_patient['f1'], DiagSim_G23O_patient['AUC'])

        list_IDH_patient = (Acc_IDH_patient, IDH_metrics_patient['sen'], IDH_metrics_patient['spec'], IDH_metrics_patient['pre'], IDH_metrics_patient['recall']
                    , IDH_metrics_patient['f1'], IDH_metrics_patient['AUC'])
        list_1p19q_patient = (Acc_1p19q_patient, p19q_metrics_patient['sen'], p19q_metrics_patient['spec'], p19q_metrics_patient['pre'], p19q_metrics_patient['recall']
                      , p19q_metrics_patient['f1'], p19q_metrics_patient['AUC'])
        list_CDKN_patient = (Acc_CDKN_patient, CDKN_metrics_patient['sen'], CDKN_metrics_patient['spec'], CDKN_metrics_patient['pre'], CDKN_metrics_patient['recall']
                     , CDKN_metrics_patient['f1'], CDKN_metrics_patient['AUC'])
    return list_Diag,list_Diag_GBM,list_Diag_G4A,list_Diag_G3A,list_Diag_G2A,list_Diag_G3O,list_Diag_G2O, \
           list_DiagSim, list_DiagSim_GBM, list_DiagSim_G4A, list_DiagSim_G23A, list_DiagSim_G23O, \
           list_Diag_patient, list_Diag_GBM_patient, list_Diag_G4A_patient, list_Diag_G3A_patient, list_Diag_G2A_patient, list_Diag_G3O_patient, list_Diag_G2O_patient, \
           list_DiagSim_patient, list_DiagSim_GBM_patient, list_DiagSim_G4A_patient, list_DiagSim_G23A_patient, list_DiagSim_G23O_patient, \
           list_IDH,list_1p19q,list_CDKN,list_IDH_patient,list_1p19q_patient,list_CDKN_patient\

def test_allmarker(opt, Mine_model_init, Mine_model_His, Mine_model_Cls,Mine_model_molecular, Mine_model_Graph, dataloader, gpuID,flag_IvYGAP=False,flag_cam=False):
    Mine_model_init.eval()
    # Mine_model_His.eval()
    # Mine_model_Cls.eval()
    Mine_model_molecular.eval()
    Mine_model_Graph.eval()
    gpuID = opt['gpus']

    if 1:
        MGMT_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        EGFR_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        ATRX_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        PTEN_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        TERT_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        P53_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        _710_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        PDGFRA_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        OLIG2_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}

        label_all_MGMT = []
        predicted_all_MGMT = []
        label_all_EGFR = []
        predicted_all_EGFR = []
        label_all_ATRX = []
        predicted_all_ATRX = []
        label_all_PTEN = []
        predicted_all_PTEN = []
        label_all_TERT = []
        predicted_all_TERT = []
        label_all_P53 = []
        predicted_all_P53 = []
        label_all_710 = []
        predicted_all_710 = []
        label_all_PDGFRA = []
        predicted_all_PDGFRA = []
        label_all_OLIG2 = []
        predicted_all_OLIG2 = []

    test_bar = tqdm(dataloader)

    for packs in test_bar:
        img = packs[0]  ##(BS,N,1024)
        label = packs[1]
        file_name = packs[2]
        img = img.cuda(gpuID[0])
        label = label.cuda(gpuID[0])
        label_MGMT, label_ATRX, label_EGFR, label_PTEN, label_TERT, label_P53, label_710, label_PDGFRA, label_OLIG2=label[:, 3],\
        label[:, 4],label[:, 5],label[:, 6],label[:, 7],label[:, 8],label[:, 9],label[:, 10],label[:,11],

        ### ### forward WHO 2007
        init_feature = Mine_model_init(img)  # (BS,2500,1024)



        ### molecular prediction
        encoded = Mine_model_molecular(init_feature)
        results_dict, saliency_IDH_wt, saliency_1p19q_codel = Mine_model_Graph(encoded)
        pred_MGMT_ori = results_dict['logits_MGMT']
        pred_710_ori = results_dict['logits_710']
        pred_ATRX_ori = results_dict['logits_ATRX']
        pred_EGFR_ori = results_dict['logits_EGFR']
        pred_TERT_ori = results_dict['logits_TERT']
        pred_OLIG2_ori = results_dict['logits_OLIG2']
        pred_PDGFRA_ori = results_dict['logits_PDGFRA']
        pred_PTEN_ori = results_dict['logits_PTEN']
        pred_P53_ori = results_dict['logits_P53']

        _, pred_MGMT0 = torch.max(pred_MGMT_ori.data, 1)
        pred_MGMT = pred_MGMT0.tolist()
        gt_MGMT = label_MGMT.tolist()
        _, pred_7100 = torch.max(pred_710_ori.data, 1)
        pred_710 = pred_7100.tolist()
        gt_710 = label_710.tolist()
        _, pred_ATRX0 = torch.max(pred_ATRX_ori.data, 1)
        pred_ATRX = pred_ATRX0.tolist()
        gt_ATRX = label_ATRX.tolist()
        _, pred_EGFR0 = torch.max(pred_EGFR_ori.data, 1)
        pred_EGFR = pred_EGFR0.tolist()
        gt_EGFR = label_EGFR.tolist()
        _, pred_TERT0 = torch.max(pred_TERT_ori.data, 1)
        pred_TERT = pred_TERT0.tolist()
        gt_TERT = label_TERT.tolist()
        _, pred_OLIG20 = torch.max(pred_OLIG2_ori.data, 1)
        pred_OLIG2 = pred_OLIG20.tolist()
        gt_OLIG2 = label_OLIG2.tolist()
        _, pred_PDGFRA0 = torch.max(pred_PDGFRA_ori.data, 1)
        pred_PDGFRA = pred_PDGFRA0.tolist()
        gt_PDGFRA = label_PDGFRA.tolist()
        _, pred_PTEN0 = torch.max(pred_PTEN_ori.data, 1)
        pred_PTEN = pred_PTEN0.tolist()
        gt_PTEN = label_PTEN.tolist()
        _, pred_P530 = torch.max(pred_P53_ori.data, 1)
        pred_P53 = pred_P530.tolist()
        gt_P53 = label_P53.tolist()

        ############################ WSI calculate tntp

        ##############MGMT
        if gt_MGMT[0] != 2:
            label_all_MGMT.append(gt_MGMT[0])
            predicted_all_MGMT.append(pred_MGMT_ori.detach().cpu().numpy()[0][1])
            if gt_MGMT[0] == 0 and pred_MGMT[0] == 0:
                MGMT_metrics['tn'] += 1
            if gt_MGMT[0] == 0 and pred_MGMT[0] == 1:
                MGMT_metrics['fp'] += 1
            if gt_MGMT[0] == 1 and pred_MGMT[0] == 0:
                MGMT_metrics['fn'] += 1
            if gt_MGMT[0] == 1 and pred_MGMT[0] == 1:
                MGMT_metrics['tp'] += 1
        if not flag_cam:
            ##############EGFR
            if gt_EGFR[0] != 2:
                label_all_EGFR.append(gt_EGFR[0])
                predicted_all_EGFR.append(pred_EGFR_ori.detach().cpu().numpy()[0][1])
                if gt_EGFR[0] == 0 and pred_EGFR[0] == 0:
                    EGFR_metrics['tn'] += 1
                if gt_EGFR[0] == 0 and pred_EGFR[0] == 1:
                    EGFR_metrics['fp'] += 1
                if gt_EGFR[0] == 1 and pred_EGFR[0] == 0:
                    EGFR_metrics['fn'] += 1
                if gt_EGFR[0] == 1 and pred_EGFR[0] == 1:
                    EGFR_metrics['tp'] += 1
            ##############PTEN
            if gt_PTEN[0] != 2:
                label_all_PTEN.append(gt_PTEN[0])
                predicted_all_PTEN.append(pred_PTEN_ori.detach().cpu().numpy()[0][1])
                if gt_PTEN[0] == 0 and pred_PTEN[0] == 0:
                    PTEN_metrics['tn'] += 1
                if gt_PTEN[0] == 0 and pred_PTEN[0] == 1:
                    PTEN_metrics['fp'] += 1
                if gt_PTEN[0] == 1 and pred_PTEN[0] == 0:
                    PTEN_metrics['fn'] += 1
                if gt_PTEN[0] == 1 and pred_PTEN[0] == 1:
                    PTEN_metrics['tp'] += 1
        if not flag_IvYGAP and not flag_cam:
            ##############ATRX
            if gt_ATRX[0] != 2:
                label_all_ATRX.append(gt_ATRX[0])
                predicted_all_ATRX.append(pred_ATRX_ori.detach().cpu().numpy()[0][1])
                if gt_ATRX[0] == 0 and pred_ATRX[0] == 0:
                    ATRX_metrics['tn'] += 1
                if gt_ATRX[0] == 0 and pred_ATRX[0] == 1:
                    ATRX_metrics['fp'] += 1
                if gt_ATRX[0] == 1 and pred_ATRX[0] == 0:
                    ATRX_metrics['fn'] += 1
                if gt_ATRX[0] == 1 and pred_ATRX[0] == 1:
                    ATRX_metrics['tp'] += 1
            ##############P53
            if gt_P53[0] != 2:
                label_all_P53.append(gt_P53[0])
                predicted_all_P53.append(pred_P53_ori.detach().cpu().numpy()[0][1])
                if gt_P53[0] == 0 and pred_P53[0] == 0:
                    P53_metrics['tn'] += 1
                if gt_P53[0] == 0 and pred_P53[0] == 1:
                    P53_metrics['fp'] += 1
                if gt_P53[0] == 1 and pred_P53[0] == 0:
                    P53_metrics['fn'] += 1
                if gt_P53[0] == 1 and pred_P53[0] == 1:
                    P53_metrics['tp'] += 1

            if opt['TrainingSet'] == 'All' or opt['TrainingSet'] == 'TCGA':
                ##############TERT
                if gt_TERT[0] != 2:
                    label_all_TERT.append(gt_TERT[0])
                    predicted_all_TERT.append(pred_TERT_ori.detach().cpu().numpy()[0][1])
                    if gt_TERT[0] == 0 and pred_TERT[0] == 0:
                        TERT_metrics['tn'] += 1
                    if gt_TERT[0] == 0 and pred_TERT[0] == 1:
                        TERT_metrics['fp'] += 1
                    if gt_TERT[0] == 1 and pred_TERT[0] == 0:
                        TERT_metrics['fn'] += 1
                    if gt_TERT[0] == 1 and pred_TERT[0] == 1:
                        TERT_metrics['tp'] += 1

                ##############710
                if gt_710[0] != 2:
                    label_all_710.append(gt_710[0])
                    predicted_all_710.append(pred_710_ori.detach().cpu().numpy()[0][1])
                    if gt_710[0] == 0 and pred_710[0] == 0:
                        _710_metrics['tn'] += 1
                    if gt_710[0] == 0 and pred_710[0] == 1:
                        _710_metrics['fp'] += 1
                    if gt_710[0] == 1 and pred_710[0] == 0:
                        _710_metrics['fn'] += 1
                    if gt_710[0] == 1 and pred_710[0] == 1:
                        _710_metrics['tp'] += 1
                ##############PDGFRA
                if gt_PDGFRA[0] != 2:
                    label_all_PDGFRA.append(gt_PDGFRA[0])
                    predicted_all_PDGFRA.append(pred_PDGFRA_ori.detach().cpu().numpy()[0][1])
                    if gt_PDGFRA[0] == 0 and pred_PDGFRA[0] == 0:
                        PDGFRA_metrics['tn'] += 1
                    if gt_PDGFRA[0] == 0 and pred_PDGFRA[0] == 1:
                        PDGFRA_metrics['fp'] += 1
                    if gt_PDGFRA[0] == 1 and pred_PDGFRA[0] == 0:
                        PDGFRA_metrics['fn'] += 1
                    if gt_PDGFRA[0] == 1 and pred_PDGFRA[0] == 1:
                        PDGFRA_metrics['tp'] += 1
            if opt['TrainingSet'] == 'All' or opt['TrainingSet'] == 'Tiantan':
                ##############OLIG2
                if gt_OLIG2[0] != 2:
                    label_all_OLIG2.append(gt_OLIG2[0])
                    predicted_all_OLIG2.append(pred_OLIG2_ori.detach().cpu().numpy()[0][1])
                    if gt_OLIG2[0] == 0 and pred_OLIG2[0] == 0:
                        OLIG2_metrics['tn'] += 1
                    if gt_OLIG2[0] == 0 and pred_OLIG2[0] == 1:
                        OLIG2_metrics['fp'] += 1
                    if gt_OLIG2[0] == 1 and pred_OLIG2[0] == 0:
                        OLIG2_metrics['fn'] += 1
                    if gt_OLIG2[0] == 1 and pred_OLIG2[0] == 1:
                        OLIG2_metrics['tp'] += 1

    ##########  MGMT
    Acc_MGMT = (MGMT_metrics['tp'] + MGMT_metrics['tn']) / (
            MGMT_metrics['tp'] + MGMT_metrics['tn'] + MGMT_metrics['fp'] + MGMT_metrics['fn'] + 0.000001)
    MGMT_metrics['sen'] = (MGMT_metrics['tp']) / (MGMT_metrics['tp'] + MGMT_metrics['fn'] + 0.000001)  # recall
    MGMT_metrics['spec'] = (MGMT_metrics['tn']) / (MGMT_metrics['tn'] + MGMT_metrics['fp'] + 0.000001)
    MGMT_metrics['pre'] = (MGMT_metrics['tp']) / (MGMT_metrics['tp'] + MGMT_metrics['fp'] + 0.000001)
    MGMT_metrics['recall'] = MGMT_metrics['sen']
    MGMT_metrics['f1'] = (2 * MGMT_metrics['pre'] * MGMT_metrics['recall']) / (
            MGMT_metrics['pre'] + MGMT_metrics['recall'] + 0.000001)
    MGMT_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_MGMT),
                                                y_score=np.array(predicted_all_MGMT))
    list_MGMT = (Acc_MGMT, MGMT_metrics['sen'], MGMT_metrics['spec'], MGMT_metrics['pre'], MGMT_metrics['recall']
                 , MGMT_metrics['f1'], MGMT_metrics['AUC'])

    if not flag_cam:
        ##########  EGFR
        if not flag_IvYGAP:
            Acc_EGFR = (EGFR_metrics['tp'] + EGFR_metrics['tn']) / (
                    EGFR_metrics['tp'] + EGFR_metrics['tn'] + EGFR_metrics['fp'] + EGFR_metrics['fn'] + 0.000001)
            EGFR_metrics['sen'] = (EGFR_metrics['tp']) / (EGFR_metrics['tp'] + EGFR_metrics['fn'] + 0.000001)  # recall
            EGFR_metrics['spec'] = (EGFR_metrics['tn']) / (EGFR_metrics['tn'] + EGFR_metrics['fp'] + 0.000001)
            EGFR_metrics['pre'] = (EGFR_metrics['tp']) / (EGFR_metrics['tp'] + EGFR_metrics['fp'] + 0.000001)
            EGFR_metrics['recall'] = EGFR_metrics['sen']
            EGFR_metrics['f1'] = (2 * EGFR_metrics['pre'] * EGFR_metrics['recall']) / (
                    EGFR_metrics['pre'] + EGFR_metrics['recall'] + 0.000001)
            EGFR_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_EGFR),
                                                        y_score=np.array(predicted_all_EGFR))
            list_EGFR = (Acc_EGFR, EGFR_metrics['sen'], EGFR_metrics['spec'], EGFR_metrics['pre'], EGFR_metrics['recall']
                     , EGFR_metrics['f1'], EGFR_metrics['AUC'])
        else:
            Acc_EGFR = (EGFR_metrics['tp'] + EGFR_metrics['tn']) / (
                    EGFR_metrics['tp'] + EGFR_metrics['tn'] + EGFR_metrics['fp'] + EGFR_metrics['fn'] + 0.000001)
            EGFR_metrics['sen'] = (EGFR_metrics['tp']) / (EGFR_metrics['tp'] + EGFR_metrics['fn'] + 0.000001)  # recall
            EGFR_metrics['spec'] =0
            EGFR_metrics['pre'] = 0
            EGFR_metrics['recall'] = EGFR_metrics['sen']
            EGFR_metrics['f1'] = 0
            EGFR_metrics['AUC'] = 0
            list_EGFR = (
            Acc_EGFR, EGFR_metrics['sen'], EGFR_metrics['spec'], EGFR_metrics['pre'], EGFR_metrics['recall']
            , EGFR_metrics['f1'], EGFR_metrics['AUC'])
        ##########  PTEN
        Acc_PTEN = (PTEN_metrics['tp'] + PTEN_metrics['tn']) / (
                PTEN_metrics['tp'] + PTEN_metrics['tn'] + PTEN_metrics['fp'] + PTEN_metrics['fn'] + 0.000001)
        PTEN_metrics['sen'] = (PTEN_metrics['tp']) / (PTEN_metrics['tp'] + PTEN_metrics['fn'] + 0.000001)  # recall
        PTEN_metrics['spec'] = (PTEN_metrics['tn']) / (PTEN_metrics['tn'] + PTEN_metrics['fp'] + 0.000001)
        PTEN_metrics['pre'] = (PTEN_metrics['tp']) / (PTEN_metrics['tp'] + PTEN_metrics['fp'] + 0.000001)
        PTEN_metrics['recall'] = PTEN_metrics['sen']
        PTEN_metrics['f1'] = (2 * PTEN_metrics['pre'] * PTEN_metrics['recall']) / (
                PTEN_metrics['pre'] + PTEN_metrics['recall'] + 0.000001)
        PTEN_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_PTEN),
                                                    y_score=np.array(predicted_all_PTEN))
        list_PTEN = (Acc_PTEN, PTEN_metrics['sen'], PTEN_metrics['spec'], PTEN_metrics['pre'], PTEN_metrics['recall']
                     , PTEN_metrics['f1'], PTEN_metrics['AUC'])

    if not flag_IvYGAP and not flag_cam:
        ##########  ATRX
        Acc_ATRX = (ATRX_metrics['tp'] + ATRX_metrics['tn']) / (
                ATRX_metrics['tp'] + ATRX_metrics['tn'] + ATRX_metrics['fp'] + ATRX_metrics['fn'] + 0.000001)
        ATRX_metrics['sen'] = (ATRX_metrics['tp']) / (ATRX_metrics['tp'] + ATRX_metrics['fn'] + 0.000001)  # recall
        ATRX_metrics['spec'] = (ATRX_metrics['tn']) / (ATRX_metrics['tn'] + ATRX_metrics['fp'] + 0.000001)
        ATRX_metrics['pre'] = (ATRX_metrics['tp']) / (ATRX_metrics['tp'] + ATRX_metrics['fp'] + 0.000001)
        ATRX_metrics['recall'] = ATRX_metrics['sen']
        ATRX_metrics['f1'] = (2 * ATRX_metrics['pre'] * ATRX_metrics['recall']) / (
                ATRX_metrics['pre'] + ATRX_metrics['recall'] + 0.000001)
        ATRX_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_ATRX),
                                                    y_score=np.array(predicted_all_ATRX))
        list_ATRX = (Acc_ATRX, ATRX_metrics['sen'], ATRX_metrics['spec'], ATRX_metrics['pre'], ATRX_metrics['recall']
                     , ATRX_metrics['f1'], ATRX_metrics['AUC'])
        ##########  P53
        Acc_P53 = (P53_metrics['tp'] + P53_metrics['tn']) / (
                P53_metrics['tp'] + P53_metrics['tn'] + P53_metrics['fp'] + P53_metrics['fn'] + 0.000001)
        P53_metrics['sen'] = (P53_metrics['tp']) / (P53_metrics['tp'] + P53_metrics['fn'] + 0.000001)  # recall
        P53_metrics['spec'] = (P53_metrics['tn']) / (P53_metrics['tn'] + P53_metrics['fp'] + 0.000001)
        P53_metrics['pre'] = (P53_metrics['tp']) / (P53_metrics['tp'] + P53_metrics['fp'] + 0.000001)
        P53_metrics['recall'] = P53_metrics['sen']
        P53_metrics['f1'] = (2 * P53_metrics['pre'] * P53_metrics['recall']) / (
                P53_metrics['pre'] + P53_metrics['recall'] + 0.000001)
        P53_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_P53),
                                                   y_score=np.array(predicted_all_P53))
        list_P53 = (Acc_P53, P53_metrics['sen'], P53_metrics['spec'], P53_metrics['pre'], P53_metrics['recall']
                    , P53_metrics['f1'], P53_metrics['AUC'])

        if opt['TrainingSet'] == 'All' or opt['TrainingSet'] == 'TCGA':
            ##########  710
            Acc_710 = (_710_metrics['tp'] + _710_metrics['tn']) / (
                    _710_metrics['tp'] + _710_metrics['tn'] + _710_metrics['fp'] + _710_metrics['fn'] + 0.000001)
            _710_metrics['sen'] = (_710_metrics['tp']) / (_710_metrics['tp'] + _710_metrics['fn'] + 0.000001)  # recall
            _710_metrics['spec'] = (_710_metrics['tn']) / (_710_metrics['tn'] + _710_metrics['fp'] + 0.000001)
            _710_metrics['pre'] = (_710_metrics['tp']) / (_710_metrics['tp'] + _710_metrics['fp'] + 0.000001)
            _710_metrics['recall'] = _710_metrics['sen']
            _710_metrics['f1'] = (2 * _710_metrics['pre'] * _710_metrics['recall']) / (
                    _710_metrics['pre'] + _710_metrics['recall'] + 0.000001)
            _710_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_710),
                                                        y_score=np.array(predicted_all_710))
            list_710 = (Acc_710, _710_metrics['sen'], _710_metrics['spec'], _710_metrics['pre'], _710_metrics['recall']
                        , _710_metrics['f1'], _710_metrics['AUC'])
            ##########  TERT
            Acc_TERT = (TERT_metrics['tp'] + TERT_metrics['tn']) / (
                    TERT_metrics['tp'] + TERT_metrics['tn'] + TERT_metrics['fp'] + TERT_metrics['fn'] + 0.000001)
            TERT_metrics['sen'] = (TERT_metrics['tp']) / (TERT_metrics['tp'] + TERT_metrics['fn'] + 0.000001)  # recall
            TERT_metrics['spec'] = (TERT_metrics['tn']) / (TERT_metrics['tn'] + TERT_metrics['fp'] + 0.000001)
            TERT_metrics['pre'] = (TERT_metrics['tp']) / (TERT_metrics['tp'] + TERT_metrics['fp'] + 0.000001)
            TERT_metrics['recall'] = TERT_metrics['sen']
            TERT_metrics['f1'] = (2 * TERT_metrics['pre'] * TERT_metrics['recall']) / (
                    TERT_metrics['pre'] + TERT_metrics['recall'] + 0.000001)
            TERT_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_TERT),
                                                        y_score=np.array(predicted_all_TERT))
            list_TERT = (
            Acc_TERT, TERT_metrics['sen'], TERT_metrics['spec'], TERT_metrics['pre'], TERT_metrics['recall']
            , TERT_metrics['f1'], TERT_metrics['AUC'])
            ##########  PDGFRA
            Acc_PDGFRA = (PDGFRA_metrics['tp'] + PDGFRA_metrics['tn']) / (
                    PDGFRA_metrics['tp'] + PDGFRA_metrics['tn'] + PDGFRA_metrics['fp'] + PDGFRA_metrics[
                'fn'] + 0.000001)
            PDGFRA_metrics['sen'] = (PDGFRA_metrics['tp']) / (
                        PDGFRA_metrics['tp'] + PDGFRA_metrics['fn'] + 0.000001)  # recall
            PDGFRA_metrics['spec'] = (PDGFRA_metrics['tn']) / (PDGFRA_metrics['tn'] + PDGFRA_metrics['fp'] + 0.000001)
            PDGFRA_metrics['pre'] = (PDGFRA_metrics['tp']) / (PDGFRA_metrics['tp'] + PDGFRA_metrics['fp'] + 0.000001)
            PDGFRA_metrics['recall'] = PDGFRA_metrics['sen']
            PDGFRA_metrics['f1'] = (2 * PDGFRA_metrics['pre'] * PDGFRA_metrics['recall']) / (
                    PDGFRA_metrics['pre'] + PDGFRA_metrics['recall'] + 0.000001)
            PDGFRA_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_PDGFRA),
                                                          y_score=np.array(predicted_all_PDGFRA))
            list_PDGFRA = (
            Acc_PDGFRA, PDGFRA_metrics['sen'], PDGFRA_metrics['spec'], PDGFRA_metrics['pre'], PDGFRA_metrics['recall']
            , PDGFRA_metrics['f1'], PDGFRA_metrics['AUC'])
        if opt['TrainingSet'] == 'All' or opt['TrainingSet'] == 'Tiantan':
            ##########  OLIG2
            Acc_OLIG2 = (OLIG2_metrics['tp'] + OLIG2_metrics['tn']) / (
                    OLIG2_metrics['tp'] + OLIG2_metrics['tn'] + OLIG2_metrics['fp'] + OLIG2_metrics['fn'] + 0.000001)
            OLIG2_metrics['sen'] = (OLIG2_metrics['tp']) / (
                        OLIG2_metrics['tp'] + OLIG2_metrics['fn'] + 0.000001)  # recall
            OLIG2_metrics['spec'] = (OLIG2_metrics['tn']) / (OLIG2_metrics['tn'] + OLIG2_metrics['fp'] + 0.000001)
            OLIG2_metrics['pre'] = (OLIG2_metrics['tp']) / (OLIG2_metrics['tp'] + OLIG2_metrics['fp'] + 0.000001)
            OLIG2_metrics['recall'] = OLIG2_metrics['sen']
            OLIG2_metrics['f1'] = (2 * OLIG2_metrics['pre'] * OLIG2_metrics['recall']) / (
                    OLIG2_metrics['pre'] + OLIG2_metrics['recall'] + 0.000001)
            OLIG2_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_OLIG2),
                                                         y_score=np.array(predicted_all_OLIG2))
            list_OLIG2 = (
            Acc_OLIG2, OLIG2_metrics['sen'], OLIG2_metrics['spec'], OLIG2_metrics['pre'], OLIG2_metrics['recall']
            , OLIG2_metrics['f1'], OLIG2_metrics['AUC'])


    if not flag_IvYGAP and not flag_cam:
        if opt['TrainingSet'] == 'All':
            return list_MGMT, list_EGFR, list_ATRX,list_PTEN, list_TERT, list_P53,list_710, list_PDGFRA, list_OLIG2
        if opt['TrainingSet'] == 'TCGA':
            return list_MGMT, list_EGFR, list_ATRX, list_PTEN, list_TERT, list_P53, list_710, list_PDGFRA
        if opt['TrainingSet'] == 'Tiantan':
            return list_MGMT, list_EGFR, list_ATRX,list_PTEN, list_P53, list_OLIG2
    elif flag_IvYGAP:
        return list_MGMT, list_EGFR, list_PTEN
    elif flag_cam:
        return list_MGMT

def test_marker_ViT(opt, Mine_model_init, Mine_model_body, Mine_model_Cls, dataloader, gpuID,flag_IvYGAP=False,flag_cam=False):
    Mine_model_init.eval()
    Mine_model_body.eval()
    Mine_model_Cls.eval()

    gpuID = opt['gpus']

    if 1:
        MGMT_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                        'AUC': 0}

        label_all_MGMT = []
        predicted_all_MGMT = []


    test_bar = tqdm(dataloader)

    for packs in test_bar:
        img = packs[0]  ##(BS,N,1024)
        label = packs[1]
        file_name = packs[2]
        img = img.cuda(gpuID[0])
        label = label.cuda(gpuID[0])
        label_MGMT= label[:,0]

        ### ### forward WHO 2007
        init_feature = Mine_model_init(img)  # (BS,2500,1024)
        hidden_states = Mine_model_body(init_feature)
        results_dict = Mine_model_Cls(hidden_states)

        pred_MGMT_ori = results_dict['logits']


        _, pred_MGMT0 = torch.max(pred_MGMT_ori.data, 1)
        pred_MGMT = pred_MGMT0.tolist()
        gt_MGMT = label_MGMT.tolist()

        ############################ WSI calculate tntp

        ##############MGMT
        if gt_MGMT[0] != 2:
            label_all_MGMT.append(gt_MGMT[0])
            predicted_all_MGMT.append(pred_MGMT_ori.detach().cpu().numpy()[0][1])
            if gt_MGMT[0] == 0 and pred_MGMT[0] == 0:
                MGMT_metrics['tn'] += 1
            if gt_MGMT[0] == 0 and pred_MGMT[0] == 1:
                MGMT_metrics['fp'] += 1
            if gt_MGMT[0] == 1 and pred_MGMT[0] == 0:
                MGMT_metrics['fn'] += 1
            if gt_MGMT[0] == 1 and pred_MGMT[0] == 1:
                MGMT_metrics['tp'] += 1

    ##########  MGMT
    Acc_MGMT = (MGMT_metrics['tp'] + MGMT_metrics['tn']) / (
            MGMT_metrics['tp'] + MGMT_metrics['tn'] + MGMT_metrics['fp'] + MGMT_metrics['fn'] + 0.000001)
    MGMT_metrics['sen'] = (MGMT_metrics['tp']) / (MGMT_metrics['tp'] + MGMT_metrics['fn'] + 0.000001)  # recall
    MGMT_metrics['spec'] = (MGMT_metrics['tn']) / (MGMT_metrics['tn'] + MGMT_metrics['fp'] + 0.000001)
    MGMT_metrics['pre'] = (MGMT_metrics['tp']) / (MGMT_metrics['tp'] + MGMT_metrics['fp'] + 0.000001)
    MGMT_metrics['recall'] = MGMT_metrics['sen']
    MGMT_metrics['f1'] = (2 * MGMT_metrics['pre'] * MGMT_metrics['recall']) / (
            MGMT_metrics['pre'] + MGMT_metrics['recall'] + 0.000001)
    MGMT_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_MGMT),
                                                y_score=np.array(predicted_all_MGMT))
    list_MGMT = (Acc_MGMT, MGMT_metrics['sen'], MGMT_metrics['spec'], MGMT_metrics['pre'], MGMT_metrics['recall']
                 , MGMT_metrics['f1'], MGMT_metrics['AUC'])



    return list_MGMT




def test_allmarker_endtoend(opt, Mine_model_init,Mine_model_body,Mine_model_Cls, dataloader, gpuID,flag_IvYGAP=False,flag_cam=False):
    Mine_model_init.eval()
    Mine_model_body.eval()
    Mine_model_Cls.eval()

    gpuID = opt['gpus']

    if 1:
        MGMT_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        EGFR_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        ATRX_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        PTEN_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        TERT_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        P53_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        _710_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}
        PDGFRA_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        OLIG2_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}

        label_all_MGMT = []
        predicted_all_MGMT = []
        label_all_EGFR = []
        predicted_all_EGFR = []
        label_all_ATRX = []
        predicted_all_ATRX = []
        label_all_PTEN = []
        predicted_all_PTEN = []
        label_all_TERT = []
        predicted_all_TERT = []
        label_all_P53 = []
        predicted_all_P53 = []
        label_all_710 = []
        predicted_all_710 = []
        label_all_PDGFRA = []
        predicted_all_PDGFRA = []
        label_all_OLIG2 = []
        predicted_all_OLIG2 = []

    test_bar = tqdm(dataloader)

    for packs in test_bar:
        img = packs[0]  ##(BS,N,1024)
        label = packs[1]
        file_name = packs[2]
        img = img.cuda(gpuID[0])
        label = label.cuda(gpuID[0])
        label_MGMT, label_ATRX, label_EGFR, label_PTEN, label_TERT, label_P53, label_710, label_PDGFRA, label_OLIG2=label[:, 3],\
        label[:, 4],label[:, 5],label[:, 6],label[:, 7],label[:, 8],label[:, 9],label[:, 10],label[:,11],

        ### ### forward WHO 2007
        init_feature = Mine_model_init(img)  # (BS,2500,1024)
        hidden_states = Mine_model_body(init_feature)
        results_dict = Mine_model_Cls(hidden_states)

        pred_MGMT_ori = results_dict['logits_MGMT']
        pred_710_ori = results_dict['logits_710']
        pred_ATRX_ori = results_dict['logits_ATRX']
        pred_EGFR_ori = results_dict['logits_EGFR']
        pred_TERT_ori = results_dict['logits_TERT']
        pred_OLIG2_ori = results_dict['logits_OLIG2']
        pred_PDGFRA_ori = results_dict['logits_PDGFRA']
        pred_PTEN_ori = results_dict['logits_PTEN']
        pred_P53_ori = results_dict['logits_P53']

        _, pred_MGMT0 = torch.max(pred_MGMT_ori.data, 1)
        pred_MGMT = pred_MGMT0.tolist()
        gt_MGMT = label_MGMT.tolist()
        _, pred_7100 = torch.max(pred_710_ori.data, 1)
        pred_710 = pred_7100.tolist()
        gt_710 = label_710.tolist()
        _, pred_ATRX0 = torch.max(pred_ATRX_ori.data, 1)
        pred_ATRX = pred_ATRX0.tolist()
        gt_ATRX = label_ATRX.tolist()
        _, pred_EGFR0 = torch.max(pred_EGFR_ori.data, 1)
        pred_EGFR = pred_EGFR0.tolist()
        gt_EGFR = label_EGFR.tolist()
        _, pred_TERT0 = torch.max(pred_TERT_ori.data, 1)
        pred_TERT = pred_TERT0.tolist()
        gt_TERT = label_TERT.tolist()
        _, pred_OLIG20 = torch.max(pred_OLIG2_ori.data, 1)
        pred_OLIG2 = pred_OLIG20.tolist()
        gt_OLIG2 = label_OLIG2.tolist()
        _, pred_PDGFRA0 = torch.max(pred_PDGFRA_ori.data, 1)
        pred_PDGFRA = pred_PDGFRA0.tolist()
        gt_PDGFRA = label_PDGFRA.tolist()
        _, pred_PTEN0 = torch.max(pred_PTEN_ori.data, 1)
        pred_PTEN = pred_PTEN0.tolist()
        gt_PTEN = label_PTEN.tolist()
        _, pred_P530 = torch.max(pred_P53_ori.data, 1)
        pred_P53 = pred_P530.tolist()
        gt_P53 = label_P53.tolist()

        ############################ WSI calculate tntp

        ##############MGMT
        if gt_MGMT[0] != 2:
            label_all_MGMT.append(gt_MGMT[0])
            predicted_all_MGMT.append(pred_MGMT_ori.detach().cpu().numpy()[0][1])
            if gt_MGMT[0] == 0 and pred_MGMT[0] == 0:
                MGMT_metrics['tn'] += 1
            if gt_MGMT[0] == 0 and pred_MGMT[0] == 1:
                MGMT_metrics['fp'] += 1
            if gt_MGMT[0] == 1 and pred_MGMT[0] == 0:
                MGMT_metrics['fn'] += 1
            if gt_MGMT[0] == 1 and pred_MGMT[0] == 1:
                MGMT_metrics['tp'] += 1
        if not flag_cam:
            ##############EGFR
            if gt_EGFR[0] != 2:
                label_all_EGFR.append(gt_EGFR[0])
                predicted_all_EGFR.append(pred_EGFR_ori.detach().cpu().numpy()[0][1])
                if gt_EGFR[0] == 0 and pred_EGFR[0] == 0:
                    EGFR_metrics['tn'] += 1
                if gt_EGFR[0] == 0 and pred_EGFR[0] == 1:
                    EGFR_metrics['fp'] += 1
                if gt_EGFR[0] == 1 and pred_EGFR[0] == 0:
                    EGFR_metrics['fn'] += 1
                if gt_EGFR[0] == 1 and pred_EGFR[0] == 1:
                    EGFR_metrics['tp'] += 1
            ##############PTEN
            if gt_PTEN[0] != 2:
                label_all_PTEN.append(gt_PTEN[0])
                predicted_all_PTEN.append(pred_PTEN_ori.detach().cpu().numpy()[0][1])
                if gt_PTEN[0] == 0 and pred_PTEN[0] == 0:
                    PTEN_metrics['tn'] += 1
                if gt_PTEN[0] == 0 and pred_PTEN[0] == 1:
                    PTEN_metrics['fp'] += 1
                if gt_PTEN[0] == 1 and pred_PTEN[0] == 0:
                    PTEN_metrics['fn'] += 1
                if gt_PTEN[0] == 1 and pred_PTEN[0] == 1:
                    PTEN_metrics['tp'] += 1
        if not flag_IvYGAP and not flag_cam:
            ##############ATRX
            if gt_ATRX[0] != 2:
                label_all_ATRX.append(gt_ATRX[0])
                predicted_all_ATRX.append(pred_ATRX_ori.detach().cpu().numpy()[0][1])
                if gt_ATRX[0] == 0 and pred_ATRX[0] == 0:
                    ATRX_metrics['tn'] += 1
                if gt_ATRX[0] == 0 and pred_ATRX[0] == 1:
                    ATRX_metrics['fp'] += 1
                if gt_ATRX[0] == 1 and pred_ATRX[0] == 0:
                    ATRX_metrics['fn'] += 1
                if gt_ATRX[0] == 1 and pred_ATRX[0] == 1:
                    ATRX_metrics['tp'] += 1
            ##############P53
            if gt_P53[0] != 2:
                label_all_P53.append(gt_P53[0])
                predicted_all_P53.append(pred_P53_ori.detach().cpu().numpy()[0][1])
                if gt_P53[0] == 0 and pred_P53[0] == 0:
                    P53_metrics['tn'] += 1
                if gt_P53[0] == 0 and pred_P53[0] == 1:
                    P53_metrics['fp'] += 1
                if gt_P53[0] == 1 and pred_P53[0] == 0:
                    P53_metrics['fn'] += 1
                if gt_P53[0] == 1 and pred_P53[0] == 1:
                    P53_metrics['tp'] += 1

            if opt['TrainingSet'] == 'All' or opt['TrainingSet'] == 'TCGA':
                ##############TERT
                if gt_TERT[0] != 2:
                    label_all_TERT.append(gt_TERT[0])
                    predicted_all_TERT.append(pred_TERT_ori.detach().cpu().numpy()[0][1])
                    if gt_TERT[0] == 0 and pred_TERT[0] == 0:
                        TERT_metrics['tn'] += 1
                    if gt_TERT[0] == 0 and pred_TERT[0] == 1:
                        TERT_metrics['fp'] += 1
                    if gt_TERT[0] == 1 and pred_TERT[0] == 0:
                        TERT_metrics['fn'] += 1
                    if gt_TERT[0] == 1 and pred_TERT[0] == 1:
                        TERT_metrics['tp'] += 1

                ##############710
                if gt_710[0] != 2:
                    label_all_710.append(gt_710[0])
                    predicted_all_710.append(pred_710_ori.detach().cpu().numpy()[0][1])
                    if gt_710[0] == 0 and pred_710[0] == 0:
                        _710_metrics['tn'] += 1
                    if gt_710[0] == 0 and pred_710[0] == 1:
                        _710_metrics['fp'] += 1
                    if gt_710[0] == 1 and pred_710[0] == 0:
                        _710_metrics['fn'] += 1
                    if gt_710[0] == 1 and pred_710[0] == 1:
                        _710_metrics['tp'] += 1
                ##############PDGFRA
                if gt_PDGFRA[0] != 2:
                    label_all_PDGFRA.append(gt_PDGFRA[0])
                    predicted_all_PDGFRA.append(pred_PDGFRA_ori.detach().cpu().numpy()[0][1])
                    if gt_PDGFRA[0] == 0 and pred_PDGFRA[0] == 0:
                        PDGFRA_metrics['tn'] += 1
                    if gt_PDGFRA[0] == 0 and pred_PDGFRA[0] == 1:
                        PDGFRA_metrics['fp'] += 1
                    if gt_PDGFRA[0] == 1 and pred_PDGFRA[0] == 0:
                        PDGFRA_metrics['fn'] += 1
                    if gt_PDGFRA[0] == 1 and pred_PDGFRA[0] == 1:
                        PDGFRA_metrics['tp'] += 1
            if opt['TrainingSet'] == 'All' or opt['TrainingSet'] == 'Tiantan':
                ##############OLIG2
                if gt_OLIG2[0] != 2:
                    label_all_OLIG2.append(gt_OLIG2[0])
                    predicted_all_OLIG2.append(pred_OLIG2_ori.detach().cpu().numpy()[0][1])
                    if gt_OLIG2[0] == 0 and pred_OLIG2[0] == 0:
                        OLIG2_metrics['tn'] += 1
                    if gt_OLIG2[0] == 0 and pred_OLIG2[0] == 1:
                        OLIG2_metrics['fp'] += 1
                    if gt_OLIG2[0] == 1 and pred_OLIG2[0] == 0:
                        OLIG2_metrics['fn'] += 1
                    if gt_OLIG2[0] == 1 and pred_OLIG2[0] == 1:
                        OLIG2_metrics['tp'] += 1

    ##########  MGMT
    Acc_MGMT = (MGMT_metrics['tp'] + MGMT_metrics['tn']) / (
            MGMT_metrics['tp'] + MGMT_metrics['tn'] + MGMT_metrics['fp'] + MGMT_metrics['fn'] + 0.000001)
    MGMT_metrics['sen'] = (MGMT_metrics['tp']) / (MGMT_metrics['tp'] + MGMT_metrics['fn'] + 0.000001)  # recall
    MGMT_metrics['spec'] = (MGMT_metrics['tn']) / (MGMT_metrics['tn'] + MGMT_metrics['fp'] + 0.000001)
    MGMT_metrics['pre'] = (MGMT_metrics['tp']) / (MGMT_metrics['tp'] + MGMT_metrics['fp'] + 0.000001)
    MGMT_metrics['recall'] = MGMT_metrics['sen']
    MGMT_metrics['f1'] = (2 * MGMT_metrics['pre'] * MGMT_metrics['recall']) / (
            MGMT_metrics['pre'] + MGMT_metrics['recall'] + 0.000001)
    MGMT_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_MGMT),
                                                y_score=np.array(predicted_all_MGMT))
    list_MGMT = (Acc_MGMT, MGMT_metrics['sen'], MGMT_metrics['spec'], MGMT_metrics['pre'], MGMT_metrics['recall']
                 , MGMT_metrics['f1'], MGMT_metrics['AUC'])

    if not flag_cam:
        ##########  EGFR
        if not flag_IvYGAP:
            Acc_EGFR = (EGFR_metrics['tp'] + EGFR_metrics['tn']) / (
                    EGFR_metrics['tp'] + EGFR_metrics['tn'] + EGFR_metrics['fp'] + EGFR_metrics['fn'] + 0.000001)
            EGFR_metrics['sen'] = (EGFR_metrics['tp']) / (EGFR_metrics['tp'] + EGFR_metrics['fn'] + 0.000001)  # recall
            EGFR_metrics['spec'] = (EGFR_metrics['tn']) / (EGFR_metrics['tn'] + EGFR_metrics['fp'] + 0.000001)
            EGFR_metrics['pre'] = (EGFR_metrics['tp']) / (EGFR_metrics['tp'] + EGFR_metrics['fp'] + 0.000001)
            EGFR_metrics['recall'] = EGFR_metrics['sen']
            EGFR_metrics['f1'] = (2 * EGFR_metrics['pre'] * EGFR_metrics['recall']) / (
                    EGFR_metrics['pre'] + EGFR_metrics['recall'] + 0.000001)
            EGFR_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_EGFR),
                                                        y_score=np.array(predicted_all_EGFR))
            list_EGFR = (Acc_EGFR, EGFR_metrics['sen'], EGFR_metrics['spec'], EGFR_metrics['pre'], EGFR_metrics['recall']
                     , EGFR_metrics['f1'], EGFR_metrics['AUC'])
        else:
            Acc_EGFR = (EGFR_metrics['tp'] + EGFR_metrics['tn']) / (
                    EGFR_metrics['tp'] + EGFR_metrics['tn'] + EGFR_metrics['fp'] + EGFR_metrics['fn'] + 0.000001)
            EGFR_metrics['sen'] = (EGFR_metrics['tp']) / (EGFR_metrics['tp'] + EGFR_metrics['fn'] + 0.000001)  # recall
            EGFR_metrics['spec'] =0
            EGFR_metrics['pre'] = 0
            EGFR_metrics['recall'] = EGFR_metrics['sen']
            EGFR_metrics['f1'] = 0
            EGFR_metrics['AUC'] = 0
            list_EGFR = (
            Acc_EGFR, EGFR_metrics['sen'], EGFR_metrics['spec'], EGFR_metrics['pre'], EGFR_metrics['recall']
            , EGFR_metrics['f1'], EGFR_metrics['AUC'])
        ##########  PTEN
        Acc_PTEN = (PTEN_metrics['tp'] + PTEN_metrics['tn']) / (
                PTEN_metrics['tp'] + PTEN_metrics['tn'] + PTEN_metrics['fp'] + PTEN_metrics['fn'] + 0.000001)
        PTEN_metrics['sen'] = (PTEN_metrics['tp']) / (PTEN_metrics['tp'] + PTEN_metrics['fn'] + 0.000001)  # recall
        PTEN_metrics['spec'] = (PTEN_metrics['tn']) / (PTEN_metrics['tn'] + PTEN_metrics['fp'] + 0.000001)
        PTEN_metrics['pre'] = (PTEN_metrics['tp']) / (PTEN_metrics['tp'] + PTEN_metrics['fp'] + 0.000001)
        PTEN_metrics['recall'] = PTEN_metrics['sen']
        PTEN_metrics['f1'] = (2 * PTEN_metrics['pre'] * PTEN_metrics['recall']) / (
                PTEN_metrics['pre'] + PTEN_metrics['recall'] + 0.000001)
        PTEN_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_PTEN),
                                                    y_score=np.array(predicted_all_PTEN))
        list_PTEN = (Acc_PTEN, PTEN_metrics['sen'], PTEN_metrics['spec'], PTEN_metrics['pre'], PTEN_metrics['recall']
                     , PTEN_metrics['f1'], PTEN_metrics['AUC'])

    if not flag_IvYGAP and not flag_cam:
        ##########  ATRX
        Acc_ATRX = (ATRX_metrics['tp'] + ATRX_metrics['tn']) / (
                ATRX_metrics['tp'] + ATRX_metrics['tn'] + ATRX_metrics['fp'] + ATRX_metrics['fn'] + 0.000001)
        ATRX_metrics['sen'] = (ATRX_metrics['tp']) / (ATRX_metrics['tp'] + ATRX_metrics['fn'] + 0.000001)  # recall
        ATRX_metrics['spec'] = (ATRX_metrics['tn']) / (ATRX_metrics['tn'] + ATRX_metrics['fp'] + 0.000001)
        ATRX_metrics['pre'] = (ATRX_metrics['tp']) / (ATRX_metrics['tp'] + ATRX_metrics['fp'] + 0.000001)
        ATRX_metrics['recall'] = ATRX_metrics['sen']
        ATRX_metrics['f1'] = (2 * ATRX_metrics['pre'] * ATRX_metrics['recall']) / (
                ATRX_metrics['pre'] + ATRX_metrics['recall'] + 0.000001)
        ATRX_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_ATRX),
                                                    y_score=np.array(predicted_all_ATRX))
        list_ATRX = (Acc_ATRX, ATRX_metrics['sen'], ATRX_metrics['spec'], ATRX_metrics['pre'], ATRX_metrics['recall']
                     , ATRX_metrics['f1'], ATRX_metrics['AUC'])
        ##########  P53
        Acc_P53 = (P53_metrics['tp'] + P53_metrics['tn']) / (
                P53_metrics['tp'] + P53_metrics['tn'] + P53_metrics['fp'] + P53_metrics['fn'] + 0.000001)
        P53_metrics['sen'] = (P53_metrics['tp']) / (P53_metrics['tp'] + P53_metrics['fn'] + 0.000001)  # recall
        P53_metrics['spec'] = (P53_metrics['tn']) / (P53_metrics['tn'] + P53_metrics['fp'] + 0.000001)
        P53_metrics['pre'] = (P53_metrics['tp']) / (P53_metrics['tp'] + P53_metrics['fp'] + 0.000001)
        P53_metrics['recall'] = P53_metrics['sen']
        P53_metrics['f1'] = (2 * P53_metrics['pre'] * P53_metrics['recall']) / (
                P53_metrics['pre'] + P53_metrics['recall'] + 0.000001)
        P53_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_P53),
                                                   y_score=np.array(predicted_all_P53))
        list_P53 = (Acc_P53, P53_metrics['sen'], P53_metrics['spec'], P53_metrics['pre'], P53_metrics['recall']
                    , P53_metrics['f1'], P53_metrics['AUC'])

        if opt['TrainingSet'] == 'All' or opt['TrainingSet'] == 'TCGA':
            ##########  710
            Acc_710 = (_710_metrics['tp'] + _710_metrics['tn']) / (
                    _710_metrics['tp'] + _710_metrics['tn'] + _710_metrics['fp'] + _710_metrics['fn'] + 0.000001)
            _710_metrics['sen'] = (_710_metrics['tp']) / (_710_metrics['tp'] + _710_metrics['fn'] + 0.000001)  # recall
            _710_metrics['spec'] = (_710_metrics['tn']) / (_710_metrics['tn'] + _710_metrics['fp'] + 0.000001)
            _710_metrics['pre'] = (_710_metrics['tp']) / (_710_metrics['tp'] + _710_metrics['fp'] + 0.000001)
            _710_metrics['recall'] = _710_metrics['sen']
            _710_metrics['f1'] = (2 * _710_metrics['pre'] * _710_metrics['recall']) / (
                    _710_metrics['pre'] + _710_metrics['recall'] + 0.000001)
            _710_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_710),
                                                        y_score=np.array(predicted_all_710))
            list_710 = (Acc_710, _710_metrics['sen'], _710_metrics['spec'], _710_metrics['pre'], _710_metrics['recall']
                        , _710_metrics['f1'], _710_metrics['AUC'])
            ##########  TERT
            Acc_TERT = (TERT_metrics['tp'] + TERT_metrics['tn']) / (
                    TERT_metrics['tp'] + TERT_metrics['tn'] + TERT_metrics['fp'] + TERT_metrics['fn'] + 0.000001)
            TERT_metrics['sen'] = (TERT_metrics['tp']) / (TERT_metrics['tp'] + TERT_metrics['fn'] + 0.000001)  # recall
            TERT_metrics['spec'] = (TERT_metrics['tn']) / (TERT_metrics['tn'] + TERT_metrics['fp'] + 0.000001)
            TERT_metrics['pre'] = (TERT_metrics['tp']) / (TERT_metrics['tp'] + TERT_metrics['fp'] + 0.000001)
            TERT_metrics['recall'] = TERT_metrics['sen']
            TERT_metrics['f1'] = (2 * TERT_metrics['pre'] * TERT_metrics['recall']) / (
                    TERT_metrics['pre'] + TERT_metrics['recall'] + 0.000001)
            TERT_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_TERT),
                                                        y_score=np.array(predicted_all_TERT))
            list_TERT = (
            Acc_TERT, TERT_metrics['sen'], TERT_metrics['spec'], TERT_metrics['pre'], TERT_metrics['recall']
            , TERT_metrics['f1'], TERT_metrics['AUC'])
            ##########  PDGFRA
            Acc_PDGFRA = (PDGFRA_metrics['tp'] + PDGFRA_metrics['tn']) / (
                    PDGFRA_metrics['tp'] + PDGFRA_metrics['tn'] + PDGFRA_metrics['fp'] + PDGFRA_metrics[
                'fn'] + 0.000001)
            PDGFRA_metrics['sen'] = (PDGFRA_metrics['tp']) / (
                        PDGFRA_metrics['tp'] + PDGFRA_metrics['fn'] + 0.000001)  # recall
            PDGFRA_metrics['spec'] = (PDGFRA_metrics['tn']) / (PDGFRA_metrics['tn'] + PDGFRA_metrics['fp'] + 0.000001)
            PDGFRA_metrics['pre'] = (PDGFRA_metrics['tp']) / (PDGFRA_metrics['tp'] + PDGFRA_metrics['fp'] + 0.000001)
            PDGFRA_metrics['recall'] = PDGFRA_metrics['sen']
            PDGFRA_metrics['f1'] = (2 * PDGFRA_metrics['pre'] * PDGFRA_metrics['recall']) / (
                    PDGFRA_metrics['pre'] + PDGFRA_metrics['recall'] + 0.000001)
            PDGFRA_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_PDGFRA),
                                                          y_score=np.array(predicted_all_PDGFRA))
            list_PDGFRA = (
            Acc_PDGFRA, PDGFRA_metrics['sen'], PDGFRA_metrics['spec'], PDGFRA_metrics['pre'], PDGFRA_metrics['recall']
            , PDGFRA_metrics['f1'], PDGFRA_metrics['AUC'])
        if opt['TrainingSet'] == 'All' or opt['TrainingSet'] == 'Tiantan':
            ##########  OLIG2
            Acc_OLIG2 = (OLIG2_metrics['tp'] + OLIG2_metrics['tn']) / (
                    OLIG2_metrics['tp'] + OLIG2_metrics['tn'] + OLIG2_metrics['fp'] + OLIG2_metrics['fn'] + 0.000001)
            OLIG2_metrics['sen'] = (OLIG2_metrics['tp']) / (
                        OLIG2_metrics['tp'] + OLIG2_metrics['fn'] + 0.000001)  # recall
            OLIG2_metrics['spec'] = (OLIG2_metrics['tn']) / (OLIG2_metrics['tn'] + OLIG2_metrics['fp'] + 0.000001)
            OLIG2_metrics['pre'] = (OLIG2_metrics['tp']) / (OLIG2_metrics['tp'] + OLIG2_metrics['fp'] + 0.000001)
            OLIG2_metrics['recall'] = OLIG2_metrics['sen']
            OLIG2_metrics['f1'] = (2 * OLIG2_metrics['pre'] * OLIG2_metrics['recall']) / (
                    OLIG2_metrics['pre'] + OLIG2_metrics['recall'] + 0.000001)
            OLIG2_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_OLIG2),
                                                         y_score=np.array(predicted_all_OLIG2))
            list_OLIG2 = (
            Acc_OLIG2, OLIG2_metrics['sen'], OLIG2_metrics['spec'], OLIG2_metrics['pre'], OLIG2_metrics['recall']
            , OLIG2_metrics['f1'], OLIG2_metrics['AUC'])


    if not flag_IvYGAP and not flag_cam:
        if opt['TrainingSet'] == 'All':
            return list_MGMT, list_EGFR, list_ATRX,list_PTEN, list_TERT, list_P53,list_710, list_PDGFRA, list_OLIG2
        if opt['TrainingSet'] == 'TCGA':
            return list_MGMT, list_EGFR, list_ATRX, list_PTEN, list_TERT, list_P53, list_710, list_PDGFRA
        if opt['TrainingSet'] == 'Tiantan':
            return list_MGMT, list_EGFR, list_ATRX,list_PTEN, list_P53, list_OLIG2
    elif flag_IvYGAP:
        return list_MGMT, list_EGFR, list_PTEN
    elif flag_cam:
        return list_MGMT


def test_stage1(opt,Mine_model_init,Mine_model_His,Mine_model_Cls, dataloader, gpuID,epoch,external=False):
    Mine_model_init.eval()
    Mine_model_His.eval()
    Mine_model_Cls.eval()

    if 1:
        count_His = 0
        count_Grade = 0
        correct_His = 0
        correct_Grade = 0
        A_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                       'AUC': 0}
        O_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                       'AUC': 0}
        GBM_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                        'AUC': 0}
        all_metrics_His = {'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        label_all_His = []
        predicted_all_His = []
        G2_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                      'AUC': 0}
        G3_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                      'AUC': 0}
        G4_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                      'AUC': 0}
        all_metrics_Grade = {'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        label_all_Grade = []
        predicted_all_Grade = []


    test_bar = tqdm(dataloader)

    for packs in test_bar:
        img = packs[0]
        label = packs[1]
        file_name = packs[2]
        img = img.cuda(gpuID[0])
        label = label.cuda(gpuID[0])
        label_his = label[:, 0]
        label_grade = label[:, 1]

        saliency_map_His, saliency_map_Grade = saliency_map_read(opt, file_name,epoch)
        saliency_map_His = torch.from_numpy(np.array(saliency_map_His)).float().cuda(gpuID[0])
        saliency_map_Grade = torch.from_numpy(np.array(saliency_map_Grade)).float().cuda(gpuID[0])
        # saliency_map_His, saliency_map_Grade = imp_gene(opt, img)

        init_feature = Mine_model_init(img)  # (BS,2500,1024)
        hidden_states_his, hidden_states_grade, encoded_His, encoded_Grade = Mine_model_His(init_feature,saliency_map_His, saliency_map_Grade)
        results_dict, saliency_A, saliency_O, saliency_GBM, saliency_G2, saliency_G3, saliency_G4 = Mine_model_Cls(
            encoded_His, encoded_Grade)

        pred_His_ori = results_dict['logits_His']
        pred_Grade_ori = results_dict['logits_Grade']
        _, pred_His = torch.max(pred_His_ori.data, 1)
        pred_His = pred_His.tolist()  # [BS] A  O GBM //0 1 2
        gt_His = label_his.tolist()  # [BS] A  O GBM//0 1 2
        _, pred_Grade = torch.max(pred_Grade_ori.data, 1)
        pred_Grade = pred_Grade.tolist()  # [BS] A  O GBM //0 1 2
        gt_Grade = label_grade.tolist()  # [BS] A  O GBM//0 1 2


        ##################   His
        if gt_His[0] != 3:
            label_all_His.append(gt_His[0])
            predicted_all_His.append(pred_His_ori.detach().cpu().numpy()[0])
            count_His += 1
            if gt_His[0] == pred_His[0]:
                correct_His += 1
            if gt_His[0] == 0:
                if pred_His[0] == 0:
                    A_metrics['tp'] += 1
                else:
                    A_metrics['fn'] += 1
            else:
                if not pred_His[0] == 0:
                    A_metrics['tn'] += 1
                else:
                    A_metrics['fp'] += 1
                # O
            if gt_His[0] == 1:
                if pred_His[0] == 1:
                    O_metrics['tp'] += 1
                else:
                    O_metrics['fn'] += 1
            else:
                if not pred_His[0] == 1:
                    O_metrics['tn'] += 1
                else:
                    O_metrics['fp'] += 1
                # GBM
            if gt_His[0] == 2:
                if pred_His[0] == 2:
                    GBM_metrics['tp'] += 1
                else:
                    GBM_metrics['fn'] += 1
            else:
                if not pred_His[0] == 2:
                    GBM_metrics['tn'] += 1
                else:
                    GBM_metrics['fp'] += 1
        ##################   Grade
        if 1:
            # G2
            if gt_Grade[0] == 0:
                if pred_Grade[0] == 0:
                    G2_metrics['tp'] += 1
                else:
                    G2_metrics['fn'] += 1
            else:
                if not pred_Grade[0] == 0:
                    G2_metrics['tn'] += 1
                else:
                    G2_metrics['fp'] += 1
            # G3
            if gt_Grade[0] == 1:
                if pred_Grade[0] == 1:
                    G3_metrics['tp'] += 1
                else:
                    G3_metrics['fn'] += 1
            else:
                if not pred_Grade[0] == 1:
                    G3_metrics['tn'] += 1
                else:
                    G3_metrics['fp'] += 1
            # G4
            if gt_Grade[0] == 2:
                if pred_Grade[0] == 2:
                    G4_metrics['tp'] += 1
                else:
                    G4_metrics['fn'] += 1
            else:
                if not pred_Grade[0] == 2:
                    G4_metrics['tn'] += 1
                else:
                    G4_metrics['fp'] += 1
            label_all_Grade.append(gt_Grade[0])
            predicted_all_Grade.append(pred_Grade_ori.detach().cpu().numpy()[0])
            count_Grade += 1
            if gt_Grade[0] == pred_Grade[0]:
                correct_Grade += 1

    ################################################ His
    Acc_His = correct_His / count_His

    #  Sensitivity
    A_metrics['sen'] = (A_metrics['tp']) / (A_metrics['tp'] + A_metrics['fn'] + 0.000001)
    O_metrics['sen'] = (O_metrics['tp']) / (O_metrics['tp'] + O_metrics['fn'] + 0.000001)
    GBM_metrics['sen'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fn'] + 0.000001)
    all_metrics_His['sen'] = A_metrics['sen'] * label_all_His.count(0) / len(label_all_His) + \
                                O_metrics['sen'] * label_all_His.count(1) / len(label_all_His) + \
                                GBM_metrics['sen'] * label_all_His.count(2) / len(label_all_His)

    #  Spec
    A_metrics['spec'] = (A_metrics['tn']) / (A_metrics['tn'] + A_metrics['fp'] + 0.000001)
    O_metrics['spec'] = (O_metrics['tn']) / (O_metrics['tn'] + O_metrics['fp'] + 0.000001)
    GBM_metrics['spec'] = (GBM_metrics['tn']) / (GBM_metrics['tn'] + GBM_metrics['fp'] + 0.000001)
    all_metrics_His['spec'] = A_metrics['spec'] * label_all_His.count(0) / len(label_all_His) + \
                             O_metrics['spec'] * label_all_His.count(1) / len(label_all_His) + \
                             GBM_metrics['spec'] * label_all_His.count(2) / len(label_all_His)
    #  Precision
    A_metrics['pre'] = (A_metrics['tp']) / (A_metrics['tp'] + A_metrics['fp'] + 0.000001)
    O_metrics['pre'] = (O_metrics['tp']) / (O_metrics['tp'] + O_metrics['fp'] + 0.000001)
    GBM_metrics['pre'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fp'] + 0.000001)
    all_metrics_His['pre'] = A_metrics['pre'] * label_all_His.count(0) / len(label_all_His) + \
                              O_metrics['pre'] * label_all_His.count(1) / len(label_all_His) + \
                              GBM_metrics['pre'] * label_all_His.count(2) / len(label_all_His)
    #  Recall
    A_metrics['recall'] = (A_metrics['tp']) / (A_metrics['tp'] + A_metrics['fn'] + 0.000001)
    O_metrics['recall'] = (O_metrics['tp']) / (O_metrics['tp'] + O_metrics['fn'] + 0.000001)
    GBM_metrics['recall'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fn'] + 0.000001)
    all_metrics_His['recall'] = A_metrics['recall'] * label_all_His.count(0) / len(label_all_His) + \
                             O_metrics['recall'] * label_all_His.count(1) / len(label_all_His) + \
                             GBM_metrics['recall'] * label_all_His.count(2) / len(label_all_His)
    #  F1
    A_metrics['f1'] = (2 * A_metrics['pre'] * A_metrics['recall']) / (
            A_metrics['pre'] + A_metrics['recall'] + 0.000001)
    O_metrics['f1'] = (2 * O_metrics['pre'] * O_metrics['recall']) / (
            O_metrics['pre'] + O_metrics['recall'] + 0.000001)
    GBM_metrics['f1'] = (2 * GBM_metrics['pre'] * GBM_metrics['recall']) / (
                GBM_metrics['pre'] + GBM_metrics['recall'] + 0.000001)
    all_metrics_His['f1'] = A_metrics['f1'] * label_all_His.count(0) / len(label_all_His) + \
                             O_metrics['f1'] * label_all_His.count(1) / len(label_all_His) + \
                             GBM_metrics['f1'] * label_all_His.count(2) / len(label_all_His)

    # AUC
    if not external:
        out_cls_all_softmax_His = F.softmax(torch.from_numpy(np.array(predicted_all_His)), dim=1).numpy()
        label_all_np = np.array(label_all_His)
        label_all_onehot = make_one_hot(label_all_np)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_His[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(3):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= 3
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), out_cls_all_softmax_His.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        all_metrics_His['AUC'] = roc_auc["micro"]
        A_metrics['AUC'] = roc_auc[0]
        O_metrics['AUC'] = roc_auc[1]
        GBM_metrics['AUC'] = roc_auc[2]
    else:

        out_cls_all_softmax_His = F.softmax(torch.from_numpy(np.array(predicted_all_His)), dim=1).numpy()
        label_all_np = np.array(label_all_His)
        label_all_onehot = make_one_hot(label_all_np,N=3)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_His[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(3):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= 3
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), out_cls_all_softmax_His.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        all_metrics_His['AUC'] = roc_auc["micro"]
        A_metrics['AUC'] = 0
        O_metrics['AUC'] = 0
        GBM_metrics['AUC'] = roc_auc[2]

    list_His = (Acc_His, all_metrics_His['sen'], all_metrics_His['spec'], all_metrics_His['pre'], all_metrics_His['recall']
            , all_metrics_His['f1'], all_metrics_His['AUC'])
    list_A = (0, A_metrics['sen'], A_metrics['spec'], A_metrics['pre'], A_metrics['recall']
                , A_metrics['f1'], A_metrics['AUC'])
    list_O = (0, O_metrics['sen'], O_metrics['spec'], O_metrics['pre'], O_metrics['recall']
                , O_metrics['f1'], O_metrics['AUC'])
    list_GBM = (0, GBM_metrics['sen'], GBM_metrics['spec'], GBM_metrics['pre'], GBM_metrics['recall']
                 , GBM_metrics['f1'], GBM_metrics['AUC'])

    ################################################ Grade
    Acc_Grade = correct_Grade / count_Grade

    #  Sensitivity
    G2_metrics['sen'] = (G2_metrics['tp']) / (G2_metrics['tp'] + G2_metrics['fn'] + 0.000001)
    G3_metrics['sen'] = (G3_metrics['tp']) / (G3_metrics['tp'] + G3_metrics['fn'] + 0.000001)
    G4_metrics['sen'] = (G4_metrics['tp']) / (G4_metrics['tp'] + G4_metrics['fn'] + 0.000001)
    all_metrics_Grade['sen'] = G2_metrics['sen'] * label_all_Grade.count(0) / len(label_all_Grade) + \
                               G3_metrics['sen'] * label_all_Grade.count(1) / len(label_all_Grade) + \
                               G4_metrics['sen'] * label_all_Grade.count(2) / len(label_all_Grade)

    #  Spec
    G2_metrics['spec'] = (G2_metrics['tn']) / (G2_metrics['tn'] + G2_metrics['fp'] + 0.000001)
    G3_metrics['spec'] = (G3_metrics['tn']) / (G3_metrics['tn'] + G3_metrics['fp'] + 0.000001)
    G4_metrics['spec'] = (G4_metrics['tn']) / (G4_metrics['tn'] + G4_metrics['fp'] + 0.000001)
    all_metrics_Grade['spec'] = G2_metrics['spec'] * label_all_Grade.count(0) / len(label_all_Grade) + \
                                G3_metrics['spec'] * label_all_Grade.count(1) / len(label_all_Grade) + \
                                G4_metrics['spec'] * label_all_Grade.count(2) / len(label_all_Grade)
    #  Precision
    G2_metrics['pre'] = (G2_metrics['tp']) / (G2_metrics['tp'] + G2_metrics['fp'] + 0.000001)
    G3_metrics['pre'] = (G3_metrics['tp']) / (G3_metrics['tp'] + G3_metrics['fp'] + 0.000001)
    G4_metrics['pre'] = (G4_metrics['tp']) / (G4_metrics['tp'] + G4_metrics['fp'] + 0.000001)
    all_metrics_Grade['pre'] = G2_metrics['pre'] * label_all_Grade.count(0) / len(label_all_Grade) + \
                               G3_metrics['pre'] * label_all_Grade.count(1) / len(label_all_Grade) + \
                               G4_metrics['pre'] * label_all_Grade.count(2) / len(label_all_Grade)
    #  Recall
    G2_metrics['recall'] = (G2_metrics['tp']) / (G2_metrics['tp'] + G2_metrics['fn'] + 0.000001)
    G3_metrics['recall'] = (G3_metrics['tp']) / (G3_metrics['tp'] + G3_metrics['fn'] + 0.000001)
    G4_metrics['recall'] = (G4_metrics['tp']) / (G4_metrics['tp'] + G4_metrics['fn'] + 0.000001)
    all_metrics_Grade['recall'] = G2_metrics['recall'] * label_all_Grade.count(0) / len(label_all_Grade) + \
                                  G3_metrics['recall'] * label_all_Grade.count(1) / len(label_all_Grade) + \
                                  G4_metrics['recall'] * label_all_Grade.count(2) / len(label_all_Grade)
    #  F1
    G2_metrics['f1'] = (2 * G2_metrics['pre'] * G2_metrics['recall']) / (
            G2_metrics['pre'] + G2_metrics['recall'] + 0.000001)
    G3_metrics['f1'] = (2 * G3_metrics['pre'] * G3_metrics['recall']) / (
            G3_metrics['pre'] + G3_metrics['recall'] + 0.000001)
    G4_metrics['f1'] = (2 * G4_metrics['pre'] * G4_metrics['recall']) / (
            G4_metrics['pre'] + G4_metrics['recall'] + 0.000001)
    all_metrics_Grade['f1'] = G2_metrics['f1'] * label_all_Grade.count(0) / len(label_all_Grade) + \
                              G3_metrics['f1'] * label_all_Grade.count(1) / len(label_all_Grade) + \
                              G4_metrics['f1'] * label_all_Grade.count(2) / len(label_all_Grade)

    # AUC
    if not external:
        out_cls_all_softmax_Grade = F.softmax(torch.from_numpy(np.array(predicted_all_Grade)), dim=1).numpy()
        label_all_np = np.array(label_all_Grade)
        label_all_onehot = make_one_hot(label_all_np)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_Grade[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(3):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= 3
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), out_cls_all_softmax_Grade.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        all_metrics_Grade['AUC'] = roc_auc["micro"]
        G2_metrics['AUC'] = roc_auc[0]
        G3_metrics['AUC'] = roc_auc[1]
        G4_metrics['AUC'] = roc_auc[2]
    else:

        out_cls_all_softmax_Grade = F.softmax(torch.from_numpy(np.array(predicted_all_Grade)), dim=1).numpy()
        label_all_np = np.array(label_all_Grade)
        label_all_onehot = make_one_hot(label_all_np, N=3)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_Grade[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(3):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= 3
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), out_cls_all_softmax_Grade.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        all_metrics_Grade['AUC'] = roc_auc["micro"]
        G2_metrics['AUC'] = 0
        G3_metrics['AUC'] = 0
        G4_metrics['AUC'] = roc_auc[2]

    list_Grade = (
        Acc_Grade, all_metrics_Grade['sen'], all_metrics_Grade['spec'], all_metrics_Grade['pre'],
        all_metrics_Grade['recall']
        , all_metrics_Grade['f1'], all_metrics_Grade['AUC'])
    list_G2 = (0, G2_metrics['sen'], G2_metrics['spec'], G2_metrics['pre'], G2_metrics['recall']
               , G2_metrics['f1'], G2_metrics['AUC'])
    list_G3 = (0, G3_metrics['sen'], G3_metrics['spec'], G3_metrics['pre'], G3_metrics['recall']
               , G3_metrics['f1'], G3_metrics['AUC'])
    list_G4 = (0, G4_metrics['sen'], G4_metrics['spec'], G4_metrics['pre'], G4_metrics['recall']
               , G4_metrics['f1'], G4_metrics['AUC'])

    return list_His, list_A, list_O, list_GBM, list_Grade, list_G2,\
    list_G3, list_G4


def test_allmarker_CNN(opt, Mine_CNN_cls, dataloader, gpuID):
    Mine_CNN_cls.eval()
    gpuID = opt['gpus']

    if 1:
        MGMT_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,'AUC': 0}

        label_all_MGMT = []
        predicted_all_MGMT = []

    test_bar = tqdm(dataloader)

    for packs in test_bar:
        img = packs[0]  ##(BS,N,1024)
        label = packs[1]
        file_name = packs[2]
        img = img.cuda(gpuID[0])
        label = label.cuda(gpuID[0])
        label_MGMT=label[:, 0]

        ### ### forward WHO 2007
        results_dict = Mine_CNN_cls(img)  # (BS,2500,1024)

        pred_MGMT_ori = results_dict['logits']


        _, pred_MGMT0 = torch.max(pred_MGMT_ori.data, 1)
        pred_MGMT = pred_MGMT0.tolist()
        gt_MGMT = label_MGMT.tolist()


        ############################ WSI calculate tntp

        ##############MGMT
        if gt_MGMT[0] != 2:
            label_all_MGMT.append(gt_MGMT[0])
            predicted_all_MGMT.append(pred_MGMT_ori.detach().cpu().numpy()[0][1])
            if gt_MGMT[0] == 0 and pred_MGMT[0] == 0:
                MGMT_metrics['tn'] += 1
            if gt_MGMT[0] == 0 and pred_MGMT[0] == 1:
                MGMT_metrics['fp'] += 1
            if gt_MGMT[0] == 1 and pred_MGMT[0] == 0:
                MGMT_metrics['fn'] += 1
            if gt_MGMT[0] == 1 and pred_MGMT[0] == 1:
                MGMT_metrics['tp'] += 1

    ##########  MGMT
    Acc_MGMT = (MGMT_metrics['tp'] + MGMT_metrics['tn']) / (
            MGMT_metrics['tp'] + MGMT_metrics['tn'] + MGMT_metrics['fp'] + MGMT_metrics['fn'] + 0.000001)
    MGMT_metrics['sen'] = (MGMT_metrics['tp']) / (MGMT_metrics['tp'] + MGMT_metrics['fn'] + 0.000001)  # recall
    MGMT_metrics['spec'] = (MGMT_metrics['tn']) / (MGMT_metrics['tn'] + MGMT_metrics['fp'] + 0.000001)
    MGMT_metrics['pre'] = (MGMT_metrics['tp']) / (MGMT_metrics['tp'] + MGMT_metrics['fp'] + 0.000001)
    MGMT_metrics['recall'] = MGMT_metrics['sen']
    MGMT_metrics['f1'] = (2 * MGMT_metrics['pre'] * MGMT_metrics['recall']) / (
            MGMT_metrics['pre'] + MGMT_metrics['recall'] + 0.000001)
    MGMT_metrics['AUC'] = metrics.roc_auc_score(y_true=np.array(label_all_MGMT),
                                                y_score=np.array(predicted_all_MGMT))
    list_MGMT = (Acc_MGMT, MGMT_metrics['sen'], MGMT_metrics['spec'], MGMT_metrics['pre'], MGMT_metrics['recall']
                 , MGMT_metrics['f1'], MGMT_metrics['AUC'])


    return list_MGMT


def test_stage1_pre(opt,Mine_model_init,Mine_model_His,Mine_model_Cls, dataloader, gpuID,external=False,trainLoader=0):
    Mine_model_init.eval()
    Mine_model_His.eval()
    Mine_model_Cls.eval()
    if not os.path.exists('./saliency/init/'+opt['name']+'/Grade/'):
        os.makedirs('./saliency/init/'+opt['name']+'/Grade/')
    if not os.path.exists('./saliency/init/'+opt['name']+'/His/'):
        os.makedirs('./saliency/init/'+opt['name']+'/His/')
    if 1:
        count_His = 0
        count_Grade = 0
        correct_His = 0
        correct_Grade = 0
        A_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                       'AUC': 0}
        O_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                       'AUC': 0}
        GBM_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                        'AUC': 0}
        all_metrics_His = {'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        label_all_His = []
        predicted_all_His = []
        G2_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                      'AUC': 0}
        G3_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                      'AUC': 0}
        G4_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                      'AUC': 0}
        all_metrics_Grade = {'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        label_all_Grade = []
        predicted_all_Grade = []

    if not trainLoader ==0:
        train_bar = tqdm(trainLoader)
        for packs in train_bar:
            img = packs[0]
            label = packs[1]
            file_name = packs[2]
            img = img.cuda(gpuID[0])
            imp_his, imp_grade = imp_gene(opt, img)
            init_feature = Mine_model_init(img)  # (BS,2500,1024)
            hidden_states_his, hidden_states_grade, encoded_His, encoded_Grade = Mine_model_His(init_feature, imp_his,
                                                                                                imp_grade)
            results_dict, saliency_A, saliency_O, saliency_GBM, saliency_G2, saliency_G3, saliency_G4 = Mine_model_Cls(
                encoded_His, encoded_Grade)

            pred_His_ori = results_dict['logits_His']
            pred_Grade_ori = results_dict['logits_Grade']
            _, pred_His = torch.max(pred_His_ori.data, 1)
            pred_His = pred_His.tolist()  # [BS] A  O GBM //0 1 2
            _, pred_Grade = torch.max(pred_Grade_ori.data, 1)
            pred_Grade = pred_Grade.tolist()  # [BS] A  O GBM //0 1 2

            ################################ VISUALIZATION INIT GENERATION################################
            saliency_final_His, saliency_final_Grade = saliency_comparison(saliency_A, saliency_O, saliency_GBM,
                                                                           saliency_G2, saliency_G3, saliency_G4, pred_His,
                                                                           pred_Grade)
            # print(file_name[0])
            np.save('./saliency/init/'+opt['name']+'/Grade/' + file_name[0] + '.npy', saliency_final_Grade)
            np.save('./saliency/init/'+opt['name']+'/His/' + file_name[0] + '.npy', saliency_final_His)
            ################################ VISUALIZATION INIT GENERATION################################

    test_bar = tqdm(dataloader)

    for packs in test_bar:
        img = packs[0]
        label = packs[1]
        file_name = packs[2]
        img = img.cuda(gpuID[0])
        label = label.cuda(gpuID[0])
        label_his = label[:, 0]
        label_grade = label[:, 1]

        imp_his, imp_grade = imp_gene(opt,img)
        init_feature = Mine_model_init(img)  # (BS,2500,1024)
        hidden_states_his, hidden_states_grade, encoded_His, encoded_Grade = Mine_model_His(init_feature,imp_his, imp_grade)
        results_dict, saliency_A, saliency_O, saliency_GBM, saliency_G2, saliency_G3, saliency_G4 = Mine_model_Cls(
            encoded_His, encoded_Grade)

        pred_His_ori = results_dict['logits_His']
        pred_Grade_ori = results_dict['logits_Grade']
        _, pred_His = torch.max(pred_His_ori.data, 1)
        pred_His = pred_His.tolist()  # [BS] A  O GBM //0 1 2
        gt_His = label_his.tolist()  # [BS] A  O GBM//0 1 2
        _, pred_Grade = torch.max(pred_Grade_ori.data, 1)
        pred_Grade = pred_Grade.tolist()  # [BS] A  O GBM //0 1 2
        gt_Grade = label_grade.tolist()  # [BS] A  O GBM//0 1 2

        ################################ VISUALIZATION INIT GENERATION################################
        saliency_final_His, saliency_final_Grade = saliency_comparison(saliency_A, saliency_O, saliency_GBM,
                                                                       saliency_G2, saliency_G3, saliency_G4, pred_His,
                                                                       pred_Grade)
        # print(file_name[0])
        np.save('./saliency/init/' + opt['name'] + '/Grade/' + file_name[0] + '.npy', saliency_final_Grade)
        np.save('./saliency/init/' + opt['name'] + '/His/' + file_name[0] + '.npy', saliency_final_His)
        ################################ VISUALIZATION INIT GENERATION################################

        ##################   His
        if gt_His[0] != 3:
            label_all_His.append(gt_His[0])
            predicted_all_His.append(pred_His_ori.detach().cpu().numpy()[0])
            count_His += 1
            if gt_His[0] == pred_His[0]:
                correct_His += 1
            if gt_His[0] == 0:
                if pred_His[0] == 0:
                    A_metrics['tp'] += 1
                else:
                    A_metrics['fn'] += 1
            else:
                if not pred_His[0] == 0:
                    A_metrics['tn'] += 1
                else:
                    A_metrics['fp'] += 1
                # O
            if gt_His[0] == 1:
                if pred_His[0] == 1:
                    O_metrics['tp'] += 1
                else:
                    O_metrics['fn'] += 1
            else:
                if not pred_His[0] == 1:
                    O_metrics['tn'] += 1
                else:
                    O_metrics['fp'] += 1
                # GBM
            if gt_His[0] == 2:
                if pred_His[0] == 2:
                    GBM_metrics['tp'] += 1
                else:
                    GBM_metrics['fn'] += 1
            else:
                if not pred_His[0] == 2:
                    GBM_metrics['tn'] += 1
                else:
                    GBM_metrics['fp'] += 1
        ##################   Grade
        if 1:
            # G2
            if gt_Grade[0] == 0:
                if pred_Grade[0] == 0:
                    G2_metrics['tp'] += 1
                else:
                    G2_metrics['fn'] += 1
            else:
                if not pred_Grade[0] == 0:
                    G2_metrics['tn'] += 1
                else:
                    G2_metrics['fp'] += 1
            # G3
            if gt_Grade[0] == 1:
                if pred_Grade[0] == 1:
                    G3_metrics['tp'] += 1
                else:
                    G3_metrics['fn'] += 1
            else:
                if not pred_Grade[0] == 1:
                    G3_metrics['tn'] += 1
                else:
                    G3_metrics['fp'] += 1
            # G4
            if gt_Grade[0] == 2:
                if pred_Grade[0] == 2:
                    G4_metrics['tp'] += 1
                else:
                    G4_metrics['fn'] += 1
            else:
                if not pred_Grade[0] == 2:
                    G4_metrics['tn'] += 1
                else:
                    G4_metrics['fp'] += 1
            label_all_Grade.append(gt_Grade[0])
            predicted_all_Grade.append(pred_Grade_ori.detach().cpu().numpy()[0])
            count_Grade += 1
            if gt_Grade[0] == pred_Grade[0]:
                correct_Grade += 1

    ################################################ His
    Acc_His = correct_His / count_His

    #  Sensitivity
    A_metrics['sen'] = (A_metrics['tp']) / (A_metrics['tp'] + A_metrics['fn'] + 0.000001)
    O_metrics['sen'] = (O_metrics['tp']) / (O_metrics['tp'] + O_metrics['fn'] + 0.000001)
    GBM_metrics['sen'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fn'] + 0.000001)
    all_metrics_His['sen'] = A_metrics['sen'] * label_all_His.count(0) / len(label_all_His) + \
                                O_metrics['sen'] * label_all_His.count(1) / len(label_all_His) + \
                                GBM_metrics['sen'] * label_all_His.count(2) / len(label_all_His)

    #  Spec
    A_metrics['spec'] = (A_metrics['tn']) / (A_metrics['tn'] + A_metrics['fp'] + 0.000001)
    O_metrics['spec'] = (O_metrics['tn']) / (O_metrics['tn'] + O_metrics['fp'] + 0.000001)
    GBM_metrics['spec'] = (GBM_metrics['tn']) / (GBM_metrics['tn'] + GBM_metrics['fp'] + 0.000001)
    all_metrics_His['spec'] = A_metrics['spec'] * label_all_His.count(0) / len(label_all_His) + \
                             O_metrics['spec'] * label_all_His.count(1) / len(label_all_His) + \
                             GBM_metrics['spec'] * label_all_His.count(2) / len(label_all_His)
    #  Precision
    A_metrics['pre'] = (A_metrics['tp']) / (A_metrics['tp'] + A_metrics['fp'] + 0.000001)
    O_metrics['pre'] = (O_metrics['tp']) / (O_metrics['tp'] + O_metrics['fp'] + 0.000001)
    GBM_metrics['pre'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fp'] + 0.000001)
    all_metrics_His['pre'] = A_metrics['pre'] * label_all_His.count(0) / len(label_all_His) + \
                              O_metrics['pre'] * label_all_His.count(1) / len(label_all_His) + \
                              GBM_metrics['pre'] * label_all_His.count(2) / len(label_all_His)
    #  Recall
    A_metrics['recall'] = (A_metrics['tp']) / (A_metrics['tp'] + A_metrics['fn'] + 0.000001)
    O_metrics['recall'] = (O_metrics['tp']) / (O_metrics['tp'] + O_metrics['fn'] + 0.000001)
    GBM_metrics['recall'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fn'] + 0.000001)
    all_metrics_His['recall'] = A_metrics['recall'] * label_all_His.count(0) / len(label_all_His) + \
                             O_metrics['recall'] * label_all_His.count(1) / len(label_all_His) + \
                             GBM_metrics['recall'] * label_all_His.count(2) / len(label_all_His)
    #  F1
    A_metrics['f1'] = (2 * A_metrics['pre'] * A_metrics['recall']) / (
            A_metrics['pre'] + A_metrics['recall'] + 0.000001)
    O_metrics['f1'] = (2 * O_metrics['pre'] * O_metrics['recall']) / (
            O_metrics['pre'] + O_metrics['recall'] + 0.000001)
    GBM_metrics['f1'] = (2 * GBM_metrics['pre'] * GBM_metrics['recall']) / (
                GBM_metrics['pre'] + GBM_metrics['recall'] + 0.000001)
    all_metrics_His['f1'] = A_metrics['f1'] * label_all_His.count(0) / len(label_all_His) + \
                             O_metrics['f1'] * label_all_His.count(1) / len(label_all_His) + \
                             GBM_metrics['f1'] * label_all_His.count(2) / len(label_all_His)

    # AUC
    if not external:
        out_cls_all_softmax_His = F.softmax(torch.from_numpy(np.array(predicted_all_His)), dim=1).numpy()
        label_all_np = np.array(label_all_His)
        label_all_onehot = make_one_hot(label_all_np)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_His[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(3):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= 3
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), out_cls_all_softmax_His.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        all_metrics_His['AUC'] = roc_auc["micro"]
        A_metrics['AUC'] = roc_auc[0]
        O_metrics['AUC'] = roc_auc[1]
        GBM_metrics['AUC'] = roc_auc[2]
    else:
        out_cls_all_softmax_His = F.softmax(torch.from_numpy(np.array(predicted_all_His)), dim=1).numpy()
        label_all_np = np.array(label_all_His)
        label_all_onehot = make_one_hot(label_all_np,N=3)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_His[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(3):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= 3
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), out_cls_all_softmax_His.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        all_metrics_His['AUC'] = roc_auc["micro"]
        A_metrics['AUC'] = 0
        O_metrics['AUC'] = 0
        GBM_metrics['AUC'] = roc_auc[2]

    list_His = (Acc_His, all_metrics_His['sen'], all_metrics_His['spec'], all_metrics_His['pre'], all_metrics_His['recall']
            , all_metrics_His['f1'], all_metrics_His['AUC'])
    list_A = (0, A_metrics['sen'], A_metrics['spec'], A_metrics['pre'], A_metrics['recall']
                , A_metrics['f1'], A_metrics['AUC'])
    list_O = (0, O_metrics['sen'], O_metrics['spec'], O_metrics['pre'], O_metrics['recall']
                , O_metrics['f1'], O_metrics['AUC'])
    list_GBM = (0, GBM_metrics['sen'], GBM_metrics['spec'], GBM_metrics['pre'], GBM_metrics['recall']
                 , GBM_metrics['f1'], GBM_metrics['AUC'])

    ################################################ Grade
    Acc_Grade = correct_Grade / count_Grade

    #  Sensitivity
    G2_metrics['sen'] = (G2_metrics['tp']) / (G2_metrics['tp'] + G2_metrics['fn'] + 0.000001)
    G3_metrics['sen'] = (G3_metrics['tp']) / (G3_metrics['tp'] + G3_metrics['fn'] + 0.000001)
    G4_metrics['sen'] = (G4_metrics['tp']) / (G4_metrics['tp'] + G4_metrics['fn'] + 0.000001)
    all_metrics_Grade['sen'] = G2_metrics['sen'] * label_all_Grade.count(0) / len(label_all_Grade) + \
                               G3_metrics['sen'] * label_all_Grade.count(1) / len(label_all_Grade) + \
                               G4_metrics['sen'] * label_all_Grade.count(2) / len(label_all_Grade)

    #  Spec
    G2_metrics['spec'] = (G2_metrics['tn']) / (G2_metrics['tn'] + G2_metrics['fp'] + 0.000001)
    G3_metrics['spec'] = (G3_metrics['tn']) / (G3_metrics['tn'] + G3_metrics['fp'] + 0.000001)
    G4_metrics['spec'] = (G4_metrics['tn']) / (G4_metrics['tn'] + G4_metrics['fp'] + 0.000001)
    all_metrics_Grade['spec'] = G2_metrics['spec'] * label_all_Grade.count(0) / len(label_all_Grade) + \
                                G3_metrics['spec'] * label_all_Grade.count(1) / len(label_all_Grade) + \
                                G4_metrics['spec'] * label_all_Grade.count(2) / len(label_all_Grade)
    #  Precision
    G2_metrics['pre'] = (G2_metrics['tp']) / (G2_metrics['tp'] + G2_metrics['fp'] + 0.000001)
    G3_metrics['pre'] = (G3_metrics['tp']) / (G3_metrics['tp'] + G3_metrics['fp'] + 0.000001)
    G4_metrics['pre'] = (G4_metrics['tp']) / (G4_metrics['tp'] + G4_metrics['fp'] + 0.000001)
    all_metrics_Grade['pre'] = G2_metrics['pre'] * label_all_Grade.count(0) / len(label_all_Grade) + \
                               G3_metrics['pre'] * label_all_Grade.count(1) / len(label_all_Grade) + \
                               G4_metrics['pre'] * label_all_Grade.count(2) / len(label_all_Grade)
    #  Recall
    G2_metrics['recall'] = (G2_metrics['tp']) / (G2_metrics['tp'] + G2_metrics['fn'] + 0.000001)
    G3_metrics['recall'] = (G3_metrics['tp']) / (G3_metrics['tp'] + G3_metrics['fn'] + 0.000001)
    G4_metrics['recall'] = (G4_metrics['tp']) / (G4_metrics['tp'] + G4_metrics['fn'] + 0.000001)
    all_metrics_Grade['recall'] = G2_metrics['recall'] * label_all_Grade.count(0) / len(label_all_Grade) + \
                                  G3_metrics['recall'] * label_all_Grade.count(1) / len(label_all_Grade) + \
                                  G4_metrics['recall'] * label_all_Grade.count(2) / len(label_all_Grade)
    #  F1
    G2_metrics['f1'] = (2 * G2_metrics['pre'] * G2_metrics['recall']) / (
            G2_metrics['pre'] + G2_metrics['recall'] + 0.000001)
    G3_metrics['f1'] = (2 * G3_metrics['pre'] * G3_metrics['recall']) / (
            G3_metrics['pre'] + G3_metrics['recall'] + 0.000001)
    G4_metrics['f1'] = (2 * G4_metrics['pre'] * G4_metrics['recall']) / (
            G4_metrics['pre'] + G4_metrics['recall'] + 0.000001)
    all_metrics_Grade['f1'] = G2_metrics['f1'] * label_all_Grade.count(0) / len(label_all_Grade) + \
                              G3_metrics['f1'] * label_all_Grade.count(1) / len(label_all_Grade) + \
                              G4_metrics['f1'] * label_all_Grade.count(2) / len(label_all_Grade)

    # AUC
    if not external:
        out_cls_all_softmax_Grade = F.softmax(torch.from_numpy(np.array(predicted_all_Grade)), dim=1).numpy()
        label_all_np = np.array(label_all_Grade)
        label_all_onehot = make_one_hot(label_all_np)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_Grade[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(3):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= 3
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), out_cls_all_softmax_Grade.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        all_metrics_Grade['AUC'] = roc_auc["micro"]
        G2_metrics['AUC'] = roc_auc[0]
        G3_metrics['AUC'] = roc_auc[1]
        G4_metrics['AUC'] = roc_auc[2]
    else:

        out_cls_all_softmax_Grade = F.softmax(torch.from_numpy(np.array(predicted_all_Grade)), dim=1).numpy()
        label_all_np = np.array(label_all_Grade)
        label_all_onehot = make_one_hot(label_all_np, N=3)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(3):
            fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_Grade[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(3):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= 3
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), out_cls_all_softmax_Grade.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        all_metrics_Grade['AUC'] = roc_auc["micro"]
        G2_metrics['AUC'] = 0
        G3_metrics['AUC'] = 0
        G4_metrics['AUC'] = roc_auc[2]

    list_Grade = (
        Acc_Grade, all_metrics_Grade['sen'], all_metrics_Grade['spec'], all_metrics_Grade['pre'],
        all_metrics_Grade['recall']
        , all_metrics_Grade['f1'], all_metrics_Grade['AUC'])
    list_G2 = (0, G2_metrics['sen'], G2_metrics['spec'], G2_metrics['pre'], G2_metrics['recall']
               , G2_metrics['f1'], G2_metrics['AUC'])
    list_G3 = (0, G3_metrics['sen'], G3_metrics['spec'], G3_metrics['pre'], G3_metrics['recall']
               , G3_metrics['f1'], G3_metrics['AUC'])
    list_G4 = (0, G4_metrics['sen'], G4_metrics['spec'], G4_metrics['pre'], G4_metrics['recall']
               , G4_metrics['f1'], G4_metrics['AUC'])

    return list_His, list_A, list_O, list_GBM, list_Grade, list_G2,\
    list_G3, list_G4


def  test_endtoend_Diag(opt,Mine_model_init,Mine_model_body,Mine_model_Cls, dataloader, gpuID,external=False):
    Mine_model_init.eval()
    Mine_model_body.eval()
    Mine_model_Cls.eval()

    if 1:
        GBM_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                     'AUC': 0}
        G4A_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                     'AUC': 0}
        G3A_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                       'AUC': 0}
        G2A_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                       'AUC': 0}
        G3O_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                       'AUC': 0}
        G2O_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                       'AUC': 0}
        all_metrics = {'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        label_all = []
        predicted_all = []
        GBM_metrics_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                               'AUC': 0}
        G4A_metrics_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                               'AUC': 0}
        G3A_metrics_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                               'AUC': 0}
        G2A_metrics_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                               'AUC': 0}
        G3O_metrics_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                               'AUC': 0}
        G2O_metrics_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                               'AUC': 0}
        all_metrics_patient = {'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        label_all_patient = []
        predicted_all_patient = []

    test_bar = tqdm(dataloader)
    count = 0
    count_patient = 0
    correct = 0
    correct_patient = 0
    Patient_predori = {}

    for packs in test_bar:
        img = packs[0]
        label = packs[1]
        file_name=packs[2]
        patient_name=packs[3][0]
        if torch.cuda.is_available():
            img = img.cuda(gpuID[0])
            label = label.cuda(gpuID[0])
        GT = label[:, 1]

        init_feature = Mine_model_init(img)  # (BS,2500,1024)
        hidden_states = Mine_model_body(init_feature)
        results_dict = Mine_model_Cls(hidden_states)
        pred_ori = results_dict['logits']

        _, pred = torch.max(pred_ori.data, 1)
        pred = pred.tolist()  # [BS] A  O GBM //0 1 2
        gt = GT.tolist() #[BS] A  O GBM//0 1 2

        ####patient
        if gt[0] != 6:
            if patient_name not in Patient_predori:
                Patient_predori[patient_name] = []
                Patient_predori[patient_name].append(F.softmax(pred_ori, dim=1).detach().cpu().numpy()[0])
                label_all_patient.append(gt[0])
                count_patient += 1
            else:
                Patient_predori[patient_name].append(F.softmax(pred_ori, dim=1).detach().cpu().numpy()[0])

        ####wsi
        if gt[0] != 6:
            label_all.append(gt[0])
            predicted_all.append(pred_ori.detach().cpu().numpy()[0])
            count += 1
            if gt[0] == pred[0]:
                correct += 1
            # GBM
            if gt[0] == 0:
                if pred[0] == 0:
                    GBM_metrics['tp'] += 1
                else:
                    GBM_metrics['fn'] += 1
            else:
                if not pred[0] == 0:
                    GBM_metrics['tn'] += 1
                else:
                    GBM_metrics['fp'] += 1
            # G4A
            if gt[0] == 1:
                if pred[0] == 1:
                    G4A_metrics['tp'] += 1
                else:
                    G4A_metrics['fn'] += 1
            else:
                if not pred[0] == 1:
                    G4A_metrics['tn'] += 1
                else:
                    G4A_metrics['fp'] += 1
            # G3A
            if gt[0] == 2:
                if pred[0] == 2:
                    G3A_metrics['tp'] += 1
                else:
                    G3A_metrics['fn'] += 1
            else:
                if not pred[0] == 2:
                    G3A_metrics['tn'] += 1
                else:
                    G3A_metrics['fp'] += 1
            # G2A
            if gt[0] == 3:
                if pred[0] == 3:
                    G2A_metrics['tp'] += 1
                else:
                    G2A_metrics['fn'] += 1
            else:
                if not pred[0] == 3:
                    G2A_metrics['tn'] += 1
                else:
                    G2A_metrics['fp'] += 1
            # G3O
            if gt[0] == 4:
                if pred[0] == 4:
                    G3O_metrics['tp'] += 1
                else:
                    G3O_metrics['fn'] += 1
            else:
                if not pred[0] == 4:
                    G3O_metrics['tn'] += 1
                else:
                    G3O_metrics['fp'] += 1
            # G2O
            if gt[0] == 5:
                if pred[0] == 5:
                    G2O_metrics['tp'] += 1
                else:
                    G2O_metrics['fn'] += 1
            else:
                if not pred[0] == 5:
                    G2O_metrics['tn'] += 1
                else:
                    G2O_metrics['fp'] += 1

    ####patient process
    for patient, pred_wsis in Patient_predori.items():
        predicted_all_patient.append(np.mean(np.array(Patient_predori[patient]),axis=0))
    for i in range(len(label_all_patient)):
        count_patient += 1
        pred_patient=np.argmax(predicted_all_patient[i])
        if label_all_patient[i] == pred_patient:
            correct_patient += 1
        # GBM
        if label_all_patient[i] == 0:
            if pred_patient == 0:
                GBM_metrics_patient['tp'] += 1
            else:
                GBM_metrics_patient['fn'] += 1
        else:
            if not pred_patient == 0:
                GBM_metrics_patient['tn'] += 1
            else:
                GBM_metrics_patient['fp'] += 1
        # G4A
        if label_all_patient[i] == 1:
            if pred_patient == 1:
                G4A_metrics_patient['tp'] += 1
            else:
                G4A_metrics_patient['fn'] += 1
        else:
            if not pred_patient == 1:
                G4A_metrics_patient['tn'] += 1
            else:
                G4A_metrics_patient['fp'] += 1
        # G3A
        if label_all_patient[i] == 2:
            if pred_patient == 2:
                G3A_metrics_patient['tp'] += 1
            else:
                G3A_metrics_patient['fn'] += 1
        else:
            if not pred_patient == 2:
                G3A_metrics_patient['tn'] += 1
            else:
                G3A_metrics_patient['fp'] += 1
        # G2A
        if label_all_patient[i] == 3:
            if pred_patient == 3:
                G2A_metrics_patient['tp'] += 1
            else:
                G2A_metrics_patient['fn'] += 1
        else:
            if not pred_patient == 3:
                G2A_metrics_patient['tn'] += 1
            else:
                G2A_metrics_patient['fp'] += 1
        # G3O
        if label_all_patient[i] == 4:
            if pred_patient == 4:
                G3O_metrics_patient['tp'] += 1
            else:
                G3O_metrics_patient['fn'] += 1
        else:
            if not pred_patient == 4:
                G3O_metrics_patient['tn'] += 1
            else:
                G3O_metrics_patient['fp'] += 1
        # G2O
        if label_all_patient[i] == 5:
            if pred_patient == 5:
                G2O_metrics_patient['tp'] += 1
            else:
                G2O_metrics_patient['fn'] += 1
        else:
            if not pred_patient == 5:
                G2O_metrics_patient['tn'] += 1
            else:
                G2O_metrics_patient['fp'] += 1

    if 1: #################################### wsi
        ################################################ His
        Acc = correct / count

        #  Sensitivity
        GBM_metrics['sen'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fn']+0.000001)
        G4A_metrics['sen'] = (G4A_metrics['tp']) / (G4A_metrics['tp'] + G4A_metrics['fn'] + 0.000001)
        G3A_metrics['sen'] = (G3A_metrics['tp']) / (G3A_metrics['tp'] + G3A_metrics['fn'] + 0.000001)
        G2A_metrics['sen'] = (G2A_metrics['tp']) / (G2A_metrics['tp'] + G2A_metrics['fn'] + 0.000001)
        G3O_metrics['sen'] = (G3O_metrics['tp']) / (G3O_metrics['tp'] + G3O_metrics['fn'] + 0.000001)
        G2O_metrics['sen'] = (G2O_metrics['tp']) / (G2O_metrics['tp'] + G2O_metrics['fn'] + 0.000001)
        all_metrics['sen'] = GBM_metrics['sen'] * label_all.count(0) / len(label_all) + \
                             G4A_metrics['sen'] * label_all.count(1) / len(label_all) + \
                             G3A_metrics['sen'] * label_all.count(2) / len(label_all) + \
                             G2A_metrics['sen'] * label_all.count(3) / len(label_all) + \
                             G3O_metrics['sen'] * label_all.count(4) / len(label_all) + \
                             G2O_metrics['sen'] * label_all.count(5) / len(label_all)
        #  Spec
        GBM_metrics['spec'] = (GBM_metrics['tn']) / (GBM_metrics['tn'] + GBM_metrics['fp'] + 0.000001)
        G4A_metrics['spec'] = (G4A_metrics['tn']) / (G4A_metrics['tn'] + G4A_metrics['fp'] + 0.000001)
        G3A_metrics['spec'] = (G3A_metrics['tn']) / (G3A_metrics['tn'] + G3A_metrics['fp'] + 0.000001)
        G2A_metrics['spec'] = (G2A_metrics['tn']) / (G2A_metrics['tn'] + G2A_metrics['fp'] + 0.000001)
        G3O_metrics['spec'] = (G3O_metrics['tn']) / (G3O_metrics['tn'] + G3O_metrics['fp'] + 0.000001)
        G2O_metrics['spec'] = (G2O_metrics['tn']) / (G2O_metrics['tn'] + G2O_metrics['fp'] + 0.000001)
        all_metrics['spec'] = GBM_metrics['spec'] * label_all.count(0) / len(label_all) + \
                             G4A_metrics['spec'] * label_all.count(1) / len(label_all) + \
                             G3A_metrics['spec'] * label_all.count(2) / len(label_all) + \
                             G2A_metrics['spec'] * label_all.count(3) / len(label_all) + \
                             G3O_metrics['spec'] * label_all.count(4) / len(label_all) + \
                             G2O_metrics['spec'] * label_all.count(5) / len(label_all)
        #  Precision
        GBM_metrics['pre'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fp'] + 0.000001)
        G4A_metrics['pre'] = (G4A_metrics['tp']) / (G4A_metrics['tp'] + G4A_metrics['fp'] + 0.000001)
        G3A_metrics['pre'] = (G3A_metrics['tp']) / (G3A_metrics['tp'] + G3A_metrics['fp'] + 0.000001)
        G2A_metrics['pre'] = (G2A_metrics['tp']) / (G2A_metrics['tp'] + G2A_metrics['fp'] + 0.000001)
        G3O_metrics['pre'] = (G3O_metrics['tp']) / (G3O_metrics['tp'] + G3O_metrics['fp'] + 0.000001)
        G2O_metrics['pre'] = (G2O_metrics['tp']) / (G2O_metrics['tp'] + G2O_metrics['fp'] + 0.000001)
        all_metrics['pre'] = GBM_metrics['pre'] * label_all.count(0) / len(label_all) + \
                              G4A_metrics['pre'] * label_all.count(1) / len(label_all) + \
                              G3A_metrics['pre'] * label_all.count(2) / len(label_all) + \
                              G2A_metrics['pre'] * label_all.count(3) / len(label_all) + \
                              G3O_metrics['pre'] * label_all.count(4) / len(label_all) + \
                              G2O_metrics['pre'] * label_all.count(5) / len(label_all)
        #  Recall
        GBM_metrics['recall'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fn'] + 0.000001)
        G4A_metrics['recall'] = (G4A_metrics['tp']) / (G4A_metrics['tp'] + G4A_metrics['fn'] + 0.000001)
        G3A_metrics['recall'] = (G3A_metrics['tp']) / (G3A_metrics['tp'] + G3A_metrics['fn'] + 0.000001)
        G2A_metrics['recall'] = (G2A_metrics['tp']) / (G2A_metrics['tp'] + G2A_metrics['fn'] + 0.000001)
        G3O_metrics['recall'] = (G3O_metrics['tp']) / (G3O_metrics['tp'] + G3O_metrics['fn'] + 0.000001)
        G2O_metrics['recall'] = (G2O_metrics['tp']) / (G2O_metrics['tp'] + G2O_metrics['fn'] + 0.000001)
        all_metrics['recall'] = GBM_metrics['recall'] * label_all.count(0) / len(label_all) + \
                             G4A_metrics['recall'] * label_all.count(1) / len(label_all) + \
                             G3A_metrics['recall'] * label_all.count(2) / len(label_all) + \
                             G2A_metrics['recall'] * label_all.count(3) / len(label_all) + \
                             G3O_metrics['recall'] * label_all.count(4) / len(label_all) + \
                             G2O_metrics['recall'] * label_all.count(5) / len(label_all)
        #  F1
        GBM_metrics['f1'] = (2 * GBM_metrics['pre'] * GBM_metrics['recall']) / (
                    GBM_metrics['pre'] + GBM_metrics['recall']+0.000001)
        G4A_metrics['f1'] = (2 * G4A_metrics['pre'] * G4A_metrics['recall']) / (
                G4A_metrics['pre'] + G4A_metrics['recall'] + 0.000001)
        G3A_metrics['f1'] = (2 * G3A_metrics['pre'] * G3A_metrics['recall']) / (
                G3A_metrics['pre'] + G3A_metrics['recall'] + 0.000001)
        G2A_metrics['f1'] = (2 * G2A_metrics['pre'] * G2A_metrics['recall']) / (
                G2A_metrics['pre'] + G2A_metrics['recall'] + 0.000001)
        G3O_metrics['f1'] = (2 * G3O_metrics['pre'] * G3O_metrics['recall']) / (
                G3O_metrics['pre'] + G3O_metrics['recall'] + 0.000001)
        G2O_metrics['f1'] = (2 * G2O_metrics['pre'] * G2O_metrics['recall']) / (
                G2O_metrics['pre'] + G2O_metrics['recall'] + 0.000001)
        all_metrics['f1'] = GBM_metrics['f1'] * label_all.count(0) / len(label_all) + \
                                G4A_metrics['f1'] * label_all.count(1) / len(label_all) + \
                                G3A_metrics['f1'] * label_all.count(2) / len(label_all) + \
                                G2A_metrics['f1'] * label_all.count(3) / len(label_all) + \
                                G3O_metrics['f1'] * label_all.count(4) / len(label_all) + \
                                G2O_metrics['f1'] * label_all.count(5) / len(label_all)

        # AUC
        if not external:
            out_cls_all_softmax_His = F.softmax(torch.from_numpy(np.array(predicted_all)), dim=1).numpy()
            label_all_np = np.array(label_all)
            label_all_onehot = make_one_hot(label_all_np)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(6):
                fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_His[:, i])
                roc_auc[i]=auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(6)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(6):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= 6
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), out_cls_all_softmax_His.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            all_metrics['AUC'] = roc_auc["micro"]
            GBM_metrics['AUC'] = roc_auc[0]
            G4A_metrics['AUC'] = roc_auc[1]
            G3A_metrics['AUC'] = roc_auc[2]
            G2A_metrics['AUC'] = roc_auc[3]
            G3O_metrics['AUC'] = roc_auc[4]
            G2O_metrics['AUC'] = roc_auc[5]
        else:
            # L=len(predicted_all)
            # for kk in range(L):
            #     predicted_all[kk]=predicted_all[kk][0:2]
            out_cls_all_softmax_His = F.softmax(torch.from_numpy(np.array(predicted_all)), dim=1).numpy()
            label_all_np = np.array(label_all)
            label_all_onehot = make_one_hot(label_all_np,N=2)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(2):
                fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_His[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(2)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(2):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= 2
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), out_cls_all_softmax_His[:,0:2].ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            all_metrics['AUC'] = roc_auc["micro"]
            GBM_metrics['AUC'] = roc_auc[0]
            G4A_metrics['AUC'] = roc_auc[1]
            G3A_metrics['AUC'] = 0
            G2A_metrics['AUC'] = 0
            G3O_metrics['AUC'] = 0
            G2O_metrics['AUC'] = 0



        list = ( Acc, all_metrics['sen'], all_metrics['spec'],all_metrics['pre'],all_metrics['recall']
                     , all_metrics['f1'] ,all_metrics['AUC'])

        list_GBM = (None, GBM_metrics['sen'], GBM_metrics['spec'], GBM_metrics['pre'], GBM_metrics['recall']
                 , GBM_metrics['f1'], GBM_metrics['AUC'])
        list_G4A = (None, G4A_metrics['sen'], G4A_metrics['spec'], G4A_metrics['pre'], G4A_metrics['recall']
                    , G4A_metrics['f1'], G4A_metrics['AUC'])
        list_G3A = (None, G3A_metrics['sen'], G3A_metrics['spec'], G3A_metrics['pre'], G3A_metrics['recall']
                    , G3A_metrics['f1'], G3A_metrics['AUC'])
        list_G2A = (None, G2A_metrics['sen'], G2A_metrics['spec'], G2A_metrics['pre'], G2A_metrics['recall']
                    , G2A_metrics['f1'], G2A_metrics['AUC'])
        list_G3O = (None, G3O_metrics['sen'], G3O_metrics['spec'], G3O_metrics['pre'], G3O_metrics['recall']
                    , G3O_metrics['f1'], G3O_metrics['AUC'])
        list_G2O= (None, G2O_metrics['sen'], G2O_metrics['spec'], G2O_metrics['pre'], G2O_metrics['recall']
                    , G2O_metrics['f1'], G2O_metrics['AUC'])


    if 1: #################################### patient
        ################################################ His
        Acc_patient = correct_patient / count_patient

        #  Sensitivity
        GBM_metrics_patient['sen'] = (GBM_metrics_patient['tp']) / (
                    GBM_metrics_patient['tp'] + GBM_metrics_patient['fn'] + 0.000001)
        G4A_metrics_patient['sen'] = (G4A_metrics_patient['tp']) / (
                    G4A_metrics_patient['tp'] + G4A_metrics_patient['fn'] + 0.000001)
        G3A_metrics_patient['sen'] = (G3A_metrics_patient['tp']) / (
                    G3A_metrics_patient['tp'] + G3A_metrics_patient['fn'] + 0.000001)
        G2A_metrics_patient['sen'] = (G2A_metrics_patient['tp']) / (
                    G2A_metrics_patient['tp'] + G2A_metrics_patient['fn'] + 0.000001)
        G3O_metrics_patient['sen'] = (G3O_metrics_patient['tp']) / (
                    G3O_metrics_patient['tp'] + G3O_metrics_patient['fn'] + 0.000001)
        G2O_metrics_patient['sen'] = (G2O_metrics_patient['tp']) / (
                    G2O_metrics_patient['tp'] + G2O_metrics_patient['fn'] + 0.000001)
        all_metrics_patient['sen'] = GBM_metrics_patient['sen'] * label_all_patient.count(0) / len(label_all_patient) + \
                             G4A_metrics_patient['sen'] * label_all_patient.count(1) / len(label_all_patient) + \
                             G3A_metrics_patient['sen'] * label_all_patient.count(2) / len(label_all_patient) + \
                             G2A_metrics_patient['sen'] * label_all_patient.count(3) / len(label_all_patient) + \
                             G3O_metrics_patient['sen'] * label_all_patient.count(4) / len(label_all_patient) + \
                             G2O_metrics_patient['sen'] * label_all_patient.count(5) / len(label_all_patient)
        #  Spec
        GBM_metrics_patient['spec'] = (GBM_metrics_patient['tn']) / (
                    GBM_metrics_patient['tn'] + GBM_metrics_patient['fp'] + 0.000001)
        G4A_metrics_patient['spec'] = (G4A_metrics_patient['tn']) / (
                    G4A_metrics_patient['tn'] + G4A_metrics_patient['fp'] + 0.000001)
        G3A_metrics_patient['spec'] = (G3A_metrics_patient['tn']) / (
                    G3A_metrics_patient['tn'] + G3A_metrics_patient['fp'] + 0.000001)
        G2A_metrics_patient['spec'] = (G2A_metrics_patient['tn']) / (
                    G2A_metrics_patient['tn'] + G2A_metrics_patient['fp'] + 0.000001)
        G3O_metrics_patient['spec'] = (G3O_metrics_patient['tn']) / (
                    G3O_metrics_patient['tn'] + G3O_metrics_patient['fp'] + 0.000001)
        G2O_metrics_patient['spec'] = (G2O_metrics_patient['tn']) / (
                    G2O_metrics_patient['tn'] + G2O_metrics_patient['fp'] + 0.000001)
        all_metrics_patient['spec'] = GBM_metrics_patient['spec'] * label_all_patient.count(0) / len(label_all_patient) + \
                                     G4A_metrics_patient['spec'] * label_all_patient.count(1) / len(label_all_patient) + \
                                     G3A_metrics_patient['spec'] * label_all_patient.count(2) / len(label_all_patient) + \
                                     G2A_metrics_patient['spec'] * label_all_patient.count(3) / len(label_all_patient) + \
                                     G3O_metrics_patient['spec'] * label_all_patient.count(4) / len(label_all_patient) + \
                                     G2O_metrics_patient['spec'] * label_all_patient.count(5) / len(label_all_patient)
        #  Precision
        GBM_metrics_patient['pre'] = (GBM_metrics_patient['tp']) / (
                    GBM_metrics_patient['tp'] + GBM_metrics_patient['fp'] + 0.000001)
        G4A_metrics_patient['pre'] = (G4A_metrics_patient['tp']) / (
                    G4A_metrics_patient['tp'] + G4A_metrics_patient['fp'] + 0.000001)
        G3A_metrics_patient['pre'] = (G3A_metrics_patient['tp']) / (
                    G3A_metrics_patient['tp'] + G3A_metrics_patient['fp'] + 0.000001)
        G2A_metrics_patient['pre'] = (G2A_metrics_patient['tp']) / (
                    G2A_metrics_patient['tp'] + G2A_metrics_patient['fp'] + 0.000001)
        G3O_metrics_patient['pre'] = (G3O_metrics_patient['tp']) / (
                    G3O_metrics_patient['tp'] + G3O_metrics_patient['fp'] + 0.000001)
        G2O_metrics_patient['pre'] = (G2O_metrics_patient['tp']) / (
                    G2O_metrics_patient['tp'] + G2O_metrics_patient['fp'] + 0.000001)
        all_metrics_patient['pre'] = GBM_metrics_patient['pre'] * label_all_patient.count(0) / len(label_all_patient) + \
                                      G4A_metrics_patient['pre'] * label_all_patient.count(1) / len(label_all_patient) + \
                                      G3A_metrics_patient['pre'] * label_all_patient.count(2) / len(label_all_patient) + \
                                      G2A_metrics_patient['pre'] * label_all_patient.count(3) / len(label_all_patient) + \
                                      G3O_metrics_patient['pre'] * label_all_patient.count(4) / len(label_all_patient) + \
                                      G2O_metrics_patient['pre'] * label_all_patient.count(5) / len(label_all_patient)
        #  Recall
        GBM_metrics_patient['recall'] = (GBM_metrics_patient['tp']) / (
                    GBM_metrics_patient['tp'] + GBM_metrics_patient['fn'] + 0.000001)
        G4A_metrics_patient['recall'] = (G4A_metrics_patient['tp']) / (
                    G4A_metrics_patient['tp'] + G4A_metrics_patient['fn'] + 0.000001)
        G3A_metrics_patient['recall'] = (G3A_metrics_patient['tp']) / (
                    G3A_metrics_patient['tp'] + G3A_metrics_patient['fn'] + 0.000001)
        G2A_metrics_patient['recall'] = (G2A_metrics_patient['tp']) / (
                    G2A_metrics_patient['tp'] + G2A_metrics_patient['fn'] + 0.000001)
        G3O_metrics_patient['recall'] = (G3O_metrics_patient['tp']) / (
                    G3O_metrics_patient['tp'] + G3O_metrics_patient['fn'] + 0.000001)
        G2O_metrics_patient['recall'] = (G2O_metrics_patient['tp']) / (
                    G2O_metrics_patient['tp'] + G2O_metrics_patient['fn'] + 0.000001)
        all_metrics_patient['recall'] = GBM_metrics_patient['recall'] * label_all_patient.count(0) / len(label_all_patient) + \
                                     G4A_metrics_patient['recall'] * label_all_patient.count(1) / len(label_all_patient) + \
                                     G3A_metrics_patient['recall'] * label_all_patient.count(2) / len(label_all_patient) + \
                                     G2A_metrics_patient['recall'] * label_all_patient.count(3) / len(label_all_patient) + \
                                     G3O_metrics_patient['recall'] * label_all_patient.count(4) / len(label_all_patient) + \
                                     G2O_metrics_patient['recall'] * label_all_patient.count(5) / len(label_all_patient)
        #  F1
        GBM_metrics_patient['f1'] = (2 * GBM_metrics_patient['pre'] * GBM_metrics_patient['recall']) / (
                GBM_metrics_patient['pre'] + GBM_metrics_patient['recall'] + 0.000001)
        G4A_metrics_patient['f1'] = (2 * G4A_metrics_patient['pre'] * G4A_metrics_patient['recall']) / (
                G4A_metrics_patient['pre'] + G4A_metrics_patient['recall'] + 0.000001)
        G3A_metrics_patient['f1'] = (2 * G3A_metrics_patient['pre'] * G3A_metrics_patient['recall']) / (
                G3A_metrics_patient['pre'] + G3A_metrics_patient['recall'] + 0.000001)
        G2A_metrics_patient['f1'] = (2 * G2A_metrics_patient['pre'] * G2A_metrics_patient['recall']) / (
                G2A_metrics_patient['pre'] + G2A_metrics_patient['recall'] + 0.000001)
        G3O_metrics_patient['f1'] = (2 * G3O_metrics_patient['pre'] * G3O_metrics_patient['recall']) / (
                G3O_metrics_patient['pre'] + G3O_metrics_patient['recall'] + 0.000001)
        G2O_metrics_patient['f1'] = (2 * G2O_metrics_patient['pre'] * G2O_metrics_patient['recall']) / (
                G2O_metrics_patient['pre'] + G2O_metrics_patient['recall'] + 0.000001)
        all_metrics_patient['f1'] = GBM_metrics_patient['f1'] * label_all_patient.count(0) / len(label_all_patient) + \
                                        G4A_metrics_patient['f1'] * label_all_patient.count(1) / len(label_all_patient) + \
                                        G3A_metrics_patient['f1'] * label_all_patient.count(2) / len(label_all_patient) + \
                                        G2A_metrics_patient['f1'] * label_all_patient.count(3) / len(label_all_patient) + \
                                        G3O_metrics_patient['f1'] * label_all_patient.count(4) / len(label_all_patient) + \
                                        G2O_metrics_patient['f1'] * label_all_patient.count(5) / len(label_all_patient)

        # AUC

        if not external:
            out_cls_all_softmax_His = np.array(predicted_all_patient)
            label_all_np = np.array(label_all_patient)
            label_all_onehot = make_one_hot(label_all_np)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(6):
                fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_His[:, i])
                roc_auc[i]=auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(6)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(6):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= 6
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), out_cls_all_softmax_His.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            all_metrics_patient['AUC'] = roc_auc["micro"]
            GBM_metrics_patient['AUC'] = roc_auc[0]
            G4A_metrics_patient['AUC'] = roc_auc[1]
            G3A_metrics_patient['AUC'] = roc_auc[2]
            G2A_metrics_patient['AUC'] = roc_auc[3]
            G3O_metrics_patient['AUC'] = roc_auc[4]
            G2O_metrics_patient['AUC'] = roc_auc[5]


        else:
            # L = len(predicted_all_patient)
            # for kk in range(L):
            #     predicted_all_patient[kk] = predicted_all_patient[kk][0:2]
            out_cls_all_softmax_His = np.array(predicted_all_patient)
            label_all_np = np.array(label_all_patient)
            label_all_onehot = make_one_hot(label_all_np,N=2)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(2):
                fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_His[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(2)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(2):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= 2
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), out_cls_all_softmax_His[:,0:2].ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            all_metrics_patient['AUC'] = roc_auc["micro"]
            GBM_metrics_patient['AUC'] = roc_auc[0]
            G4A_metrics_patient['AUC'] = roc_auc[1]
            G3A_metrics_patient['AUC'] = 0
            G2A_metrics_patient['AUC'] = 0
            G3O_metrics_patient['AUC'] = 0
            G2O_metrics_patient['AUC'] = 0

        list_patient = (Acc_patient, all_metrics_patient['sen'], all_metrics_patient['spec'], all_metrics_patient['pre'],
                all_metrics_patient['recall']
                , all_metrics_patient['f1'], all_metrics_patient['AUC'])

        list_GBM_patient = (None, GBM_metrics_patient['sen'], GBM_metrics_patient['spec'], GBM_metrics_patient['pre'],
                    GBM_metrics_patient['recall']
                    , GBM_metrics_patient['f1'], GBM_metrics_patient['AUC'])
        list_G4A_patient = (None, G4A_metrics_patient['sen'], G4A_metrics_patient['spec'], G4A_metrics_patient['pre'],
                    G4A_metrics_patient['recall']
                    , G4A_metrics_patient['f1'], G4A_metrics_patient['AUC'])
        list_G3A_patient = (None, G3A_metrics_patient['sen'], G3A_metrics_patient['spec'], G3A_metrics_patient['pre'],
                    G3A_metrics_patient['recall']
                    , G3A_metrics_patient['f1'], G3A_metrics_patient['AUC'])
        list_G2A_patient = (None, G2A_metrics_patient['sen'], G2A_metrics_patient['spec'], G2A_metrics_patient['pre'],
                    G2A_metrics_patient['recall']
                    , G2A_metrics_patient['f1'], G2A_metrics_patient['AUC'])
        list_G3O_patient = (None, G3O_metrics_patient['sen'], G3O_metrics_patient['spec'], G3O_metrics_patient['pre'],
                    G3O_metrics_patient['recall']
                    , G3O_metrics_patient['f1'], G3O_metrics_patient['AUC'])
        list_G2O_patient = (None, G2O_metrics_patient['sen'], G2O_metrics_patient['spec'], G2O_metrics_patient['pre'],
                    G2O_metrics_patient['recall']
                    , G2O_metrics_patient['f1'], G2O_metrics_patient['AUC']) #################################### wsi#################################### wsi

    return list,list_GBM,list_G4A,list_G3A,list_G2A,list_G3O,list_G2O,list_patient,list_GBM_patient\
        ,list_G4A_patient,list_G3A_patient,list_G2A_patient,list_G3O_patient,list_G2O_patient


def test_endtoend_DiagSim(opt, Mine_model_init, Mine_model_body, Mine_model_Cls, dataloader, gpuID, external=False):
    Mine_model_init.eval()
    Mine_model_body.eval()
    Mine_model_Cls.eval()

    if 1:
        GBM_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                       'AUC': 0}
        G4A_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                       'AUC': 0}
        G23A_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                        'AUC': 0}
        G23O_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                        'AUC': 0}
        all_metrics = {'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        label_all = []
        predicted_all = []
        GBM_metrics_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                               'AUC': 0}
        G4A_metrics_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                               'AUC': 0}
        G23A_metrics_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                                'AUC': 0}
        G23O_metrics_patient = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                                'AUC': 0}

        all_metrics_patient = {'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        label_all_patient = []
        predicted_all_patient = []

    test_bar = tqdm(dataloader)
    count = 0
    count_patient = 0
    correct = 0
    correct_patient = 0
    Patient_predori = {}
    Patient_label = []
    for packs in test_bar:
        img = packs[0]
        label = packs[1]
        file_name = packs[2]
        patient_name = packs[3][0]
        img = img.cuda(gpuID[0])
        label = label.cuda(gpuID[0])
        GT = label[:, 0]

        init_feature = Mine_model_init(img)  # (BS,2500,1024)
        hidden_states = Mine_model_body(init_feature)
        results_dict = Mine_model_Cls(hidden_states)
        pred_ori = results_dict['logits']

        _, pred = torch.max(pred_ori.data, 1)
        pred = pred.tolist()  # [BS] A  O GBM //0 1 2
        gt = GT.tolist()  # [BS] A  O GBM//0 1 2

        ####patient
        if gt[0] != 4:
            if patient_name not in Patient_predori:
                Patient_predori[patient_name] = []
                Patient_predori[patient_name].append(F.softmax(pred_ori, dim=1).detach().cpu().numpy()[0])
                label_all_patient.append(gt[0])
                count_patient += 1
            else:
                Patient_predori[patient_name].append(F.softmax(pred_ori, dim=1).detach().cpu().numpy()[0])

        if gt[0] != 4:
            label_all.append(gt[0])
            predicted_all.append(pred_ori.detach().cpu().numpy()[0])
            count += 1
            if gt[0] == pred[0]:
                correct += 1
            # GBM
            if gt[0] == 0:
                if pred[0] == 0:
                    GBM_metrics['tp'] += 1
                else:
                    GBM_metrics['fn'] += 1
            else:
                if not pred[0] == 0:
                    GBM_metrics['tn'] += 1
                else:
                    GBM_metrics['fp'] += 1
            # G4A
            if gt[0] == 1:
                if pred[0] == 1:
                    G4A_metrics['tp'] += 1
                else:
                    G4A_metrics['fn'] += 1
            else:
                if not pred[0] == 1:
                    G4A_metrics['tn'] += 1
                else:
                    G4A_metrics['fp'] += 1
            # G23A
            if gt[0] == 2:
                if pred[0] == 2:
                    G23A_metrics['tp'] += 1
                else:
                    G23A_metrics['fn'] += 1
            else:
                if not pred[0] == 2:
                    G23A_metrics['tn'] += 1
                else:
                    G23A_metrics['fp'] += 1
            # G23O
            if gt[0] == 4:
                if pred[0] == 4:
                    G23O_metrics['tp'] += 1
                else:
                    G23O_metrics['fn'] += 1
            else:
                if not pred[0] == 4:
                    G23O_metrics['tn'] += 1
                else:
                    G23O_metrics['fp'] += 1

    ####patient process
    for patient, pred_wsis in Patient_predori.items():
        predicted_all_patient.append(np.mean(np.array(Patient_predori[patient]), axis=0))
    for i in range(len(label_all_patient)):
        count_patient += 1
        pred_patient = np.argmax(predicted_all_patient[i])
        if label_all_patient[i] == pred_patient:
            correct_patient += 1
        # GBM
        if label_all_patient[i] == 0:
            if pred_patient == 0:
                GBM_metrics_patient['tp'] += 1
            else:
                GBM_metrics_patient['fn'] += 1
        else:
            if not pred_patient == 0:
                GBM_metrics_patient['tn'] += 1
            else:
                GBM_metrics_patient['fp'] += 1
        # G4A
        if label_all_patient[i] == 1:
            if pred_patient == 1:
                G4A_metrics_patient['tp'] += 1
            else:
                G4A_metrics_patient['fn'] += 1
        else:
            if not pred_patient == 1:
                G4A_metrics_patient['tn'] += 1
            else:
                G4A_metrics_patient['fp'] += 1
        # G23A
        if label_all_patient[i] == 2:
            if pred_patient == 2:
                G23A_metrics_patient['tp'] += 1
            else:
                G23A_metrics_patient['fn'] += 1
        else:
            if not pred_patient == 2:
                G23A_metrics_patient['tn'] += 1
            else:
                G23A_metrics_patient['fp'] += 1
        # G23O
        if label_all_patient[i] == 3:
            if pred_patient == 3:
                G23O_metrics_patient['tp'] += 1
            else:
                G23O_metrics_patient['fn'] += 1
        else:
            if not pred_patient == 3:
                G23O_metrics_patient['tn'] += 1
            else:
                G23O_metrics_patient['fp'] += 1

    if 1:
        ################################################ His
        Acc = correct / count

        #  Sensitivity
        GBM_metrics['sen'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fn'] + 0.000001)
        G4A_metrics['sen'] = (G4A_metrics['tp']) / (G4A_metrics['tp'] + G4A_metrics['fn'] + 0.000001)
        G23A_metrics['sen'] = (G23A_metrics['tp']) / (G23A_metrics['tp'] + G23A_metrics['fn'] + 0.000001)
        G23O_metrics['sen'] = (G23O_metrics['tp']) / (G23O_metrics['tp'] + G23O_metrics['fn'] + 0.000001)
        all_metrics['sen'] = GBM_metrics['sen'] * label_all.count(0) / len(label_all) + \
                                    G4A_metrics['sen'] * label_all.count(1) / len(label_all) + \
                                    G23A_metrics['sen'] * label_all.count(2) / len(label_all) + \
                                    G23O_metrics['sen'] * label_all.count(3) / len(label_all)

        #  Spec
        GBM_metrics['spec'] = (GBM_metrics['tn']) / (GBM_metrics['tn'] + GBM_metrics['fp'] + 0.000001)
        G4A_metrics['spec'] = (G4A_metrics['tn']) / (G4A_metrics['tn'] + G4A_metrics['fp'] + 0.000001)
        G23A_metrics['spec'] = (G23A_metrics['tn']) / (G23A_metrics['tn'] + G23A_metrics['fp'] + 0.000001)
        G23O_metrics['spec'] = (G23O_metrics['tn']) / (G23O_metrics['tn'] + G23O_metrics['fp'] + 0.000001)
        all_metrics['spec'] = GBM_metrics['spec'] * label_all.count(0) / len(label_all) + \
                             G4A_metrics['spec'] * label_all.count(1) / len(label_all) + \
                             G23A_metrics['spec'] * label_all.count(2) / len(label_all) + \
                             G23O_metrics['spec'] * label_all.count(3) / len(label_all)
        #  Precision
        GBM_metrics['pre'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fp'] + 0.000001)
        G4A_metrics['pre'] = (G4A_metrics['tp']) / (G4A_metrics['tp'] + G4A_metrics['fp'] + 0.000001)
        G23A_metrics['pre'] = (G23A_metrics['tp']) / (G23A_metrics['tp'] + G23A_metrics['fp'] + 0.000001)
        G23O_metrics['pre'] = (G23O_metrics['tp']) / (G23O_metrics['tp'] + G23O_metrics['fp'] + 0.000001)
        all_metrics['pre'] = GBM_metrics['pre'] * label_all.count(0) / len(label_all) + \
                              G4A_metrics['pre'] * label_all.count(1) / len(label_all) + \
                              G23A_metrics['pre'] * label_all.count(2) / len(label_all) + \
                              G23O_metrics['pre'] * label_all.count(3) / len(label_all)
        #  Recall
        GBM_metrics['recall'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fn'] + 0.000001)
        G4A_metrics['recall'] = (G4A_metrics['tp']) / (G4A_metrics['tp'] + G4A_metrics['fn'] + 0.000001)
        G23A_metrics['recall'] = (G23A_metrics['tp']) / (G23A_metrics['tp'] + G23A_metrics['fn'] + 0.000001)
        G23O_metrics['recall'] = (G23O_metrics['tp']) / (G23O_metrics['tp'] + G23O_metrics['fn'] + 0.000001)
        all_metrics['recall'] = GBM_metrics['recall'] * label_all.count(0) / len(label_all) + \
                             G4A_metrics['recall'] * label_all.count(1) / len(label_all) + \
                             G23A_metrics['recall'] * label_all.count(2) / len(label_all) + \
                             G23O_metrics['recall'] * label_all.count(3) / len(label_all)
        #  F1
        GBM_metrics['f1'] = (2 * GBM_metrics['pre'] * GBM_metrics['recall']) / (
                GBM_metrics['pre'] + GBM_metrics['recall'] + 0.000001)
        G4A_metrics['f1'] = (2 * G4A_metrics['pre'] * G4A_metrics['recall']) / (
                G4A_metrics['pre'] + G4A_metrics['recall'] + 0.000001)
        G23A_metrics['f1'] = (2 * G23A_metrics['pre'] * G23A_metrics['recall']) / (
                G23A_metrics['pre'] + G23A_metrics['recall'] + 0.000001)
        G23O_metrics['f1'] = (2 * G23O_metrics['pre'] * G23O_metrics['recall']) / (
                G23O_metrics['pre'] + G23O_metrics['recall'] + 0.000001)
        all_metrics['f1'] = GBM_metrics['f1'] * label_all.count(0) / len(label_all) + \
                                G4A_metrics['f1'] * label_all.count(1) / len(label_all) + \
                                G23A_metrics['f1'] * label_all.count(2) / len(label_all) + \
                                G23O_metrics['f1'] * label_all.count(3) / len(label_all)

        # AUC
        if not external:
            out_cls_all_softmax_His = F.softmax(torch.from_numpy(np.array(predicted_all)), dim=1).numpy()
            label_all_np = np.array(label_all)
            label_all_onehot = make_one_hot(label_all_np)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(4):
                fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_His[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(4):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= 4
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), out_cls_all_softmax_His.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            all_metrics['AUC'] = roc_auc["micro"]
            GBM_metrics['AUC'] = roc_auc[0]
            G4A_metrics['AUC'] = roc_auc[1]
            G23A_metrics['AUC'] = roc_auc[2]
            G23O_metrics['AUC'] = roc_auc[3]
        else:
            # L = len(predicted_all)
            # for kk in range(L):
            #     predicted_all[kk] = predicted_all[kk][0:2]
            out_cls_all_softmax_His = F.softmax(torch.from_numpy(np.array(predicted_all)), dim=1).numpy()
            label_all_np = np.array(label_all)
            label_all_onehot = make_one_hot(label_all_np,N=4)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(2):
                fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_His[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(2)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(2):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= 2
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), out_cls_all_softmax_His.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            all_metrics['AUC'] = roc_auc["micro"]
            GBM_metrics['AUC'] = roc_auc[0]
            G4A_metrics['AUC'] = roc_auc[1]
            G23A_metrics['AUC'] = 0
            G23O_metrics['AUC'] = 0

        list = (Acc, all_metrics['sen'], all_metrics['spec'], all_metrics['pre'], all_metrics['recall']
                , all_metrics['f1'], all_metrics['AUC'])

        list_GBM = (0, GBM_metrics['sen'], GBM_metrics['spec'], GBM_metrics['pre'], GBM_metrics['recall']
                    , GBM_metrics['f1'], GBM_metrics['AUC'])
        list_G4A = (0, G4A_metrics['sen'], G4A_metrics['spec'], G4A_metrics['pre'], G4A_metrics['recall']
                    , G4A_metrics['f1'], G4A_metrics['AUC'])
        list_G23A = (0, G23A_metrics['sen'], G23A_metrics['spec'], G23A_metrics['pre'], G23A_metrics['recall']
                     , G23A_metrics['f1'], G23A_metrics['AUC'])
        list_G23O = (0, G23O_metrics['sen'], G23O_metrics['spec'], G23O_metrics['pre'], G23O_metrics['recall']
                     , G23O_metrics['f1'], G23O_metrics['AUC'])
    if 1:
        ################################################ His
        Acc_patient = correct_patient / count_patient

        #  Sensitivity
        GBM_metrics_patient['sen'] = (GBM_metrics_patient['tp']) / (
                GBM_metrics_patient['tp'] + GBM_metrics_patient['fn'] + 0.000001)
        G4A_metrics_patient['sen'] = (G4A_metrics_patient['tp']) / (
                G4A_metrics_patient['tp'] + G4A_metrics_patient['fn'] + 0.000001)
        G23A_metrics_patient['sen'] = (G23A_metrics_patient['tp']) / (
                G23A_metrics_patient['tp'] + G23A_metrics_patient['fn'] + 0.000001)
        G23O_metrics_patient['sen'] = (G23O_metrics_patient['tp']) / (
                G23O_metrics_patient['tp'] + G23O_metrics_patient['fn'] + 0.000001)
        all_metrics_patient['sen'] = GBM_metrics_patient['sen'] * label_all_patient.count(0) / len(label_all_patient) + \
                            G4A_metrics_patient['sen'] * label_all_patient.count(1) / len(label_all_patient) + \
                            G23A_metrics_patient['sen'] * label_all_patient.count(2) / len(label_all_patient) + \
                            G23O_metrics_patient['sen'] * label_all_patient.count(3) / len(label_all_patient)
        #  Spec
        GBM_metrics_patient['spec'] = (GBM_metrics_patient['tn']) / (
                GBM_metrics_patient['tn'] + GBM_metrics_patient['fp'] + 0.000001)
        G4A_metrics_patient['spec'] = (G4A_metrics_patient['tn']) / (
                G4A_metrics_patient['tn'] + G4A_metrics_patient['fp'] + 0.000001)
        G23A_metrics_patient['spec'] = (G23A_metrics_patient['tn']) / (
                G23A_metrics_patient['tn'] + G23A_metrics_patient['fp'] + 0.000001)
        G23O_metrics_patient['spec'] = (G23O_metrics_patient['tn']) / (
                G23O_metrics_patient['tn'] + G23O_metrics_patient['fp'] + 0.000001)
        all_metrics_patient['spec'] = GBM_metrics_patient['spec'] * label_all_patient.count(0) / len(label_all_patient) + \
                                     G4A_metrics_patient['spec'] * label_all_patient.count(1) / len(label_all_patient) + \
                                     G23A_metrics_patient['spec'] * label_all_patient.count(2) / len(label_all_patient) + \
                                     G23O_metrics_patient['spec'] * label_all_patient.count(3) / len(label_all_patient)
        #  Precision
        GBM_metrics_patient['pre'] = (GBM_metrics_patient['tp']) / (
                GBM_metrics_patient['tp'] + GBM_metrics_patient['fp'] + 0.000001)
        G4A_metrics_patient['pre'] = (G4A_metrics_patient['tp']) / (
                G4A_metrics_patient['tp'] + G4A_metrics_patient['fp'] + 0.000001)
        G23A_metrics_patient['pre'] = (G23A_metrics_patient['tp']) / (
                G23A_metrics_patient['tp'] + G23A_metrics_patient['fp'] + 0.000001)
        G23O_metrics_patient['pre'] = (G23O_metrics_patient['tp']) / (
                G23O_metrics_patient['tp'] + G23O_metrics_patient['fp'] + 0.000001)
        all_metrics_patient['pre'] = GBM_metrics_patient['pre'] * label_all_patient.count(0) / len(
            label_all_patient) + \
                                      G4A_metrics_patient['pre'] * label_all_patient.count(1) / len(
            label_all_patient) + \
                                      G23A_metrics_patient['pre'] * label_all_patient.count(2) / len(
            label_all_patient) + \
                                      G23O_metrics_patient['pre'] * label_all_patient.count(3) / len(label_all_patient)
        #  Recall
        GBM_metrics_patient['recall'] = (GBM_metrics_patient['tp']) / (
                GBM_metrics_patient['tp'] + GBM_metrics_patient['fn'] + 0.000001)
        G4A_metrics_patient['recall'] = (G4A_metrics_patient['tp']) / (
                G4A_metrics_patient['tp'] + G4A_metrics_patient['fn'] + 0.000001)
        G23A_metrics_patient['recall'] = (G23A_metrics_patient['tp']) / (
                G23A_metrics_patient['tp'] + G23A_metrics_patient['fn'] + 0.000001)
        G23O_metrics_patient['recall'] = (G23O_metrics_patient['tp']) / (
                G23O_metrics_patient['tp'] + G23O_metrics_patient['fn'] + 0.000001)
        all_metrics_patient['recall'] = GBM_metrics_patient['recall'] * label_all_patient.count(0) / len(label_all_patient) + \
                                     G4A_metrics_patient['recall'] * label_all_patient.count(1) / len(label_all_patient) + \
                                     G23A_metrics_patient['recall'] * label_all_patient.count(2) / len(label_all_patient) + \
                                     G23O_metrics_patient['recall'] * label_all_patient.count(3) / len(label_all_patient)
        #  F1
        GBM_metrics_patient['f1'] = (2 * GBM_metrics_patient['pre'] * GBM_metrics_patient['recall']) / (
                GBM_metrics_patient['pre'] + GBM_metrics_patient['recall'] + 0.000001)
        G4A_metrics_patient['f1'] = (2 * G4A_metrics_patient['pre'] * G4A_metrics_patient['recall']) / (
                G4A_metrics_patient['pre'] + G4A_metrics_patient['recall'] + 0.000001)
        G23A_metrics_patient['f1'] = (2 * G23A_metrics_patient['pre'] * G23A_metrics_patient['recall']) / (
                    G23A_metrics_patient['pre'] + G23A_metrics_patient['recall'] + 0.000001)
        G23O_metrics_patient['f1'] = (2 * G23O_metrics_patient['pre'] * G23O_metrics_patient['recall']) / (
                    G23O_metrics_patient['pre'] + G23O_metrics_patient['recall'] + 0.000001)
        all_metrics_patient['f1'] = GBM_metrics_patient['f1'] * label_all_patient.count(0) / len(
            label_all_patient) + \
                                        G4A_metrics_patient['f1'] * label_all_patient.count(1) / len(
            label_all_patient) + \
                                        G23A_metrics_patient['f1'] * label_all_patient.count(2) / len(
            label_all_patient) + \
                                        G23O_metrics_patient['f1'] * label_all_patient.count(3) / len(
            label_all_patient)

        # AUC
        if not external:
            out_cls_all_softmax_His = np.array(predicted_all_patient)
            label_all_np = np.array(label_all_patient)
            label_all_onehot = make_one_hot(label_all_np)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(4):
                fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_His[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(4):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= 4
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), out_cls_all_softmax_His.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            all_metrics_patient['AUC'] = roc_auc["micro"]
            GBM_metrics_patient['AUC'] = roc_auc[0]
            G4A_metrics_patient['AUC'] = roc_auc[1]
            G23A_metrics_patient['AUC'] = roc_auc[2]
            G23O_metrics_patient['AUC'] = roc_auc[3]

        else:
            # L = len(predicted_all_patient)
            # for kk in range(L):
            #     predicted_all_patient[kk] = predicted_all_patient[kk][0:2]
            out_cls_all_softmax_His = np.array(predicted_all_patient)
            label_all_np = np.array(label_all_patient)
            label_all_onehot = make_one_hot(label_all_np)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(2):
                fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_His[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(2)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(2):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= 2
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), out_cls_all_softmax_His.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            all_metrics_patient['AUC'] = roc_auc["micro"]
            GBM_metrics_patient['AUC'] = roc_auc[0]
            G4A_metrics_patient['AUC'] = roc_auc[1]
            G23A_metrics_patient['AUC'] = 0
            G23O_metrics_patient['AUC'] = 0


        list_patient = (
        Acc_patient, all_metrics_patient['sen'], all_metrics_patient['spec'], all_metrics_patient['pre'],
        all_metrics_patient['recall']
        , all_metrics_patient['f1'], all_metrics_patient['AUC'])

        list_GBM_patient = (0, GBM_metrics_patient['sen'], GBM_metrics_patient['spec'], GBM_metrics_patient['pre'],
                            GBM_metrics_patient['recall']
                            , GBM_metrics_patient['f1'], GBM_metrics_patient['AUC'])
        list_G4A_patient = (0, G4A_metrics_patient['sen'], G4A_metrics_patient['spec'], G4A_metrics_patient['pre'],
                            G4A_metrics_patient['recall']
                            , G4A_metrics_patient['f1'], G4A_metrics_patient['AUC'])
        list_G23A_patient = (0, G23A_metrics_patient['sen'], G23A_metrics_patient['spec'], G23A_metrics_patient['pre'],
                            G23A_metrics_patient['recall']
                            , G23A_metrics_patient['f1'], G23A_metrics_patient['AUC'])
        list_G23O_patient = (0, G23O_metrics_patient['sen'], G23O_metrics_patient['spec'], G23O_metrics_patient['pre'],
                            G23O_metrics_patient['recall']
                            , G23O_metrics_patient['f1'], G23O_metrics_patient['AUC'])

    return list, list_GBM, list_G4A, list_G23A, list_G23O, list_patient, list_GBM_patient \
        , list_G4A_patient, list_G23A_patient, list_G23O_patient


def validation_2016_multiclas(opt,Mine_model_init,Mine_model_His,Mine_model_Cls, dataloader, gpuID,epoch,opt_name=None):
    Mine_model_init.eval()
    Mine_model_His.eval()
    Mine_model_Cls.eval()


    if 1:

        count_His = 0
        count_Grade = 0
        correct_His = 0
        correct_Grade=0
        A_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                        'AUC': 0}
        O_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                        'AUC': 0}
        GBM_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                        'AUC': 0}
        all_metrics_His = {'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        label_all_His = []
        predicted_all_His = []

        G2_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                     'AUC': 0}
        G3_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                     'AUC': 0}
        G4_metrics = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0,
                       'AUC': 0}
        all_metrics_Grade = {'sen': 0, 'spec': 0, 'pre': 0, 'recall': 0, 'f1': 0, 'AUC': 0}
        label_all_Grade = []
        predicted_all_Grade = []

    test_bar = tqdm(dataloader)
    bs = opt['Val_batchSize']
    count = 0
    a_pred=0
    o_pred = 0
    gbm_pred = 0
    for packs in test_bar:
        img = packs[0]
        label = packs[1]
        file_name=packs[2]

        count += 1

        if torch.cuda.is_available():
            img = img.cuda(gpuID[0])
            label = label.cuda(gpuID[0])
        label_his = label[:, 0]
        label_grade = label[:, 1]


        # imp_his, imp_grade = imp_gene(opt,img)
        saliency_map_His, saliency_map_Grade = saliency_map_read(opt, file_name, epoch,opt_name=opt_name)
        saliency_map_His = torch.from_numpy(np.array(saliency_map_His)).float().cuda(gpuID[0])
        saliency_map_Grade = torch.from_numpy(np.array(saliency_map_Grade)).float().cuda(gpuID[0])
        init_feature = Mine_model_init(img)  # (BS,2500,1024)
        hidden_states_his, hidden_states_grade, encoded_His, encoded_Grade = Mine_model_His(init_feature,saliency_map_His, saliency_map_Grade)
        results_dict,saliency_A,saliency_O,saliency_GBM,saliency_G2,saliency_G3,saliency_G4= Mine_model_Cls(encoded_His, encoded_Grade)

        pred_His_ori = results_dict['logits_His']
        pred_Grade_ori = results_dict['logits_Grade']
        _, pred_His = torch.max(pred_His_ori.data, 1)
        pred_His = pred_His.tolist()  # [BS] A  O GBM //0 1 2
        gt_His = label_his.tolist() #[BS] A  O GBM//0 1 2
        _, pred_Grade = torch.max(pred_Grade_ori.data, 1)
        pred_Grade = pred_Grade.tolist()  # [BS] A  O GBM //0 1 2
        gt_Grade = label_grade.tolist()  # [BS] A  O GBM//0 1 2


        for j in range(bs):
            ##################   His
            # A
            if gt_His[j] == 0:
                if pred_His[j] == 0:
                    A_metrics['tp'] += 1
                else:
                    A_metrics['fn'] += 1
            else:
                if not pred_His[j] == 0:
                    A_metrics['tn'] += 1
                else:
                    A_metrics['fp'] += 1
            # O
            if gt_His[j] == 1:
                if pred_His[j] == 1:
                    O_metrics['tp'] += 1
                    o_pred += 1
                else:
                    if pred_His[j] == 0:
                        a_pred += 1
                    if pred_His[j] == 2:
                        gbm_pred += 1
                    O_metrics['fn'] += 1
            else:
                if not pred_His[j] == 1:
                    O_metrics['tn'] += 1
                else:
                    O_metrics['fp'] += 1
            # GBM
            if gt_His[j] == 2:
                if pred_His[j] == 2:
                    GBM_metrics['tp'] += 1
                else:
                    GBM_metrics['fn'] += 1
            else:
                if not pred_His[j] == 2:
                    GBM_metrics['tn'] += 1
                else:
                    GBM_metrics['fp'] += 1
            label_all_His.append(gt_His[j])
            predicted_all_His.append(pred_His_ori.detach().cpu().numpy()[j])
            count_His += 1
            if gt_His[j] == pred_His[j]:
                correct_His += 1
            ##################   Grade
            # G2
            if gt_Grade[j] == 0:
                if pred_Grade[j] == 0:
                    G2_metrics['tp'] += 1
                else:
                    G2_metrics['fn'] += 1
            else:
                if not pred_Grade[j] == 0:
                    G2_metrics['tn'] += 1
                else:
                    G2_metrics['fp'] += 1
            # G3
            if gt_Grade[j] == 1:
                if pred_Grade[j] == 1:
                    G3_metrics['tp'] += 1
                else:
                    G3_metrics['fn'] += 1
            else:
                if not pred_Grade[j] == 1:
                    G3_metrics['tn'] += 1
                else:
                    G3_metrics['fp'] += 1
            # G4
            if gt_Grade[j] == 2:
                if pred_Grade[j] == 2:
                    G4_metrics['tp'] += 1
                else:
                    G4_metrics['fn'] += 1
            else:
                if not pred_Grade[j] == 2:
                    G4_metrics['tn'] += 1
                else:
                    G4_metrics['fp'] += 1
            label_all_Grade.append(gt_Grade[j])
            predicted_all_Grade.append(pred_Grade_ori.detach().cpu().numpy()[j])
            count_Grade += 1
            if gt_Grade[j] == pred_Grade[j]:
                correct_Grade += 1


    ################################################ His
    Acc_His = correct_His / count_His

    #  Sensitivity
    A_metrics['sen'] = (A_metrics['tp']) / (A_metrics['tp'] + A_metrics['fn']+0.000001)
    O_metrics['sen'] = (O_metrics['tp']) / (O_metrics['tp'] + O_metrics['fn']+0.000001)
    GBM_metrics['sen'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fn']+0.000001)
    all_metrics_His['sen'] = (A_metrics['sen'] +  O_metrics['sen'] +
                          GBM_metrics['sen'] ) / 3
    all_His_sen_micro=(A_metrics['tp']+O_metrics['tp']+GBM_metrics['tp'])/\
                      ((A_metrics['tp']+O_metrics['tp']+GBM_metrics['tp'])+(A_metrics['fn']+O_metrics['fn']+GBM_metrics['fn']))
    all_His_sen_weight=A_metrics['sen']*label_all_His.count(0)/len(label_all_His)+O_metrics['sen']*label_all_His.count(1)/len(label_all_His)\
            +GBM_metrics['sen']*label_all_His.count(2)/len(label_all_His)
    #  Spec
    A_metrics['spec'] = (A_metrics['tn']) / (A_metrics['tn'] + A_metrics['fp']+0.000001)
    O_metrics['spec'] = (O_metrics['tn']) / (O_metrics['tn'] + O_metrics['fp']+0.000001)
    GBM_metrics['spec'] = (GBM_metrics['tn']) / (GBM_metrics['tn'] + GBM_metrics['fp']+0.000001)
    all_metrics_His['spec'] = (A_metrics['spec'] + O_metrics['spec'] +
                           GBM_metrics['spec'] ) / 3
    #  Precision
    A_metrics['pre'] = (A_metrics['tp']) / (A_metrics['tp'] + A_metrics['fp']+0.000001)
    O_metrics['pre'] = (O_metrics['tp']) / (O_metrics['tp'] + O_metrics['fp']+0.000001)
    GBM_metrics['pre'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fp']+0.000001)
    all_metrics_His['pre'] = (A_metrics['pre']  + O_metrics['pre'] +
                          GBM_metrics['pre'] ) / 3
    #  Recall
    A_metrics['recall'] = (A_metrics['tp']) / (A_metrics['tp'] + A_metrics['fn']+0.000001)
    O_metrics['recall'] = (O_metrics['tp']) / (O_metrics['tp'] + O_metrics['fn']+0.000001)
    GBM_metrics['recall'] = (GBM_metrics['tp']) / (GBM_metrics['tp'] + GBM_metrics['fn']+0.000001)
    all_metrics_His['recall'] = (A_metrics['recall']  + O_metrics['recall'] +
                             GBM_metrics['recall'] ) / 3
    #  F1
    A_metrics['f1'] = (2 * A_metrics['pre'] * A_metrics['recall']) / (
                A_metrics['pre'] + A_metrics['recall']+0.000001)
    O_metrics['f1'] = (2 * O_metrics['pre'] * O_metrics['recall']) / (
                O_metrics['pre'] + O_metrics['recall']+0.000001)
    GBM_metrics['f1'] = (2 * GBM_metrics['pre'] * GBM_metrics['recall']) / (GBM_metrics['pre'] + GBM_metrics['recall']+0.000001)
    all_metrics_His['f1'] = (A_metrics['f1']  + O_metrics['f1'] +
                          GBM_metrics['f1']) / 3
    # AUC
    out_cls_all_softmax_His = F.softmax(torch.from_numpy(np.array(predicted_all_His)), dim=1).numpy()
    label_all_np = np.array(label_all_His)
    label_all_onehot = make_one_hot(label_all_np)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_His[:, i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= 3
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    all_metrics_His['AUC'] = roc_auc["macro"]
    A_metrics['AUC'] = auc(fpr[0], tpr[0])
    O_metrics['AUC'] = auc(fpr[1], tpr[1])
    GBM_metrics['AUC'] = auc(fpr[2], tpr[2])
    k=A_metrics['AUC']*label_all_His.count(0)/len(label_all_His)+O_metrics['AUC']*label_all_His.count(1)/len(label_all_His)\
      +GBM_metrics['AUC']*label_all_His.count(2)/len(label_all_His)
    # Compute micro-average auc
    fpr["micro"], tpr["micro"], _ = roc_curve(label_all_onehot.ravel(), out_cls_all_softmax_His.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    ################################################ Grade
    Acc_Grade = correct_Grade / count_Grade

    #  Sensitivity
    G2_metrics['sen'] = (G2_metrics['tp']) / (G2_metrics['tp'] + G2_metrics['fn'] + 0.000001)
    G3_metrics['sen'] = (G3_metrics['tp']) / (G3_metrics['tp'] + G3_metrics['fn'] + 0.000001)
    G4_metrics['sen'] = (G4_metrics['tp']) / (G4_metrics['tp'] + G4_metrics['fn'] + 0.000001)
    all_metrics_Grade['sen'] = (G2_metrics['sen'] + G3_metrics['sen'] +
                                G4_metrics['sen']) / 3
    #  Spec
    G2_metrics['spec'] = (G2_metrics['tn']) / (G2_metrics['tn'] + G2_metrics['fp'] + 0.000001)
    G3_metrics['spec'] = (G3_metrics['tn']) / (G3_metrics['tn'] + G3_metrics['fp'] + 0.000001)
    G4_metrics['spec'] = (G4_metrics['tn']) / (G4_metrics['tn'] + G4_metrics['fp'] + 0.000001)
    all_metrics_Grade['spec'] = (G2_metrics['spec'] + G3_metrics['spec'] +
                                 G4_metrics['spec']) / 3
    #  Precision
    G2_metrics['pre'] = (G2_metrics['tp']) / (G2_metrics['tp'] + G2_metrics['fp'] + 0.000001)
    G3_metrics['pre'] = (G3_metrics['tp']) / (G3_metrics['tp'] + G3_metrics['fp'] + 0.000001)
    G4_metrics['pre'] = (G4_metrics['tp']) / (G4_metrics['tp'] + G4_metrics['fp'] + 0.000001)
    all_metrics_Grade['pre'] = (G2_metrics['pre'] + G3_metrics['pre'] +
                                G4_metrics['pre']) / 3
    #  Recall
    G2_metrics['recall'] = (G2_metrics['tp']) / (G2_metrics['tp'] + G2_metrics['fn'] + 0.000001)
    G3_metrics['recall'] = (G3_metrics['tp']) / (G3_metrics['tp'] + G3_metrics['fn'] + 0.000001)
    G4_metrics['recall'] = (G4_metrics['tp']) / (G4_metrics['tp'] + G4_metrics['fn'] + 0.000001)
    all_metrics_Grade['recall'] = (G2_metrics['recall'] + G3_metrics['recall'] +
                                   G4_metrics['recall']) / 3
    #  F1
    G2_metrics['f1'] = (2 * G2_metrics['pre'] * G2_metrics['recall']) / (
            G2_metrics['pre'] + G2_metrics['recall'] + 0.000001)
    G3_metrics['f1'] = (2 * G3_metrics['pre'] * G3_metrics['recall']) / (
            G3_metrics['pre'] + G3_metrics['recall'] + 0.000001)
    G4_metrics['f1'] = (2 * G4_metrics['pre'] * G4_metrics['recall']) / (
                G4_metrics['pre'] + G4_metrics['recall'] + 0.000001)
    all_metrics_Grade['f1'] = (G2_metrics['f1'] + G3_metrics['f1'] +
                               G4_metrics['f1']) / 3
    # AUC
    out_cls_all_softmax_Grade = F.softmax(torch.from_numpy(np.array(predicted_all_Grade)), dim=1).numpy()
    label_all_np = np.array(label_all_Grade)
    label_all_onehot = make_one_hot(label_all_np)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(label_all_onehot[:, i], out_cls_all_softmax_Grade[:, i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= 3
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    all_metrics_Grade['AUC'] = roc_auc["macro"]
    G2_metrics['AUC'] = auc(fpr[0], tpr[0])
    G3_metrics['AUC'] = auc(fpr[1], tpr[1])
    G4_metrics['AUC'] = auc(fpr[2], tpr[2])


    list_His = ( Acc_His, all_metrics_His['sen'], all_metrics_His['spec'],all_metrics_His['pre'],all_metrics_His['recall']
                 , all_metrics_His['f1'] ,all_metrics_His['AUC'])
    list_Grade = ( Acc_Grade, all_metrics_Grade['sen'], all_metrics_Grade['spec'], all_metrics_Grade['pre'], all_metrics_Grade['recall']
    , all_metrics_Grade['f1'], all_metrics_Grade['AUC'])

    list_A=(0, A_metrics['sen'], A_metrics['spec'],A_metrics['pre'],A_metrics['recall']
                 ,A_metrics['f1'] ,A_metrics['AUC'])
    list_O = (0, O_metrics['sen'], O_metrics['spec'], O_metrics['pre'], O_metrics['recall']
             , O_metrics['f1'], O_metrics['AUC'])
    list_GBM = (0, GBM_metrics['sen'], GBM_metrics['spec'], GBM_metrics['pre'], GBM_metrics['recall']
             , GBM_metrics['f1'], GBM_metrics['AUC'])
    list_G2 = (0, G2_metrics['sen'], G2_metrics['spec'], G2_metrics['pre'], G2_metrics['recall']
             , G2_metrics['f1'], G2_metrics['AUC'])
    list_G3 = (0, G3_metrics['sen'], G3_metrics['spec'], G3_metrics['pre'], G3_metrics['recall']
              , G3_metrics['f1'], G3_metrics['AUC'])
    list_G4 = (0, G4_metrics['sen'], G4_metrics['spec'], G4_metrics['pre'], G4_metrics['recall']
              , G4_metrics['f1'], G4_metrics['AUC'])


    return list_A,list_O,list_GBM,list_G2,list_G3,list_G4




def test_endtoend_stem(opt,Mine_model_init,Mine_model_body,Mine_model_Cls, dataloader, gpuID,epoch,testDataset_CPTAC,testDataset_IvYGAP,testDataset_GBMatch,testDataset_tiantan,testDataset_cam):
    metrics = ['acc', 'sen', 'spec', 'precision', 'recall', 'f1', 'AUC']
    if opt['Clstype'] == 'Diag':
        list_all, list_GBM, list_G4A, list_G3A, list_G2A, list_G3O, list_G2O,list_all_patient,list_GBM_patient\
        ,list_G4A_patient,list_G3A_patient,list_G2A_patient,list_G3O_patient,list_G2O_patient = test_endtoend_Diag(opt, Mine_model_init,Mine_model_body,Mine_model_Cls,dataloader, gpuID)
        print('test in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all[0], list_all[1], list_all[2], list_all[6]))
        print('test in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
        print('test in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A[1], list_G4A[2], list_G4A[6]))
        print('test in epoch: %d/%d, G3A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G3A[1], list_G3A[2], list_G3A[6]))
        print('test in epoch: %d/%d, G2A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G2A[1], list_G2A[2], list_G2A[6]))
        print('test in epoch: %d/%d, G3O || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G3O[1], list_G3O[2], list_G3O[6]))
        print('test in epoch: %d/%d, G2O || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G2O[1], list_G2O[2], list_G2O[6]))
        dataframe = pd.DataFrame({'metrics': metrics, 'all': list_all, 'GBM': list_GBM, 'G4A': list_G4A, 'G3A': list_G3A, 'G2A': list_G2A,'G3O': list_G3O, 'G2O': list_G2O})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi.xlsx', sheet_name='epoch' + str(epoch), index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi.xlsx', mode='a', engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
        print('test-patient in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_patient[0], list_all_patient[1], list_all_patient[2], list_all_patient[6]))
        print('test-patient in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM_patient[1], list_GBM_patient[2], list_GBM_patient[6]))
        print('test-patient in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A_patient[1], list_G4A_patient[2], list_G4A_patient[6]))
        print('test-patient in epoch: %d/%d, G3A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G3A_patient[1], list_G3A_patient[2], list_G3A_patient[6]))
        print('test-patient in epoch: %d/%d, G2A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G2A_patient[1], list_G2A_patient[2], list_G2A_patient[6]))
        print('test-patient in epoch: %d/%d, G3O || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G3O_patient[1], list_G3O_patient[2], list_G3O_patient[6]))
        print('test-patient in epoch: %d/%d, G2O || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G2O_patient[1], list_G2O_patient[2], list_G2O_patient[6]))
        dataframe = pd.DataFrame({'metrics': metrics, 'all': list_all_patient, 'GBM': list_GBM_patient, 'G4A': list_G4A_patient, 'G3A': list_G3A_patient, 'G2A': list_G2A_patient,'G3O': list_G3O_patient, 'G2O': list_G2O_patient})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-patient.xlsx', sheet_name='epoch' + str(epoch), index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-patient.xlsx', mode='a', engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
        if opt['TrainingSet'] == 'TCGA':
            ###############CPTAC
            list_all, list_GBM, list_G4A, list_G3A, list_G2A, list_G3O, list_G2O,list_all_patient,list_GBM_patient\
            ,list_G4A_patient,list_G3A_patient,list_G2A_patient,list_G3O_patient,list_G2O_patient = test_endtoend_Diag(opt, Mine_model_init,Mine_model_body,Mine_model_Cls,testDataset_CPTAC, gpuID,external=True)
            print('test-CPTAC in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all[0], list_all[1], list_all[2], list_all[6]))
            print('test-CPTAC in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
            print('test-CPTAC in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A[1], list_G4A[2], list_G4A[6]))
            dataframe = pd.DataFrame({'metrics': metrics, 'all': list_all, 'GBM': list_GBM, 'G4A': list_G4A})
            if epoch < 7:
                dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi-CPTAC.xlsx', sheet_name='epoch' + str(epoch), index=False)
            else:
                with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi-CPTAC.xlsx', mode='a', engine='openpyxl') as writer:
                    dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
            print('test-CPTAC-patient in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_patient[0], list_all_patient[1], list_all_patient[2], list_all_patient[6]))
            print('test-CPTAC-patient in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM_patient[1], list_GBM_patient[2], list_GBM_patient[6]))
            print('test-CPTAC-patient in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A_patient[1], list_G4A_patient[2], list_G4A_patient[6]))
            dataframe = pd.DataFrame({'metrics': metrics, 'all': list_all_patient, 'GBM': list_GBM_patient, 'G4A': list_G4A_patient})
            if epoch < 7:
                dataframe.to_excel('./logs/' + opt['name'] + '/result-patient-CPTAC.xlsx', sheet_name='epoch' + str(epoch), index=False)
            else:
                with pd.ExcelWriter('./logs/' + opt['name'] + '/result-patient-CPTAC.xlsx', mode='a', engine='openpyxl') as writer:
                    dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)

            ###############IvYGAP
            list_all, list_GBM, list_G4A, list_G3A, list_G2A, list_G3O, list_G2O,list_all_patient,list_GBM_patient\
            ,list_G4A_patient,list_G3A_patient,list_G2A_patient,list_G3O_patient,list_G2O_patient = test_endtoend_Diag(opt,Mine_model_init,Mine_model_body,Mine_model_Cls,testDataset_IvYGAP,gpuID,external=True)
            print('test-IvYGAP in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all[0], list_all[1], list_all[2], list_all[6]))
            print('test-IvYGAP in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
            print('test-IvYGAP in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A[1], list_G4A[2], list_G4A[6]))
            dataframe = pd.DataFrame({'metrics': metrics, 'all': list_all, 'GBM': list_GBM, 'G4A': list_G4A})
            if epoch < 7:
                dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi-IvYGAP.xlsx', sheet_name='epoch' + str(epoch),index=False)
            else:
                with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi-IvYGAP.xlsx', mode='a',engine='openpyxl') as writer:
                    dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
            print('test-IvYGAP-patient in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_patient[0], list_all_patient[1], list_all_patient[2], list_all_patient[6]))
            print('test-IvYGAP-patient in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM_patient[1], list_GBM_patient[2], list_GBM_patient[6]))
            print('test-IvYGAP-patient in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A_patient[1], list_G4A_patient[2], list_G4A_patient[6]))
            dataframe = pd.DataFrame({'metrics': metrics, 'all': list_all_patient, 'GBM': list_GBM_patient, 'G4A': list_G4A_patient})
            if epoch < 7:
                dataframe.to_excel('./logs/' + opt['name'] + '/result-patient-IvYGAP.xlsx', sheet_name='epoch' + str(epoch), index=False)
            else:
                with pd.ExcelWriter('./logs/' + opt['name'] + '/result-patient-IvYGAP.xlsx', mode='a', engine='openpyxl') as writer:
                    dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
            # ###############GBMatch
            list_all, list_GBM, list_G4A, list_G3A, list_G2A, list_G3O, list_G2O,list_all_patient,list_GBM_patient\
            ,list_G4A_patient,list_G3A_patient,list_G2A_patient,list_G3O_patient,list_G2O_patient = test_endtoend_Diag(opt,Mine_model_init,Mine_model_body,Mine_model_Cls,testDataset_GBMatch,gpuID,external=True)
            print('test-GBMatch in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all[0], list_all[1], list_all[2], list_all[6]))
            print('test-GBMatch in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
            print('test-GBMatch in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A[1], list_G4A[2], list_G4A[6]))
            dataframe = pd.DataFrame({'metrics': metrics, 'all': list_all, 'GBM': list_GBM, 'G4A': list_G4A})
            if epoch < 7:
                dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi-GBMatch.xlsx', sheet_name='epoch' + str(epoch),index=False)
            else:
                with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi-GBMatch.xlsx', mode='a',engine='openpyxl') as writer:
                    dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
            print('test-GBMatch-patient in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_patient[0], list_all_patient[1], list_all_patient[2], list_all_patient[6]))
            print('test-GBMatch-patient in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM_patient[1], list_GBM_patient[2], list_GBM_patient[6]))
            print('test-GBMatch-patient in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A_patient[1], list_G4A_patient[2], list_G4A_patient[6]))
            dataframe = pd.DataFrame({'metrics': metrics, 'all': list_all_patient, 'GBM': list_GBM_patient, 'G4A': list_G4A_patient})
            if epoch < 7:
                dataframe.to_excel('./logs/' + opt['name'] + '/result-patient-GBMatch.xlsx', sheet_name='epoch' + str(epoch), index=False)
            else:
                with pd.ExcelWriter('./logs/' + opt['name'] + '/result-patient-GBMatch.xlsx', mode='a', engine='openpyxl') as writer:
                    dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
            ###############tiantan
            list_all, list_GBM, list_G4A, list_G3A, list_G2A, list_G3O, list_G2O,list_all_patient,list_GBM_patient\
            ,list_G4A_patient,list_G3A_patient,list_G2A_patient,list_G3O_patient,list_G2O_patient = test_endtoend_Diag(opt,Mine_model_init,Mine_model_body,Mine_model_Cls,testDataset_tiantan,gpuID)
            print('test-tiantan in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all[0], list_all[1], list_all[2], list_all[6]))
            print('test-tiantan in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
            print('test-tiantan in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A[1], list_G4A[2], list_G4A[6]))
            dataframe = pd.DataFrame({'metrics': metrics, 'all': list_all, 'GBM': list_GBM, 'G4A': list_G4A})
            if epoch < 7:
                dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi-tiantan.xlsx', sheet_name='epoch' + str(epoch),index=False)
            else:
                with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi-tiantan.xlsx', mode='a',engine='openpyxl') as writer:
                    dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
            print('test-tiantan-patient in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_patient[0], list_all_patient[1], list_all_patient[2], list_all_patient[6]))
            print('test-tiantan-patient in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM_patient[1], list_GBM_patient[2], list_GBM_patient[6]))
            print('test-tiantan-patient in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A_patient[1], list_G4A_patient[2], list_G4A_patient[6]))
            dataframe = pd.DataFrame({'metrics': metrics, 'all': list_all_patient, 'GBM': list_GBM_patient, 'G4A': list_G4A_patient})
            if epoch < 7:
                dataframe.to_excel('./logs/' + opt['name'] + '/result-patient-tiantan.xlsx', sheet_name='epoch' + str(epoch), index=False)
            else:
                with pd.ExcelWriter('./logs/' + opt['name'] + '/result-patient-tiantan.xlsx', mode='a', engine='openpyxl') as writer:
                    dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
            ###############cambridge
            list_all, list_GBM, list_G4A, list_G3A, list_G2A, list_G3O, list_G2O, list_all_patient, list_GBM_patient \
                , list_G4A_patient, list_G3A_patient, list_G2A_patient, list_G3O_patient, list_G2O_patient = test_endtoend_Diag(
                opt, Mine_model_init, Mine_model_body, Mine_model_Cls, testDataset_cam, gpuID)
            print('test-cam in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (
            epoch, 70, list_all[0], list_all[1], list_all[2], list_all[6]))
            print('test-cam in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (
            epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
            print('test-cam in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (
            epoch, 70, list_G4A[1], list_G4A[2], list_G4A[6]))
            dataframe = pd.DataFrame({'metrics': metrics, 'all': list_all, 'GBM': list_GBM, 'G4A': list_G4A})
            if epoch < 7:
                dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi-cam.xlsx',
                                   sheet_name='epoch' + str(epoch), index=False)
            else:
                with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi-cam.xlsx', mode='a',
                                    engine='openpyxl') as writer:
                    dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
            print('test-cam-patient in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (
            epoch, 70, list_all_patient[0], list_all_patient[1], list_all_patient[2], list_all_patient[6]))
            print('test-cam-patient in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (
            epoch, 70, list_GBM_patient[1], list_GBM_patient[2], list_GBM_patient[6]))
            print('test-cam-patient in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (
            epoch, 70, list_G4A_patient[1], list_G4A_patient[2], list_G4A_patient[6]))
            dataframe = pd.DataFrame(
                {'metrics': metrics, 'all': list_all_patient, 'GBM': list_GBM_patient, 'G4A': list_G4A_patient})
            if epoch < 7:
                dataframe.to_excel('./logs/' + opt['name'] + '/result-patient-cam.xlsx',
                                   sheet_name='epoch' + str(epoch), index=False)
            else:
                with pd.ExcelWriter('./logs/' + opt['name'] + '/result-patient-cam.xlsx', mode='a',
                                    engine='openpyxl') as writer:
                    dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
    elif opt['Clstype'] == 'DiagSim':
        list_all, list_GBM, list_G4A, list_G23A, list_G23O,list_all_patient,list_GBM_patient\
            ,list_G4A_patient,list_G23A_patient,list_G23O_patient = test_endtoend_DiagSim(opt, Mine_model_init,Mine_model_body, Mine_model_Cls,dataloader, gpuID)
        print('test in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all[0], list_all[1], list_all[2], list_all[6]))
        print('test in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
        print('test in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A[1], list_G4A[2], list_G4A[6]))
        print('test in epoch: %d/%d, G23A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G23A[1], list_G23A[2], list_G23A[6]))
        print('test in epoch: %d/%d, G23O || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G23O[1], list_G23O[2], list_G23O[6]))
        dataframe = pd.DataFrame({'metrics': metrics, 'all': list_all, 'GBM': list_GBM, 'G4A': list_G4A, 'G3A': list_G23A,
             'G23O': list_G23O})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi.xlsx', sheet_name='epoch' + str(epoch), index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi.xlsx', mode='a', engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
        print('test-patient in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_patient[0], list_all_patient[1], list_all_patient[2], list_all_patient[6]))
        print('test-patient in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM_patient[1], list_GBM_patient[2], list_GBM_patient[6]))
        print('test-patient in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A_patient[1], list_G4A_patient[2], list_G4A_patient[6]))
        print('test-patient in epoch: %d/%d, G23A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G23A_patient[1], list_G23A_patient[2], list_G23A_patient[6]))
        print('test-patient in epoch: %d/%d, G23O || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G23O_patient[1], list_G23O_patient[2], list_G23O_patient[6]))
        dataframe = pd.DataFrame({'metrics': metrics, 'all': list_all_patient, 'GBM': list_GBM_patient, 'G4A': list_G4A_patient, 'G23A': list_G23A_patient, 'G23O': list_G23O_patient})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-patient.xlsx', sheet_name='epoch' + str(epoch), index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-patient.xlsx', mode='a', engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
        if opt['TrainingSet'] == 'TCGA':
            ###############CPTAC
            list_all, list_GBM, list_G4A, list_G23A, list_G23O,list_all_patient,list_GBM_patient\
            ,list_G4A_patient,list_G23A_patient,list_G23O_patient = test_endtoend_DiagSim(opt, Mine_model_init,Mine_model_body,Mine_model_Cls,testDataset_CPTAC, gpuID,external=True)
            print('test-CPTAC in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all[0], list_all[1], list_all[2], list_all[6]))
            print('test-CPTAC in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
            print('test-CPTAC in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A[1], list_G4A[2], list_G4A[6]))
            dataframe = pd.DataFrame({'metrics': metrics, 'all': list_all, 'GBM': list_GBM, 'G4A': list_G4A})
            if epoch < 7:
                dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi-CPTAC.xlsx', sheet_name='epoch' + str(epoch), index=False)
            else:
                with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi-CPTAC.xlsx', mode='a', engine='openpyxl') as writer:
                    dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
            print('test-CPTAC-patient in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_patient[0], list_all_patient[1], list_all_patient[2], list_all_patient[6]))
            print('test-CPTAC-patient in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM_patient[1], list_GBM_patient[2], list_GBM_patient[6]))
            print('test-CPTAC-patient in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A_patient[1], list_G4A_patient[2], list_G4A_patient[6]))
            dataframe = pd.DataFrame({'metrics': metrics, 'all': list_all_patient, 'GBM': list_GBM_patient, 'G4A': list_G4A_patient})
            if epoch < 7:
                dataframe.to_excel('./logs/' + opt['name'] + '/result-patient-CPTAC.xlsx', sheet_name='epoch' + str(epoch), index=False)
            else:
                with pd.ExcelWriter('./logs/' + opt['name'] + '/result-patient-CPTAC.xlsx', mode='a', engine='openpyxl') as writer:
                    dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
            ###############IvYGAP
            list_all, list_GBM, list_G4A, list_G23A, list_G23O,list_all_patient,list_GBM_patient\
            ,list_G4A_patient,list_G23A_patient,list_G23O_patient= test_endtoend_DiagSim(opt,Mine_model_init,Mine_model_body,Mine_model_Cls,testDataset_IvYGAP,gpuID,external=True)
            print('test-IvYGAP in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all[0], list_all[1], list_all[2], list_all[6]))
            print('test-IvYGAP in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
            print('test-IvYGAP in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A[1], list_G4A[2], list_G4A[6]))
            dataframe = pd.DataFrame({'metrics': metrics, 'all': list_all, 'GBM': list_GBM, 'G4A': list_G4A})
            if epoch < 7:
                dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi-IvYGAP.xlsx', sheet_name='epoch' + str(epoch),index=False)
            else:
                with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi-IvYGAP.xlsx', mode='a',engine='openpyxl') as writer:
                    dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
            print('test-IvYGAP-patient in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_patient[0], list_all_patient[1], list_all_patient[2], list_all_patient[6]))
            print('test-IvYGAP-patient in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM_patient[1], list_GBM_patient[2], list_GBM_patient[6]))
            print('test-IvYGAP-patient in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A_patient[1], list_G4A_patient[2], list_G4A_patient[6]))
            dataframe = pd.DataFrame({'metrics': metrics, 'all': list_all_patient, 'GBM': list_GBM_patient, 'G4A': list_G4A_patient})
            if epoch < 7:
                dataframe.to_excel('./logs/' + opt['name'] + '/result-patient-IvYGAP.xlsx', sheet_name='epoch' + str(epoch), index=False)
            else:
                with pd.ExcelWriter('./logs/' + opt['name'] + '/result-patient-IvYGAP.xlsx', mode='a', engine='openpyxl') as writer:
                    dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
            ###############GBMatch
            list_all, list_GBM, list_G4A, list_G23A, list_G23O,list_all_patient,list_GBM_patient\
            ,list_G4A_patient,list_G23A_patient,list_G23O_patient = test_endtoend_DiagSim(opt,Mine_model_init,Mine_model_body,Mine_model_Cls,testDataset_GBMatch,gpuID,external=True)
            print('test-GBMatch in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all[0], list_all[1], list_all[2], list_all[6]))
            print('test-GBMatch in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
            print('test-GBMatch in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A[1], list_G4A[2], list_G4A[6]))
            dataframe = pd.DataFrame({'metrics': metrics, 'all': list_all, 'GBM': list_GBM, 'G4A': list_G4A})
            if epoch < 7:
                dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi-GBMatch.xlsx', sheet_name='epoch' + str(epoch),index=False)
            else:
                with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi-GBMatch.xlsx', mode='a',engine='openpyxl') as writer:
                    dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
            print('test-GBMatch-patient in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_patient[0], list_all_patient[1], list_all_patient[2], list_all_patient[6]))
            print('test-GBMatch-patient in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM_patient[1], list_GBM_patient[2], list_GBM_patient[6]))
            print('test-GBMatch-patient in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A_patient[1], list_G4A_patient[2], list_G4A_patient[6]))
            dataframe = pd.DataFrame({'metrics': metrics, 'all': list_all_patient, 'GBM': list_GBM_patient, 'G4A': list_G4A_patient})
            if epoch < 7:
                dataframe.to_excel('./logs/' + opt['name'] + '/result-patient-GBMatch.xlsx', sheet_name='epoch' + str(epoch), index=False)
            else:
                with pd.ExcelWriter('./logs/' + opt['name'] + '/result-patient-GBMatch.xlsx', mode='a', engine='openpyxl') as writer:
                    dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
            ###############tiantan
            list_all, list_GBM, list_G4A, list_G23A, list_G23O,list_all_patient,list_GBM_patient\
            ,list_G4A_patient,list_G23A_patient,list_G23O_patient = test_endtoend_DiagSim(opt,Mine_model_init,Mine_model_body,Mine_model_Cls,testDataset_tiantan,gpuID)
            print('test-tiantan in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all[0], list_all[1], list_all[2], list_all[6]))
            print('test-tiantan in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
            print('test-tiantan in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A[1], list_G4A[2], list_G4A[6]))
            dataframe = pd.DataFrame({'metrics': metrics, 'all': list_all, 'GBM': list_GBM, 'G4A': list_G4A})
            if epoch < 7:
                dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi-tiantan.xlsx', sheet_name='epoch' + str(epoch),index=False)
            else:
                with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi-tiantan.xlsx', mode='a',engine='openpyxl') as writer:
                    dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
            print('test-tiantan-patient in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_patient[0], list_all_patient[1], list_all_patient[2], list_all_patient[6]))
            print('test-tiantan-patient in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM_patient[1], list_GBM_patient[2], list_GBM_patient[6]))
            print('test-tiantan-patient in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A_patient[1], list_G4A_patient[2], list_G4A_patient[6]))
            dataframe = pd.DataFrame({'metrics': metrics, 'all': list_all_patient, 'GBM': list_GBM_patient, 'G4A': list_G4A_patient})
            if epoch < 7:
                dataframe.to_excel('./logs/' + opt['name'] + '/result-patient-tiantan.xlsx', sheet_name='epoch' + str(epoch), index=False)
            else:
                with pd.ExcelWriter('./logs/' + opt['name'] + '/result-patient-tiantan.xlsx', mode='a', engine='openpyxl') as writer:
                    dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
            ###############cambridge
            list_all, list_GBM, list_G4A, list_G23A, list_G23O, list_all_patient, list_GBM_patient \
                , list_G4A_patient, list_G23A_patient, list_G23O_patient = test_endtoend_DiagSim(opt,
                                                                                                 Mine_model_init,
                                                                                                 Mine_model_body,
                                                                                                 Mine_model_Cls,
                                                                                                 testDataset_cam,
                                                                                                 gpuID)
            print('test-cam in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (
            epoch, 70, list_all[0], list_all[1], list_all[2], list_all[6]))
            print('test-cam in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (
            epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
            print('test-cam in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (
            epoch, 70, list_G4A[1], list_G4A[2], list_G4A[6]))
            dataframe = pd.DataFrame({'metrics': metrics, 'all': list_all, 'GBM': list_GBM, 'G4A': list_G4A})
            if epoch < 7:
                dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi-cam.xlsx',
                                   sheet_name='epoch' + str(epoch), index=False)
            else:
                with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi-cam.xlsx', mode='a',
                                    engine='openpyxl') as writer:
                    dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
            print('test-cam-patient in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (
            epoch, 70, list_all_patient[0], list_all_patient[1], list_all_patient[2], list_all_patient[6]))
            print('test-cam-patient in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (
            epoch, 70, list_GBM_patient[1], list_GBM_patient[2], list_GBM_patient[6]))
            print('test-cam-patient in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (
            epoch, 70, list_G4A_patient[1], list_G4A_patient[2], list_G4A_patient[6]))
            dataframe = pd.DataFrame(
                {'metrics': metrics, 'all': list_all_patient, 'GBM': list_GBM_patient, 'G4A': list_G4A_patient})
            if epoch < 7:
                dataframe.to_excel('./logs/' + opt['name'] + '/result-patient-cam.xlsx',
                                   sheet_name='epoch' + str(epoch), index=False)
            else:
                with pd.ExcelWriter('./logs/' + opt['name'] + '/result-patient-cam.xlsx', mode='a',
                                    engine='openpyxl') as writer:
                    dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)


def test_endtoend_stem_marker(opt,Mine_model_init,Mine_model_body,Mine_model_Cls, dataloader, gpuID,epoch):
    metrics = ['acc', 'sen', 'spec', 'precision', 'recall', 'f1', 'AUC']
    list = test_marker_ViT(opt, Mine_model_init, Mine_model_body, Mine_model_Cls, dataloader, gpuID)
    print('test in epoch: %d/%d, MGMT || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list[0], list[1], list[2], list[6]))

    dataframe = pd.DataFrame(
        {'metrics': metrics, 'marker': list})
    if epoch < 7:
        dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi.xlsx', sheet_name='epoch' + str(epoch),
                           index=False)
    else:
        with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi.xlsx', mode='a', engine='openpyxl') as writer:
            dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)


def test_stage1_stem(opt,Mine_model_init,Mine_model_His,Mine_model_Cls,dataloader, gpuID,epoch,testDataset_CPTAC,testDataset_IvYGAP,testDataset_GBMatch,testDataset_tiantan,testDataset_cam):
    metrics = ['acc', 'sen', 'spec', 'precision', 'recall', 'f1', 'AUC']
    list_his, list_A, list_O, list_GBM, list_grade, list_G2, \
    list_G3, list_G4 = test_stage1(opt, Mine_model_init, Mine_model_His, Mine_model_Cls, dataloader, gpuID,epoch)
    print('test in epoch: %d/%d, Subtype || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (
    epoch, 70, list_his[0], list_his[1], list_his[2], list_his[6]))
    print('test in epoch: %d/%d, A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_A[1], list_A[2], list_A[6]))
    print('test in epoch: %d/%d, O || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_O[1], list_O[2], list_O[6]))
    print('test in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (
    epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
    print('test in epoch: %d/%d, Grade || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (
    epoch, 70, list_grade[0], list_grade[1], list_grade[2], list_grade[6]))
    print('test in epoch: %d/%d, G2 || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G2[1], list_G2[2], list_G2[6]))
    print('test in epoch: %d/%d, G3 || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G3[1], list_G3[2], list_G3[6]))
    print('test in epoch: %d/%d, G4 || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4[1], list_G4[2], list_G4[6]))
    dataframe = pd.DataFrame(
        {'metrics': metrics, 'Subtype': list_his, 'A': list_A, 'O': list_O, 'GBM': list_GBM, 'Grade': list_grade,
         'G2': list_G2, 'G3': list_G3, 'G4': list_G4})
    if epoch < 7:
        dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi.xlsx', sheet_name='epoch' + str(epoch),
                           index=False)
    else:
        with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi.xlsx', mode='a', engine='openpyxl') as writer:
            dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)

    if opt['TrainingSet'] == 'TCGA':
        ###############CPTAC
        list_his, list_A, list_O, list_GBM, list_grade, list_G2, \
        list_G3, list_G4 = test_stage1(opt, Mine_model_init, Mine_model_His, Mine_model_Cls, testDataset_CPTAC,
                                           gpuID,epoch, external=True)
        print('test-CPTAC in epoch: %d/%d, Subtype || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_his[0], list_his[1], list_his[2], list_his[6]))
        print('test-CPTAC in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
        print('test-CPTAC in epoch: %d/%d, Grade || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_grade[0],list_grade[1], list_grade[2], list_grade[6]))
        print('test-CPTAC in epoch: %d/%d, G4 || sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_G4[1], list_G4[2], list_G4[6]))
        dataframe = pd.DataFrame(
            {'metrics': metrics, 'Subtype': list_his, 'GBM': list_GBM, 'Grade': list_grade, 'G4': list_G4})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi-CPTAC.xlsx', sheet_name='epoch' + str(epoch),
                               index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi-CPTAC.xlsx', mode='a',
                                engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)

        ###############GBMatch
        list_his, list_A, list_O, list_GBM, list_grade, list_G2, \
        list_G3, list_G4 = test_stage1(opt, Mine_model_init, Mine_model_His, Mine_model_Cls, testDataset_GBMatch,
                                           gpuID,epoch, external=True)
        print('test-GBMatch in epoch: %d/%d, Subtype || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_his[0], list_his[1], list_his[2], list_his[6]))
        print('test-GBMatch in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
        print('test-GBMatch in epoch: %d/%d, Grade || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_grade[0],list_grade[1], list_grade[2], list_grade[6]))
        print('test-GBMatch in epoch: %d/%d, G4 || sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_G4[1], list_G4[2], list_G4[6]))
        dataframe = pd.DataFrame(
            {'metrics': metrics, 'Subtype': list_his, 'GBM': list_GBM, 'Grade': list_grade, 'G4': list_G4})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi-GBMatch.xlsx',
                               sheet_name='epoch' + str(epoch), index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi-GBMatch.xlsx', mode='a',
                                engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)

        ###############IvYGAP
        list_his, list_A, list_O, list_GBM, list_grade, list_G2, \
        list_G3, list_G4 = test_stage1(opt, Mine_model_init, Mine_model_His, Mine_model_Cls, testDataset_IvYGAP,
                                           gpuID,epoch, external=True)
        print('test-IvYGAP in epoch: %d/%d, Subtype || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_his[0], list_his[1], list_his[2], list_his[6]))
        print('test-IvYGAP in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
        print('test-IvYGAP in epoch: %d/%d, Grade || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_grade[0],list_grade[1], list_grade[2], list_grade[6]))
        print('test-IvYGAP in epoch: %d/%d, G4 || sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_G4[1], list_G4[2], list_G4[6]))
        dataframe = pd.DataFrame(
            {'metrics': metrics, 'Subtype': list_his, 'GBM': list_GBM, 'Grade': list_grade, 'G4': list_G4})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi-IvYGAP.xlsx', sheet_name='epoch' + str(epoch),
                               index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi-IvYGAP.xlsx', mode='a',
                                engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)

        ###############tiantan
        list_his, list_A, list_O, list_GBM, list_grade, list_G2, \
            list_G3, list_G4 = test_stage1(opt, Mine_model_init, Mine_model_His, Mine_model_Cls, testDataset_tiantan,
                                           gpuID, epoch)
        print('test-tiantan in epoch: %d/%d, Subtype || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_his[0], list_his[1], list_his[2], list_his[6]))
        print('test-tiantan in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
        print('test-tiantan in epoch: %d/%d, Grade || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_grade[0], list_grade[1], list_grade[2], list_grade[6]))
        print('test-tiantan in epoch: %d/%d, G4 || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4[1], list_G4[2], list_G4[6]))
        dataframe = pd.DataFrame(
            {'metrics': metrics, 'Subtype': list_his, 'GBM': list_GBM, 'Grade': list_grade, 'G4': list_G4})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi-tiantan.xlsx', sheet_name='epoch' + str(epoch),
                               index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi-tiantan.xlsx', mode='a',
                                engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
        ###############cambridge
        list_his, list_A, list_O, list_GBM, list_grade, list_G2, \
        list_G3, list_G4 = test_stage1(opt, Mine_model_init, Mine_model_His, Mine_model_Cls,
                                       testDataset_cam,
                                       gpuID, epoch)
        print('test-cam in epoch: %d/%d, Subtype || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_his[0], list_his[1], list_his[2], list_his[6]))
        print('test-cam in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
        print('test-cam in epoch: %d/%d, Grade || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_grade[0], list_grade[1], list_grade[2], list_grade[6]))
        print('test-cam in epoch: %d/%d, G4 || sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_G4[1], list_G4[2], list_G4[6]))
        dataframe = pd.DataFrame(
            {'metrics': metrics, 'Subtype': list_his, 'GBM': list_GBM, 'Grade': list_grade, 'G4': list_G4})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi-cam.xlsx',
                               sheet_name='epoch' + str(epoch),
                               index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi-cam.xlsx', mode='a',
                                engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)


def test_pre_stem(opt,Mine_model_init,Mine_model_His,Mine_model_Cls, trainLoader,dataloader, gpuID,epoch,testDataset_CPTAC,testDataset_IvYGAP,testDataset_GBMatch,testDataset_tiantan,testDataset_cam):
    metrics = ['acc', 'sen', 'spec', 'precision', 'recall', 'f1', 'AUC']
    list_his, list_A, list_O, list_GBM, list_grade, list_G2,\
    list_G3, list_G4 = test_stage1_pre(opt, Mine_model_init,Mine_model_His,Mine_model_Cls,dataloader, gpuID,trainLoader=trainLoader)
    print('test in epoch: %d/%d, Subtype || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_his[0], list_his[1], list_his[2], list_his[6]))
    print('test in epoch: %d/%d, A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_A[1], list_A[2], list_A[6]))
    print('test in epoch: %d/%d, O || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_O[1], list_O[2], list_O[6]))
    print('test in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
    print('test in epoch: %d/%d, Grade || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_grade[0],list_grade[1], list_grade[2], list_grade[6]))
    print('test in epoch: %d/%d, G2 || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G2[1], list_G2[2], list_G2[6]))
    print('test in epoch: %d/%d, G3 || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G3[1], list_G3[2], list_G3[6]))
    print('test in epoch: %d/%d, G4 || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4[1], list_G4[2], list_G4[6]))
    dataframe = pd.DataFrame({'metrics': metrics, 'Subtype': list_his, 'A': list_A, 'O': list_O, 'GBM': list_GBM, 'Grade': list_grade,'G2': list_G2, 'G3': list_G3, 'G4': list_G4})
    if epoch < 7:
        dataframe.to_excel('./logs/' + opt['name'] + '/Pre-result-wsi.xlsx', sheet_name='epoch' + str(epoch), index=False)
    else:
        with pd.ExcelWriter('./logs/' + opt['name'] + '/Pre-result-wsi.xlsx', mode='a', engine='openpyxl') as writer:
            dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)

    if opt['TrainingSet'] == 'TCGA':
        ###############CPTAC
        list_his, list_A, list_O, list_GBM, list_grade, list_G2, \
        list_G3, list_G4 = test_stage1_pre(opt, Mine_model_init,Mine_model_His,Mine_model_Cls,testDataset_CPTAC, gpuID,external=True)
        print('test-CPTAC in epoch: %d/%d, Subtype || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_his[0], list_his[1], list_his[2], list_his[6]))
        print('test-CPTAC in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
        print('test-CPTAC in epoch: %d/%d, Grade || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_grade[0],list_grade[1], list_grade[2], list_grade[6]))
        print('test-CPTAC in epoch: %d/%d, G4 || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4[1], list_G4[2], list_G4[6]))
        dataframe = pd.DataFrame({'metrics': metrics, 'Subtype': list_his, 'GBM': list_GBM, 'Grade': list_grade, 'G4': list_G4})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/Pre-result-wsi-CPTAC.xlsx', sheet_name='epoch' + str(epoch), index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/Pre-result-wsi-CPTAC.xlsx', mode='a', engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)

        ###############GBMatch
        list_his, list_A, list_O, list_GBM, list_grade, list_G2, \
        list_G3, list_G4 = test_stage1_pre(opt, Mine_model_init,Mine_model_His,Mine_model_Cls,testDataset_GBMatch, gpuID,external=True)
        print('test-GBMatch in epoch: %d/%d, Subtype || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_his[0], list_his[1], list_his[2], list_his[6]))
        print('test-GBMatch in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
        print('test-GBMatch in epoch: %d/%d, Grade || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_grade[0],list_grade[1], list_grade[2], list_grade[6]))
        print('test-GBMatch in epoch: %d/%d, G4 || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4[1], list_G4[2], list_G4[6]))
        dataframe = pd.DataFrame({'metrics': metrics, 'Subtype': list_his, 'GBM': list_GBM, 'Grade': list_grade, 'G4': list_G4})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/Pre-result-wsi-GBMatch.xlsx', sheet_name='epoch' + str(epoch), index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/Pre-result-wsi-GBMatch.xlsx', mode='a', engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)

        ###############IvYGAP
        list_his, list_A, list_O, list_GBM, list_grade, list_G2, \
        list_G3, list_G4 = test_stage1_pre(opt, Mine_model_init,Mine_model_His,Mine_model_Cls,testDataset_IvYGAP, gpuID,external=True)
        print('test-IvYGAP in epoch: %d/%d, Subtype || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_his[0], list_his[1], list_his[2], list_his[6]))
        print('test-IvYGAP in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
        print('test-IvYGAP in epoch: %d/%d, Grade || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_grade[0],list_grade[1], list_grade[2], list_grade[6]))
        print('test-IvYGAP in epoch: %d/%d, G4 || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4[1], list_G4[2], list_G4[6]))
        dataframe = pd.DataFrame({'metrics': metrics, 'Subtype': list_his, 'GBM': list_GBM, 'Grade': list_grade, 'G4': list_G4})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/Pre-result-wsi-IvYGAP.xlsx', sheet_name='epoch' + str(epoch), index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/Pre-result-wsi-IvYGAP.xlsx', mode='a', engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)

        # ###############tiantan
        list_his, list_A, list_O, list_GBM, list_grade, list_G2, list_G3, list_G4 = test_stage1_pre(opt, Mine_model_init, Mine_model_His, Mine_model_Cls,testDataset_tiantan, gpuID)
        print('test-tiantan in epoch: %d/%d, Subtype || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_his[0], list_his[1], list_his[2], list_his[6]))
        print('test-tiantan in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
        print('test-tiantan in epoch: %d/%d, Grade || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_grade[0], list_grade[1], list_grade[2], list_grade[6]))
        print('test-tiantan in epoch: %d/%d, G4 || sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_G4[1], list_G4[2], list_G4[6]))
        dataframe = pd.DataFrame(
            {'metrics': metrics, 'Subtype': list_his, 'GBM': list_GBM, 'Grade': list_grade, 'G4': list_G4})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/Pre-result-wsi-tiantan.xlsx',
                               sheet_name='epoch' + str(epoch), index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/Pre-result-wsi-tiantan.xlsx', mode='a',
                                engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)


        # ############### cambriddge
        list_his, list_A, list_O, list_GBM, list_grade, list_G2, list_G3, list_G4 = test_stage1_pre(opt, Mine_model_init, Mine_model_His, Mine_model_Cls,testDataset_cam, gpuID)
        print('test-cam in epoch: %d/%d, Subtype || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_his[0], list_his[1], list_his[2], list_his[6]))
        print('test-cam in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
        print('test-cam in epoch: %d/%d, Grade || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_grade[0], list_grade[1], list_grade[2], list_grade[6]))
        print('test-cam in epoch: %d/%d, G4 || sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_G4[1], list_G4[2], list_G4[6]))
        dataframe = pd.DataFrame(
            {'metrics': metrics, 'Subtype': list_his, 'GBM': list_GBM, 'Grade': list_grade, 'G4': list_G4})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/Pre-result-wsi-cam.xlsx',
                               sheet_name='epoch' + str(epoch), index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/Pre-result-wsi-cam.xlsx', mode='a',
                                engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)

def test_stage2_stem(opt,Mine_model_init,Mine_model_body,Mine_model_Cls,Mine_model_molecular, Mine_model_Graph, dataloader, gpuID,epoch,testDataset_CPTAC,testDataset_IvYGAP,testDataset_GBMatch,testDataset_tiantan,testDataset_cam):
    metrics = ['acc', 'sen', 'spec', 'precision', 'recall', 'f1', 'AUC']
    list_all_Diag, list_GBM, list_G4A, list_G3A, list_G2A, list_G3O, list_G2O,\
    list_all_DiagSim, list_GBMSim, list_G4ASim,list_G23ASim,list_G23OSim, \
            list_all_Diag_patient, list_GBM_patient,list_G4A_patient, list_G3A_patient, list_G2A_patient, list_G3O_patient, list_G2O_patient, \
              list_all_DiagSim_patient, list_GBMSim_patient,\
         list_G4ASim_patient,list_G23ASim_patient,list_G23OSim_patient,list_IDH,list_1p19q,list_CDKN,list_IDH_patient,list_1p19q_patient,list_CDKN_patient\
        = test_stage2(opt, Mine_model_init, Mine_model_body, Mine_model_Cls,Mine_model_molecular, Mine_model_Graph, dataloader, gpuID)
    print('test in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_Diag[0], list_all_Diag[1], list_all_Diag[2], list_all_Diag[6]))
    print('test in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
    print('test in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A[1], list_G4A[2], list_G4A[6]))
    print('test in epoch: %d/%d, G3A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G3A[1], list_G3A[2], list_G3A[6]))
    print('test in epoch: %d/%d, G2A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G2A[1], list_G2A[2], list_G2A[6]))
    print('test in epoch: %d/%d, G3O || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G3O[1], list_G3O[2], list_G3O[6]))
    print('test in epoch: %d/%d, G2O || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G2O[1], list_G2O[2], list_G2O[6]))
    print('test in epoch: %d/%d, DiagSim || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_DiagSim[0], list_all_DiagSim[1], list_all_DiagSim[2], list_all_DiagSim[6]))
    print('test in epoch: %d/%d, GBMSim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBMSim[1], list_GBMSim[2], list_GBMSim[6]))
    print('test in epoch: %d/%d, G4ASim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4ASim[1], list_G4ASim[2], list_G4ASim[6]))
    print('test in epoch: %d/%d, G23ASim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G23ASim[1], list_G23ASim[2], list_G23ASim[6]))
    print('test in epoch: %d/%d, G23OSim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G23OSim[1], list_G23OSim[2], list_G23OSim[6]))
    print('test in epoch: %d/%d, IDH || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_IDH[0], list_IDH[1], list_IDH[2], list_IDH[6]))
    print('test in epoch: %d/%d, 1p19q || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_1p19q[0], list_1p19q[1], list_1p19q[2], list_1p19q[6]))
    print('test in epoch: %d/%d, CDKN || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_CDKN[0], list_CDKN[1], list_CDKN[2], list_CDKN[6]))

    dataframe = pd.DataFrame(
        {'metrics': metrics, 'Diag': list_all_Diag, 'GBM': list_GBM, 'G4A': list_G4A, 'G3A': list_G3A, 'G2A': list_G2A,
         'G3O': list_G3O, 'G2O': list_G2O,'DiagSim': list_all_DiagSim, 'GBMSim': list_GBMSim, 'G4ASim': list_G4ASim, 'G23ASim': list_G23ASim, 'G23OSim': list_G23OSim,
         'IDH': list_IDH, '1p19q': list_1p19q, 'CDKN': list_CDKN})
    if epoch < 7:
        dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi.xlsx', sheet_name='epoch' + str(epoch),index=False)
    else:
        with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi.xlsx', mode='a', engine='openpyxl') as writer:
            dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)

    print('test-patient in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_Diag_patient[0], list_all_Diag_patient[1], list_all_Diag_patient[2], list_all_Diag_patient[6]))
    print('test-patient in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM_patient[1], list_GBM_patient[2], list_GBM_patient[6]))
    print('test-patient in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A_patient[1], list_G4A_patient[2], list_G4A_patient[6]))
    print('test-patient in epoch: %d/%d, G3A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G3A_patient[1], list_G3A_patient[2], list_G3A_patient[6]))
    print('test-patient in epoch: %d/%d, G2A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G2A_patient[1], list_G2A_patient[2], list_G2A_patient[6]))
    print('test-patient in epoch: %d/%d, G3O || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G3O_patient[1], list_G3O_patient[2], list_G3O_patient[6]))
    print('test-patient in epoch: %d/%d, G2O || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G2O_patient[1], list_G2O_patient[2], list_G2O_patient[6]))
    print('test-patient in epoch: %d/%d, DiagSim || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_DiagSim_patient[0], list_all_DiagSim_patient[1], list_all_DiagSim_patient[2], list_all_DiagSim_patient[6]))
    print('test-patient in epoch: %d/%d, GBMSim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBMSim_patient[1], list_GBMSim_patient[2], list_GBMSim_patient[6]))
    print('test-patient in epoch: %d/%d, G4ASim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4ASim_patient[1], list_G4ASim_patient[2], list_G4ASim_patient[6]))
    print('test-patient in epoch: %d/%d, G23ASim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G23ASim_patient[1], list_G23ASim_patient[2], list_G23ASim_patient[6]))
    print('test-patient in epoch: %d/%d, G23OSim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G23OSim_patient[1], list_G23OSim_patient[2], list_G23OSim_patient[6]))
    print('test-patient in epoch: %d/%d, IDH || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_IDH_patient[0], list_IDH_patient[1], list_IDH_patient[2], list_IDH_patient[6]))
    print('test-patient in epoch: %d/%d, 1p19q || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_1p19q_patient[0], list_1p19q_patient[1], list_1p19q_patient[2], list_1p19q_patient[6]))
    print('test-patient in epoch: %d/%d, CDKN || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_CDKN_patient[0], list_CDKN_patient[1], list_CDKN_patient[2], list_CDKN_patient[6]))

    dataframe = pd.DataFrame(
        {'metrics': metrics, 'Diag': list_all_Diag_patient, 'GBM': list_GBM_patient, 'G4A': list_G4A_patient, 'G3A': list_G3A_patient, 'G2A': list_G2A_patient,
         'G3O': list_G3O_patient, 'G2O': list_G2O_patient,'DiagSim': list_all_DiagSim_patient, 'GBMSim': list_GBMSim_patient, 'G4ASim': list_G4ASim_patient, 'G23ASim': list_G23ASim_patient, 'G23OSim': list_G23OSim_patient,
         'IDH': list_IDH_patient, '1p19q': list_1p19q_patient, 'CDKN': list_CDKN_patient})
    if epoch < 7:
        dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi_patient.xlsx', sheet_name='epoch' + str(epoch),index=False)
    else:
        with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi_patient.xlsx', mode='a', engine='openpyxl') as writer:
            dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)

    if opt['TrainingSet'] == 'TCGA':
        ###############CPTAC
        list_all_Diag, list_GBM, list_G4A, list_G3A, list_G2A, list_G3O, list_G2O, \
        list_all_DiagSim, list_GBMSim, list_G4ASim, list_G23ASim, list_G23OSim, \
        list_all_Diag_patient, list_GBM_patient, list_G4A_patient, list_G3A_patient, list_G2A_patient, list_G3O_patient, list_G2O_patient, \
        list_all_DiagSim_patient, list_GBMSim_patient, \
        list_G4ASim_patient, list_G23ASim_patient, list_G23OSim_patient, list_IDH, list_1p19q, list_CDKN, list_IDH_patient, list_1p19q_patient, list_CDKN_patient \
            = test_stage2(opt, Mine_model_init, Mine_model_body, Mine_model_Cls, Mine_model_molecular,
                                 Mine_model_Graph, testDataset_CPTAC, gpuID,external=True)

        print('test-CPTAC in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_Diag[0], list_all_Diag[1], list_all_Diag[2], list_all_Diag[6]))
        print('test-CPTAC in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
        print('test-CPTAC in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A[1], list_G4A[2], list_G4A[6]))
        print('test-CPTAC in epoch: %d/%d, DiagSim || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_DiagSim[0], list_all_DiagSim[1], list_all_DiagSim[2], list_all_DiagSim[6]))
        print('test-CPTAC in epoch: %d/%d, GBMSim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBMSim[1], list_GBMSim[2], list_GBMSim[6]))
        print('test-CPTAC in epoch: %d/%d, G4ASim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4ASim[1], list_G4ASim[2], list_G4ASim[6]))

        dataframe = pd.DataFrame(
            {'metrics': metrics, 'Diag': list_all_Diag, 'GBM': list_GBM, 'G4A': list_G4A,
             'DiagSim': list_all_DiagSim, 'GBMSim': list_GBMSim, 'G4ASim': list_G4ASim})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi-CPTAC.xlsx', sheet_name='epoch' + str(epoch),index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi-CPTAC.xlsx', mode='a', engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
        print('test-patient-CPTAC in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_Diag_patient[0], list_all_Diag_patient[1], list_all_Diag_patient[2], list_all_Diag_patient[6]))
        print('test-patient-CPTAC in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM_patient[1], list_GBM_patient[2], list_GBM_patient[6]))
        print('test-patient-CPTAC in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A_patient[1], list_G4A_patient[2], list_G4A_patient[6]))
        print('test-patient-CPTAC in epoch: %d/%d, DiagSim || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_DiagSim_patient[0], list_all_DiagSim_patient[1], list_all_DiagSim_patient[2], list_all_DiagSim_patient[6]))
        print('test-patient-CPTAC in epoch: %d/%d, GBMSim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBMSim_patient[1], list_GBMSim_patient[2], list_GBMSim_patient[6]))
        print('test-patient-CPTAC in epoch: %d/%d, G4ASim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4ASim_patient[1], list_G4ASim_patient[2], list_G4ASim_patient[6]))

        dataframe = pd.DataFrame(
            {'metrics': metrics, 'Diag': list_all_Diag_patient, 'GBM': list_GBM_patient,'G4A': list_G4A_patient,
             'DiagSim': list_all_DiagSim_patient, 'GBMSim': list_GBMSim_patient, 'G4ASim': list_G4ASim_patient})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi_patient-CPTAC.xlsx', sheet_name='epoch' + str(epoch),index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi_patient-CPTAC.xlsx', mode='a', engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
        ###############GBMatch
        list_all_Diag, list_GBM, list_G4A, list_G3A, list_G2A, list_G3O, list_G2O, \
        list_all_DiagSim, list_GBMSim, list_G4ASim, list_G23ASim, list_G23OSim, \
        list_all_Diag_patient, list_GBM_patient, list_G4A_patient, list_G3A_patient, list_G2A_patient, list_G3O_patient, list_G2O_patient, \
        list_all_DiagSim_patient, list_GBMSim_patient, \
        list_G4ASim_patient, list_G23ASim_patient, list_G23OSim_patient, list_IDH, list_1p19q, list_CDKN, list_IDH_patient, list_1p19q_patient, list_CDKN_patient \
            = test_stage2(opt, Mine_model_init, Mine_model_body, Mine_model_Cls, Mine_model_molecular,
                                 Mine_model_Graph, testDataset_GBMatch, gpuID,external=True)

        print('test-GBMatch in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_Diag[0], list_all_Diag[1], list_all_Diag[2], list_all_Diag[6]))
        print('test-GBMatch in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
        print('test-GBMatch in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A[1], list_G4A[2], list_G4A[6]))
        print('test-GBMatch in epoch: %d/%d, DiagSim || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_DiagSim[0], list_all_DiagSim[1], list_all_DiagSim[2], list_all_DiagSim[6]))
        print('test-GBMatch in epoch: %d/%d, GBMSim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBMSim[1], list_GBMSim[2], list_GBMSim[6]))
        print('test-GBMatch in epoch: %d/%d, G4ASim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4ASim[1], list_G4ASim[2], list_G4ASim[6]))

        dataframe = pd.DataFrame(
            {'metrics': metrics, 'Diag': list_all_Diag, 'GBM': list_GBM, 'G4A': list_G4A,
             'DiagSim': list_all_DiagSim, 'GBMSim': list_GBMSim, 'G4ASim': list_G4ASim})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi-GBMatch.xlsx', sheet_name='epoch' + str(epoch),index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi-GBMatch.xlsx', mode='a', engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
        print('test-patient-GBMatch in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_Diag_patient[0], list_all_Diag_patient[1], list_all_Diag_patient[2], list_all_Diag_patient[6]))
        print('test-patient-GBMatch in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM_patient[1], list_GBM_patient[2], list_GBM_patient[6]))
        print('test-patient-GBMatch in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A_patient[1], list_G4A_patient[2], list_G4A_patient[6]))
        print('test-patient-GBMatch in epoch: %d/%d, DiagSim || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_DiagSim_patient[0], list_all_DiagSim_patient[1], list_all_DiagSim_patient[2], list_all_DiagSim_patient[6]))
        print('test-patient-GBMatch in epoch: %d/%d, GBMSim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBMSim_patient[1], list_GBMSim_patient[2], list_GBMSim_patient[6]))
        print('test-patient-GBMatch in epoch: %d/%d, G4ASim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4ASim_patient[1], list_G4ASim_patient[2], list_G4ASim_patient[6]))

        dataframe = pd.DataFrame(
            {'metrics': metrics, 'Diag': list_all_Diag_patient, 'GBM': list_GBM_patient,'G4A': list_G4A_patient,
             'DiagSim': list_all_DiagSim_patient, 'GBMSim': list_GBMSim_patient, 'G4ASim': list_G4ASim_patient})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi_patient-GBMatch.xlsx', sheet_name='epoch' + str(epoch),index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi_patient-GBMatch.xlsx', mode='a', engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
        ###############IvYGAP
        list_all_Diag, list_GBM, list_G4A, list_G3A, list_G2A, list_G3O, list_G2O, \
        list_all_DiagSim, list_GBMSim, list_G4ASim, list_G23ASim, list_G23OSim, \
        list_all_Diag_patient, list_GBM_patient, list_G4A_patient, list_G3A_patient, list_G2A_patient, list_G3O_patient, list_G2O_patient, \
        list_all_DiagSim_patient, list_GBMSim_patient, \
        list_G4ASim_patient, list_G23ASim_patient, list_G23OSim_patient, list_IDH, list_1p19q, list_CDKN, list_IDH_patient, list_1p19q_patient, list_CDKN_patient \
            = test_stage2(opt, Mine_model_init, Mine_model_body, Mine_model_Cls, Mine_model_molecular,
                               Mine_model_Graph, testDataset_IvYGAP, gpuID, external=True)

        print('test-IvYGAP in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_all_Diag[0], list_all_Diag[1], list_all_Diag[2], list_all_Diag[6]))
        print('test-IvYGAP in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
        print('test-IvYGAP in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_G4A[1], list_G4A[2], list_G4A[6]))
        print('test-IvYGAP in epoch: %d/%d, DiagSim || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_all_DiagSim[0], list_all_DiagSim[1], list_all_DiagSim[2], list_all_DiagSim[6]))
        print('test-IvYGAP in epoch: %d/%d, GBMSim || sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_GBMSim[1], list_GBMSim[2], list_GBMSim[6]))
        print('test-IvYGAP in epoch: %d/%d, G4ASim || sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_G4ASim[1], list_G4ASim[2], list_G4ASim[6]))

        dataframe = pd.DataFrame(
            {'metrics': metrics, 'Diag': list_all_Diag, 'GBM': list_GBM, 'G4A': list_G4A,
             'DiagSim': list_all_DiagSim, 'GBMSim': list_GBMSim, 'G4ASim': list_G4ASim})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi-IvYGAP.xlsx',
                               sheet_name='epoch' + str(epoch), index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi-IvYGAP.xlsx', mode='a',
                                engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
        print('test-patient-IvYGAP in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_all_Diag_patient[0], list_all_Diag_patient[1], list_all_Diag_patient[2],
        list_all_Diag_patient[6]))
        print('test-patient-IvYGAP in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_GBM_patient[1], list_GBM_patient[2], list_GBM_patient[6]))
        print('test-patient-IvYGAP in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_G4A_patient[1], list_G4A_patient[2], list_G4A_patient[6]))
        print('test-patient-IvYGAP in epoch: %d/%d, DiagSim || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_all_DiagSim_patient[0], list_all_DiagSim_patient[1], list_all_DiagSim_patient[2],
        list_all_DiagSim_patient[6]))
        print('test-patient-IvYGAP in epoch: %d/%d, GBMSim || sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_GBMSim_patient[1], list_GBMSim_patient[2], list_GBMSim_patient[6]))
        print('test-patient-IvYGAP in epoch: %d/%d, G4ASim || sen:%.3f,spec:%.3f, auc:%.3f' % (
        epoch, 70, list_G4ASim_patient[1], list_G4ASim_patient[2], list_G4ASim_patient[6]))

        dataframe = pd.DataFrame(
            {'metrics': metrics, 'Diag': list_all_Diag_patient, 'GBM': list_GBM_patient,
             'G4A': list_G4A_patient,
             'DiagSim': list_all_DiagSim_patient, 'GBMSim': list_GBMSim_patient, 'G4ASim': list_G4ASim_patient})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi_patient-IvYGAP.xlsx',
                               sheet_name='epoch' + str(epoch), index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi_patient-IvYGAP.xlsx', mode='a',
                                engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
        ###############tiantan
        list_all_Diag, list_GBM, list_G4A, list_G3A, list_G2A, list_G3O, list_G2O, \
            list_all_DiagSim, list_GBMSim, list_G4ASim, list_G23ASim, list_G23OSim, \
            list_all_Diag_patient, list_GBM_patient, list_G4A_patient, list_G3A_patient, list_G2A_patient, list_G3O_patient, list_G2O_patient, \
            list_all_DiagSim_patient, list_GBMSim_patient, \
            list_G4ASim_patient, list_G23ASim_patient, list_G23OSim_patient, list_IDH, list_1p19q, list_CDKN, list_IDH_patient, list_1p19q_patient, list_CDKN_patient \
            = test_stage2(opt, Mine_model_init, Mine_model_body, Mine_model_Cls, Mine_model_molecular,Mine_model_Graph, testDataset_tiantan, gpuID,flag_tiantan=True)

        print('test-tiantan in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_Diag[0], list_all_Diag[1], list_all_Diag[2], list_all_Diag[6]))
        print('test-tiantan in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
        print('test-tiantan in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A[1], list_G4A[2], list_G4A[6]))
        print('test-tiantan in epoch: %d/%d, G3O || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G3O[1], list_G3O[2], list_G3O[6]))
        print('test-tiantan in epoch: %d/%d, G2O || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G2O[1], list_G2O[2], list_G2O[6]))
        print('test-tiantan in epoch: %d/%d, DiagSim || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_DiagSim[0], list_all_DiagSim[1], list_all_DiagSim[2], list_all_DiagSim[6]))
        print('test-tiantan in epoch: %d/%d, GBMSim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBMSim[1], list_GBMSim[2], list_GBMSim[6]))
        print('test-tiantan in epoch: %d/%d, G4ASim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4ASim[1], list_G4ASim[2], list_G4ASim[6]))
        print('test-tiantan in epoch: %d/%d, G23OSim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G23OSim[1], list_G23OSim[2], list_G23OSim[6]))

        dataframe = pd.DataFrame(
            {'metrics': metrics, 'Diag': list_all_Diag, 'GBM': list_GBM, 'G4A': list_G4A,
             'G3O': list_G3O, 'G2O': list_G2O, 'DiagSim': list_all_DiagSim, 'GBMSim': list_GBMSim,
             'G4ASim': list_G4ASim, 'G23OSim': list_G23OSim})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi-tiantan.xlsx',sheet_name='epoch' + str(epoch), index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi-tiantan.xlsx', mode='a',engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)

        print('test-tiantan-patient in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_Diag_patient[0], list_all_Diag_patient[1], list_all_Diag_patient[2], list_all_Diag_patient[6]))
        print('test-tiantan-patient in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM_patient[1], list_GBM_patient[2], list_GBM_patient[6]))
        print('test-tiantan-patient in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A_patient[1], list_G4A_patient[2], list_G4A_patient[6]))
        print('test-tiantan-patient in epoch: %d/%d, G3O || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G3O_patient[1], list_G3O_patient[2], list_G3O_patient[6]))
        print('test-tiantan-patient in epoch: %d/%d, G2O || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G2O_patient[1], list_G2O_patient[2], list_G2O_patient[6]))
        print('test-tiantan-patient in epoch: %d/%d, DiagSim || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_DiagSim_patient[0], list_all_DiagSim_patient[1], list_all_DiagSim_patient[2], list_all_DiagSim_patient[6]))
        print('test-tiantan-patient in epoch: %d/%d, GBMSim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBMSim_patient[1], list_GBMSim_patient[2], list_GBMSim_patient[6]))
        print('test-tiantan-patient in epoch: %d/%d, G4ASim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4ASim_patient[1], list_G4ASim_patient[2], list_G4ASim_patient[6]))
        print('test-tiantan-patient in epoch: %d/%d, G23OSim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G23OSim_patient[1], list_G23OSim_patient[2], list_G23OSim_patient[6]))

        dataframe = pd.DataFrame(
            {'metrics': metrics, 'Diag': list_all_Diag_patient, 'GBM': list_GBM_patient, 'G4A': list_G4A_patient,
             'G3O': list_G3O_patient, 'G2O': list_G2O_patient, 'DiagSim': list_all_DiagSim_patient,
             'GBMSim': list_GBMSim_patient, 'G4ASim': list_G4ASim_patient,'G23OSim': list_G23OSim_patient})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi_patient-tiantan.xlsx',
                               sheet_name='epoch' + str(epoch), index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi_patient-tiantan.xlsx', mode='a',
                                engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)

        ###############cambridge
        list_all_Diag, list_GBM, list_G4A, list_G3A, list_G2A, list_G3O, list_G2O, \
            list_all_DiagSim, list_GBMSim, list_G4ASim, list_G23ASim, list_G23OSim, \
            list_all_Diag_patient, list_GBM_patient, list_G4A_patient, list_G3A_patient, list_G2A_patient, list_G3O_patient, list_G2O_patient, \
            list_all_DiagSim_patient, list_GBMSim_patient, \
            list_G4ASim_patient, list_G23ASim_patient, list_G23OSim_patient, list_IDH, list_1p19q, list_CDKN, list_IDH_patient, list_1p19q_patient, list_CDKN_patient \
            = test_stage2(opt, Mine_model_init, Mine_model_body, Mine_model_Cls, Mine_model_molecular,Mine_model_Graph, testDataset_cam, gpuID,flag_tiantan=True)

        print('test-cam in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_Diag[0], list_all_Diag[1], list_all_Diag[2], list_all_Diag[6]))
        print('test-cam in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM[1], list_GBM[2], list_GBM[6]))
        print('test-cam in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A[1], list_G4A[2], list_G4A[6]))
        print('test-cam in epoch: %d/%d, G3O || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G3O[1], list_G3O[2], list_G3O[6]))
        print('test-cam in epoch: %d/%d, G2O || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G2O[1], list_G2O[2], list_G2O[6]))
        print('test-cam in epoch: %d/%d, DiagSim || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_DiagSim[0], list_all_DiagSim[1], list_all_DiagSim[2], list_all_DiagSim[6]))
        print('test-cam in epoch: %d/%d, GBMSim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBMSim[1], list_GBMSim[2], list_GBMSim[6]))
        print('test-cam in epoch: %d/%d, G4ASim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4ASim[1], list_G4ASim[2], list_G4ASim[6]))
        print('test-cam in epoch: %d/%d, G23OSim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G23OSim[1], list_G23OSim[2], list_G23OSim[6]))

        dataframe = pd.DataFrame(
            {'metrics': metrics, 'Diag': list_all_Diag, 'GBM': list_GBM, 'G4A': list_G4A,
             'G3O': list_G3O, 'G2O': list_G2O, 'DiagSim': list_all_DiagSim, 'GBMSim': list_GBMSim,
             'G4ASim': list_G4ASim, 'G23OSim': list_G23OSim})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi-cam.xlsx',sheet_name='epoch' + str(epoch), index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi-cam.xlsx', mode='a',engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)

        print('test-cam-patient in epoch: %d/%d, Diag || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_Diag_patient[0], list_all_Diag_patient[1], list_all_Diag_patient[2], list_all_Diag_patient[6]))
        print('test-cam-patient in epoch: %d/%d, GBM || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBM_patient[1], list_GBM_patient[2], list_GBM_patient[6]))
        print('test-cam-patient in epoch: %d/%d, G4A || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4A_patient[1], list_G4A_patient[2], list_G4A_patient[6]))
        print('test-cam-patient in epoch: %d/%d, G3O || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G3O_patient[1], list_G3O_patient[2], list_G3O_patient[6]))
        print('test-cam-patient in epoch: %d/%d, G2O || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G2O_patient[1], list_G2O_patient[2], list_G2O_patient[6]))
        print('test-cam-patient in epoch: %d/%d, DiagSim || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_all_DiagSim_patient[0], list_all_DiagSim_patient[1], list_all_DiagSim_patient[2], list_all_DiagSim_patient[6]))
        print('test-cam-patient in epoch: %d/%d, GBMSim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_GBMSim_patient[1], list_GBMSim_patient[2], list_GBMSim_patient[6]))
        print('test-cam-patient in epoch: %d/%d, G4ASim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G4ASim_patient[1], list_G4ASim_patient[2], list_G4ASim_patient[6]))
        print('test-cam-patient in epoch: %d/%d, G23OSim || sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_G23OSim_patient[1], list_G23OSim_patient[2], list_G23OSim_patient[6]))

        dataframe = pd.DataFrame(
            {'metrics': metrics, 'Diag': list_all_Diag_patient, 'GBM': list_GBM_patient, 'G4A': list_G4A_patient,
             'G3O': list_G3O_patient, 'G2O': list_G2O_patient, 'DiagSim': list_all_DiagSim_patient,
             'GBMSim': list_GBMSim_patient, 'G4ASim': list_G4ASim_patient,'G23OSim': list_G23OSim_patient})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi_patient-cam.xlsx',
                               sheet_name='epoch' + str(epoch), index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi_patient-cam.xlsx', mode='a',
                                engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)

# from lifelines.utils import concordance_index
# from sksurv.metrics import concordance_index_censored
def surv_acc(GT,pred,mode='none'):
    if mode=='none':
        count=0
        for i in range(GT.shape[0]):
            if GT[i]==pred[i]:
                count+=1
        return count/GT.shape[0]
    elif mode=='risk0':
        count = 0
        for i in range(GT.shape[0]):
            if GT[i] == 0 and (pred[i]>=-3 and pred[i]<-2):
                count += 1
            elif GT[i] == 1 and (pred[i]>=-2 and pred[i]<-1):
                count += 1
            elif GT[i] == 2 and (pred[i]>=-1 and pred[i]<0):
                count += 1

        return count / GT.shape[0]
    elif mode == 'risk1':
        count = 0
        for i in range(GT.shape[0]):
            if GT[i] == 2 and (pred[i] >= -3 and pred[i] < -2):
                count += 1
            elif GT[i] == 1 and (pred[i] >= -2 and pred[i] < -1):
                count += 1
            elif GT[i] == 0 and (pred[i] >= -1 and pred[i] < 0):
                count += 1

        return count / GT.shape[0]


def test_surv(opt, Mine_model_init, Mine_model_His, Mine_model_Cls,Mine_model_molecular, Mine_model_Graph,Mine_model_surv,testLoader_all, gpuID,epoch):
    Mine_model_init.eval()
    Mine_model_His.eval()
    Mine_model_Cls.eval()
    Mine_model_molecular.eval()
    Mine_model_Graph.eval()
    Mine_model_surv.eval()
    gpuID = opt['gpus']

    ############################################
    ######################  TCGA
    ##########################################
    risk_pred_all, risk_pred_all_life, censor_all, survtime_all = np.array([]), np.array([]), np.array([]), np.array([])
    test_bar = tqdm(testLoader_all)
    count = 0
    for packs in test_bar:
        img = packs[0]  ##(BS,N,1024)
        label = packs[1]
        file_name = packs[2]
        img = img.cuda(gpuID[0])
        label = label.cuda(gpuID[0])
        label_surv = label[:, 0]
        label_censor = label[:, 1]
        label_event = label[:, 2]
        event_time = label[:, 3]
        age = label[:, 4].to(torch.float32)
        sex = label[:, 5].to(torch.float32)
        count += 1
        saliency_map_His, saliency_map_Grade = saliency_map_read_stage2(opt, file_name)
        saliency_map_His = torch.from_numpy(np.array(saliency_map_His)).float().cuda(gpuID[0])
        saliency_map_Grade = torch.from_numpy(np.array(saliency_map_Grade)).float().cuda(gpuID[0])
        ### ### forward WHO 2007
        init_feature = Mine_model_init(img)  # (BS,2500,1024)
        hidden_states_his, hidden_states_grade, encoded_His, encoded_Grade = Mine_model_His(init_feature,
                                                                                            saliency_map_His,
                                                                                            saliency_map_Grade)
        encoded_Grade = Mine_model_Cls(encoded_His, encoded_Grade)
        ### ### forward molecular
        encoded_IDH, encoded_1p19q, encoded_CDKN = Mine_model_molecular(init_feature)
        encoded_IDH = Mine_model_Graph(encoded_IDH, encoded_1p19q, encoded_CDKN)
        ### ### surv pred
        hazards, S = Mine_model_surv(encoded_Grade, encoded_IDH,age,sex)
        risk = -torch.sum(S, dim=1)
        risk_pred_all = np.concatenate((risk_pred_all, risk.detach().cpu().numpy().reshape(-1)))
        censor_all = np.concatenate(
            (censor_all, label_censor.detach().cpu().numpy().reshape(-1)))  # Logging Information
        survtime_all = np.concatenate(
            (survtime_all, event_time.detach().cpu().numpy().reshape(-1)))  # Logging Information
        risk_pred_all_life = np.concatenate((risk_pred_all_life, hazards.argmax(-1).detach().cpu().numpy().reshape(-1)))

    c_index = concordance_index_censored((1 - censor_all).astype(bool), survtime_all, risk_pred_all, tied_tol=1e-08)[0]
    # c_index =concordance_index(survtime_all, -risk_pred_all_life, censor_all)
    with open('./logs/' + opt['name'] + '/cindex.txt', 'a+') as f:
        f.write('epoch: %d cindex: %.3f' % (epoch, c_index))
        f.write('\n')
    print('testLoader: Epoch: {}, c_index: {:.4f}'.format(epoch, c_index))


def test_predall(opt,Mine_model_init,Mine_model_body,Mine_model_Cls,Mine_model_molecular, Mine_model_Graph, dataloader, gpuID,epoch,testDataset_IvYGAP,testDataset_cam):
    metrics = ['acc', 'sen', 'spec', 'precision', 'recall', 'f1', 'AUC']
    if opt['TrainingSet'] == 'All':
        list_MGMT,list_EGFR,list_ATRX,list_PTEN,list_TERT,list_P53,list_710,list_PDGFRA,list_OLIG2\
            = test_allmarker(opt, Mine_model_init, Mine_model_body, Mine_model_Cls,Mine_model_molecular, Mine_model_Graph, dataloader, gpuID)
        print('test in epoch: %d/%d, MGMT || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_MGMT[0], list_MGMT[1], list_MGMT[2], list_MGMT[6]))
        print('test in epoch: %d/%d, EGFR || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_EGFR[0], list_EGFR[1], list_EGFR[2], list_EGFR[6]))
        print('test in epoch: %d/%d, ATRX || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_ATRX[0], list_ATRX[1], list_ATRX[2], list_ATRX[6]))
        print('test in epoch: %d/%d, PTEN || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_PTEN[0], list_PTEN[1], list_PTEN[2], list_PTEN[6]))
        print('test in epoch: %d/%d, TERT || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_TERT[0], list_TERT[1], list_TERT[2], list_TERT[6]))
        print('test in epoch: %d/%d, P53 || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_P53[0], list_P53[1], list_P53[2], list_P53[6]))
        print('test in epoch: %d/%d, 710 || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_710[0], list_710[1], list_710[2], list_710[6]))
        print('test in epoch: %d/%d, PDGFRA || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_PDGFRA[0], list_PDGFRA[1], list_PDGFRA[2], list_PDGFRA[6]))
        print('test in epoch: %d/%d, OLIG2 || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_OLIG2[0], list_OLIG2[1], list_OLIG2[2], list_OLIG2[6]))
        dataframe = pd.DataFrame(
            {'metrics': metrics,'MGMT': list_MGMT, 'EGFR': list_EGFR, 'ATRX': list_ATRX,'PTEN': list_PTEN, 'TERT': list_TERT, 'P53': list_P53
             ,'710': list_710, 'PDGFRA': list_PDGFRA, 'OLIG2': list_OLIG2})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi.xlsx', sheet_name='epoch' + str(epoch),index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi.xlsx', mode='a', engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
    if opt['TrainingSet'] == 'TCGA':
        list_MGMT,list_EGFR,list_ATRX,list_PTEN,list_TERT,list_P53,list_710,list_PDGFRA\
            = test_allmarker(opt, Mine_model_init, Mine_model_body, Mine_model_Cls,Mine_model_molecular, Mine_model_Graph, dataloader, gpuID)
        print('test in epoch: %d/%d, MGMT || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_MGMT[0], list_MGMT[1], list_MGMT[2], list_MGMT[6]))
        print('test in epoch: %d/%d, EGFR || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_EGFR[0], list_EGFR[1], list_EGFR[2], list_EGFR[6]))
        print('test in epoch: %d/%d, ATRX || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_ATRX[0], list_ATRX[1], list_ATRX[2], list_ATRX[6]))
        print('test in epoch: %d/%d, PTEN || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_PTEN[0], list_PTEN[1], list_PTEN[2], list_PTEN[6]))
        print('test in epoch: %d/%d, TERT || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_TERT[0], list_TERT[1], list_TERT[2], list_TERT[6]))
        print('test in epoch: %d/%d, P53 || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_P53[0], list_P53[1], list_P53[2], list_P53[6]))
        print('test in epoch: %d/%d, 710 || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_710[0], list_710[1], list_710[2], list_710[6]))
        print('test in epoch: %d/%d, PDGFRA || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_PDGFRA[0], list_PDGFRA[1], list_PDGFRA[2], list_PDGFRA[6]))
        dataframe = pd.DataFrame(
            {'metrics': metrics,'MGMT': list_MGMT, 'EGFR': list_EGFR, 'ATRX': list_ATRX,'PTEN': list_PTEN, 'TERT': list_TERT, 'P53': list_P53
             ,'710': list_710, 'PDGFRA': list_PDGFRA})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi.xlsx', sheet_name='epoch' + str(epoch),index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi.xlsx', mode='a', engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
    if opt['TrainingSet'] == 'Tiantan':
        list_MGMT,list_EGFR,list_ATRX,list_PTEN,list_P53,list_OLIG2\
                = test_allmarker(opt, Mine_model_init, Mine_model_body, Mine_model_Cls,Mine_model_molecular, Mine_model_Graph, dataloader, gpuID)
        print('test in epoch: %d/%d, MGMT || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_MGMT[0], list_MGMT[1], list_MGMT[2], list_MGMT[6]))
        print('test in epoch: %d/%d, EGFR || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_EGFR[0], list_EGFR[1], list_EGFR[2], list_EGFR[6]))
        print('test in epoch: %d/%d, ATRX || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_ATRX[0], list_ATRX[1], list_ATRX[2], list_ATRX[6]))
        print('test in epoch: %d/%d, PTEN || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_PTEN[0], list_PTEN[1], list_PTEN[2], list_PTEN[6]))
        print('test in epoch: %d/%d, P53 || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_P53[0], list_P53[1], list_P53[2], list_P53[6]))
        print('test in epoch: %d/%d, OLIG2 || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_OLIG2[0], list_OLIG2[1], list_OLIG2[2], list_OLIG2[6]))
        dataframe = pd.DataFrame(
            {'metrics': metrics,'MGMT': list_MGMT, 'EGFR': list_EGFR, 'ATRX': list_ATRX,'PTEN': list_PTEN,  'P53': list_P53
             , 'OLIG2': list_OLIG2})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi.xlsx', sheet_name='epoch' + str(epoch),index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi.xlsx', mode='a', engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
    ######### external
    ######### IvYGAP
    list_MGMT,list_EGFR,list_PTEN\
            = test_allmarker(opt, Mine_model_init, Mine_model_body, Mine_model_Cls,Mine_model_molecular, Mine_model_Graph, testDataset_IvYGAP, gpuID,flag_IvYGAP=True)
    print('test in epoch: %d/%d, MGMT || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_MGMT[0], list_MGMT[1], list_MGMT[2], list_MGMT[6]))
    print('test in epoch: %d/%d, EGFR || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_EGFR[0], list_EGFR[1], list_EGFR[2], list_EGFR[6]))
    print('test in epoch: %d/%d, PTEN || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_PTEN[0], list_PTEN[1], list_PTEN[2], list_PTEN[6]))
    dataframe = pd.DataFrame(
        {'metrics': metrics,'MGMT': list_MGMT, 'EGFR': list_EGFR,'PTEN': list_PTEN})
    if epoch < 7:
        dataframe.to_excel('./logs/' + opt['name'] + '/IvYGAP-result-wsi.xlsx', sheet_name='epoch' + str(epoch),index=False)
    else:
        with pd.ExcelWriter('./logs/' + opt['name'] + '/IvYGAP-result-wsi.xlsx', mode='a', engine='openpyxl') as writer:
            dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
    ######### Cam
    list_MGMT= test_allmarker(opt, Mine_model_init, Mine_model_body, Mine_model_Cls,Mine_model_molecular, Mine_model_Graph, testDataset_cam, gpuID,flag_cam=True)
    print('test in epoch: %d/%d, MGMT || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_MGMT[0], list_MGMT[1], list_MGMT[2], list_MGMT[6]))
    dataframe = pd.DataFrame(
        {'metrics': metrics,'MGMT': list_MGMT})
    if epoch < 7:
        dataframe.to_excel('./logs/' + opt['name'] + '/Cam-result-wsi.xlsx', sheet_name='epoch' + str(epoch),index=False)
    else:
        with pd.ExcelWriter('./logs/' + opt['name'] + '/Cam-result-wsi.xlsx', mode='a', engine='openpyxl') as writer:
            dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)



def test_predall_CNN(opt,Mine_CNN_cls, dataloader, gpuID,epoch,testDataset_IvYGAP,testDataset_cam):
    metrics = ['acc', 'sen', 'spec', 'precision', 'recall', 'f1', 'AUC']
    if opt['TrainingSet'] == 'All':
        list_MGMT,list_EGFR,list_ATRX,list_PTEN,list_TERT,list_P53,list_710,list_PDGFRA,list_OLIG2\
            = test_allmarker_CNN(opt, Mine_CNN_cls, dataloader, gpuID)
        print('test in epoch: %d/%d, MGMT || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_MGMT[0], list_MGMT[1], list_MGMT[2], list_MGMT[6]))
        print('test in epoch: %d/%d, EGFR || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_EGFR[0], list_EGFR[1], list_EGFR[2], list_EGFR[6]))
        print('test in epoch: %d/%d, ATRX || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_ATRX[0], list_ATRX[1], list_ATRX[2], list_ATRX[6]))
        print('test in epoch: %d/%d, PTEN || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_PTEN[0], list_PTEN[1], list_PTEN[2], list_PTEN[6]))
        print('test in epoch: %d/%d, TERT || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_TERT[0], list_TERT[1], list_TERT[2], list_TERT[6]))
        print('test in epoch: %d/%d, P53 || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_P53[0], list_P53[1], list_P53[2], list_P53[6]))
        print('test in epoch: %d/%d, 710 || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_710[0], list_710[1], list_710[2], list_710[6]))
        print('test in epoch: %d/%d, PDGFRA || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_PDGFRA[0], list_PDGFRA[1], list_PDGFRA[2], list_PDGFRA[6]))
        print('test in epoch: %d/%d, OLIG2 || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_OLIG2[0], list_OLIG2[1], list_OLIG2[2], list_OLIG2[6]))
        dataframe = pd.DataFrame(
            {'metrics': metrics,'MGMT': list_MGMT, 'EGFR': list_EGFR, 'ATRX': list_ATRX,'PTEN': list_PTEN, 'TERT': list_TERT, 'P53': list_P53
             ,'710': list_710, 'PDGFRA': list_PDGFRA, 'OLIG2': list_OLIG2})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi.xlsx', sheet_name='epoch' + str(epoch),index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi.xlsx', mode='a', engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
    if opt['TrainingSet'] == 'TCGA':
        list_MGMT,list_EGFR,list_ATRX,list_PTEN,list_TERT,list_P53,list_710,list_PDGFRA\
            = test_allmarker_CNN(opt, Mine_CNN_cls, dataloader, gpuID)
        print('test in epoch: %d/%d, MGMT || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_MGMT[0], list_MGMT[1], list_MGMT[2], list_MGMT[6]))
        print('test in epoch: %d/%d, EGFR || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_EGFR[0], list_EGFR[1], list_EGFR[2], list_EGFR[6]))
        print('test in epoch: %d/%d, ATRX || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_ATRX[0], list_ATRX[1], list_ATRX[2], list_ATRX[6]))
        print('test in epoch: %d/%d, PTEN || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_PTEN[0], list_PTEN[1], list_PTEN[2], list_PTEN[6]))
        print('test in epoch: %d/%d, TERT || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_TERT[0], list_TERT[1], list_TERT[2], list_TERT[6]))
        print('test in epoch: %d/%d, P53 || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_P53[0], list_P53[1], list_P53[2], list_P53[6]))
        print('test in epoch: %d/%d, 710 || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_710[0], list_710[1], list_710[2], list_710[6]))
        print('test in epoch: %d/%d, PDGFRA || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_PDGFRA[0], list_PDGFRA[1], list_PDGFRA[2], list_PDGFRA[6]))
        dataframe = pd.DataFrame(
            {'metrics': metrics,'MGMT': list_MGMT, 'EGFR': list_EGFR, 'ATRX': list_ATRX,'PTEN': list_PTEN, 'TERT': list_TERT, 'P53': list_P53
             ,'710': list_710, 'PDGFRA': list_PDGFRA})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi.xlsx', sheet_name='epoch' + str(epoch),index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi.xlsx', mode='a', engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
    if opt['TrainingSet'] == 'Tiantan':
        list_MGMT,list_EGFR,list_ATRX,list_PTEN,list_P53,list_OLIG2\
                = test_allmarker_CNN(opt, Mine_CNN_cls, dataloader, gpuID)
        print('test in epoch: %d/%d, MGMT || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_MGMT[0], list_MGMT[1], list_MGMT[2], list_MGMT[6]))
        print('test in epoch: %d/%d, EGFR || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_EGFR[0], list_EGFR[1], list_EGFR[2], list_EGFR[6]))
        print('test in epoch: %d/%d, ATRX || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_ATRX[0], list_ATRX[1], list_ATRX[2], list_ATRX[6]))
        print('test in epoch: %d/%d, PTEN || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_PTEN[0], list_PTEN[1], list_PTEN[2], list_PTEN[6]))
        print('test in epoch: %d/%d, P53 || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_P53[0], list_P53[1], list_P53[2], list_P53[6]))
        print('test in epoch: %d/%d, OLIG2 || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_OLIG2[0], list_OLIG2[1], list_OLIG2[2], list_OLIG2[6]))
        dataframe = pd.DataFrame(
            {'metrics': metrics,'MGMT': list_MGMT, 'EGFR': list_EGFR, 'ATRX': list_ATRX,'PTEN': list_PTEN,  'P53': list_P53
             , 'OLIG2': list_OLIG2})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi.xlsx', sheet_name='epoch' + str(epoch),index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi.xlsx', mode='a', engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
    ######### external
    ######### IvYGAP
    list_MGMT,list_EGFR,list_PTEN\
            = test_allmarker_CNN(opt, Mine_CNN_cls, testDataset_IvYGAP, gpuID,flag_IvYGAP=True)
    print('test in epoch: %d/%d, MGMT || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_MGMT[0], list_MGMT[1], list_MGMT[2], list_MGMT[6]))
    print('test in epoch: %d/%d, EGFR || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_EGFR[0], list_EGFR[1], list_EGFR[2], list_EGFR[6]))
    print('test in epoch: %d/%d, PTEN || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_PTEN[0], list_PTEN[1], list_PTEN[2], list_PTEN[6]))
    dataframe = pd.DataFrame(
        {'metrics': metrics,'MGMT': list_MGMT, 'EGFR': list_EGFR,'PTEN': list_PTEN})
    if epoch < 7:
        dataframe.to_excel('./logs/' + opt['name'] + '/IvYGAP-result-wsi.xlsx', sheet_name='epoch' + str(epoch),index=False)
    else:
        with pd.ExcelWriter('./logs/' + opt['name'] + '/IvYGAP-result-wsi.xlsx', mode='a', engine='openpyxl') as writer:
            dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
    ######### Cam
    list_MGMT= test_allmarker_CNN(opt, Mine_CNN_cls, testDataset_cam, gpuID,flag_cam=True)
    print('test in epoch: %d/%d, MGMT || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_MGMT[0], list_MGMT[1], list_MGMT[2], list_MGMT[6]))
    dataframe = pd.DataFrame(
        {'metrics': metrics,'MGMT': list_MGMT})
    if epoch < 7:
        dataframe.to_excel('./logs/' + opt['name'] + '/Cam-result-wsi.xlsx', sheet_name='epoch' + str(epoch),index=False)
    else:
        with pd.ExcelWriter('./logs/' + opt['name'] + '/Cam-result-wsi.xlsx', mode='a', engine='openpyxl') as writer:
            dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)

def test_predall_endtoend(opt,Mine_model_init,Mine_model_body,Mine_model_Cls, dataloader, gpuID,epoch,testDataset_IvYGAP,testDataset_cam):
    metrics = ['acc', 'sen', 'spec', 'precision', 'recall', 'f1', 'AUC']
    if opt['TrainingSet'] == 'All':
        list_MGMT,list_EGFR,list_ATRX,list_PTEN,list_TERT,list_P53,list_710,list_PDGFRA,list_OLIG2\
            = test_allmarker_endtoend(opt, Mine_model_init,Mine_model_body,Mine_model_Cls, dataloader, gpuID)
        print('test in epoch: %d/%d, MGMT || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_MGMT[0], list_MGMT[1], list_MGMT[2], list_MGMT[6]))
        print('test in epoch: %d/%d, EGFR || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_EGFR[0], list_EGFR[1], list_EGFR[2], list_EGFR[6]))
        print('test in epoch: %d/%d, ATRX || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_ATRX[0], list_ATRX[1], list_ATRX[2], list_ATRX[6]))
        print('test in epoch: %d/%d, PTEN || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_PTEN[0], list_PTEN[1], list_PTEN[2], list_PTEN[6]))
        print('test in epoch: %d/%d, TERT || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_TERT[0], list_TERT[1], list_TERT[2], list_TERT[6]))
        print('test in epoch: %d/%d, P53 || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_P53[0], list_P53[1], list_P53[2], list_P53[6]))
        print('test in epoch: %d/%d, 710 || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_710[0], list_710[1], list_710[2], list_710[6]))
        print('test in epoch: %d/%d, PDGFRA || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_PDGFRA[0], list_PDGFRA[1], list_PDGFRA[2], list_PDGFRA[6]))
        print('test in epoch: %d/%d, OLIG2 || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_OLIG2[0], list_OLIG2[1], list_OLIG2[2], list_OLIG2[6]))
        dataframe = pd.DataFrame(
            {'metrics': metrics,'MGMT': list_MGMT, 'EGFR': list_EGFR, 'ATRX': list_ATRX,'PTEN': list_PTEN, 'TERT': list_TERT, 'P53': list_P53
             ,'710': list_710, 'PDGFRA': list_PDGFRA, 'OLIG2': list_OLIG2})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi.xlsx', sheet_name='epoch' + str(epoch),index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi.xlsx', mode='a', engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
    if opt['TrainingSet'] == 'TCGA':
        list_MGMT,list_EGFR,list_ATRX,list_PTEN,list_TERT,list_P53,list_710,list_PDGFRA\
            = test_allmarker_endtoend(opt, Mine_model_init,Mine_model_body,Mine_model_Cls, dataloader, gpuID)
        print('test in epoch: %d/%d, MGMT || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_MGMT[0], list_MGMT[1], list_MGMT[2], list_MGMT[6]))
        print('test in epoch: %d/%d, EGFR || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_EGFR[0], list_EGFR[1], list_EGFR[2], list_EGFR[6]))
        print('test in epoch: %d/%d, ATRX || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_ATRX[0], list_ATRX[1], list_ATRX[2], list_ATRX[6]))
        print('test in epoch: %d/%d, PTEN || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_PTEN[0], list_PTEN[1], list_PTEN[2], list_PTEN[6]))
        print('test in epoch: %d/%d, TERT || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_TERT[0], list_TERT[1], list_TERT[2], list_TERT[6]))
        print('test in epoch: %d/%d, P53 || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_P53[0], list_P53[1], list_P53[2], list_P53[6]))
        print('test in epoch: %d/%d, 710 || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_710[0], list_710[1], list_710[2], list_710[6]))
        print('test in epoch: %d/%d, PDGFRA || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_PDGFRA[0], list_PDGFRA[1], list_PDGFRA[2], list_PDGFRA[6]))
        dataframe = pd.DataFrame(
            {'metrics': metrics,'MGMT': list_MGMT, 'EGFR': list_EGFR, 'ATRX': list_ATRX,'PTEN': list_PTEN, 'TERT': list_TERT, 'P53': list_P53
             ,'710': list_710, 'PDGFRA': list_PDGFRA})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi.xlsx', sheet_name='epoch' + str(epoch),index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi.xlsx', mode='a', engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
    if opt['TrainingSet'] == 'Tiantan':
        list_MGMT,list_EGFR,list_ATRX,list_PTEN,list_P53,list_OLIG2\
                = test_allmarker_endtoend(opt, Mine_model_init,Mine_model_body,Mine_model_Cls, dataloader, gpuID)
        print('test in epoch: %d/%d, MGMT || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_MGMT[0], list_MGMT[1], list_MGMT[2], list_MGMT[6]))
        print('test in epoch: %d/%d, EGFR || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_EGFR[0], list_EGFR[1], list_EGFR[2], list_EGFR[6]))
        print('test in epoch: %d/%d, ATRX || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_ATRX[0], list_ATRX[1], list_ATRX[2], list_ATRX[6]))
        print('test in epoch: %d/%d, PTEN || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_PTEN[0], list_PTEN[1], list_PTEN[2], list_PTEN[6]))
        print('test in epoch: %d/%d, P53 || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_P53[0], list_P53[1], list_P53[2], list_P53[6]))
        print('test in epoch: %d/%d, OLIG2 || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_OLIG2[0], list_OLIG2[1], list_OLIG2[2], list_OLIG2[6]))
        dataframe = pd.DataFrame(
            {'metrics': metrics,'MGMT': list_MGMT, 'EGFR': list_EGFR, 'ATRX': list_ATRX,'PTEN': list_PTEN,  'P53': list_P53
             , 'OLIG2': list_OLIG2})
        if epoch < 7:
            dataframe.to_excel('./logs/' + opt['name'] + '/result-wsi.xlsx', sheet_name='epoch' + str(epoch),index=False)
        else:
            with pd.ExcelWriter('./logs/' + opt['name'] + '/result-wsi.xlsx', mode='a', engine='openpyxl') as writer:
                dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
    ######### external
    ######### IvYGAP
    list_MGMT,list_EGFR,list_PTEN\
            = test_allmarker_endtoend(opt, Mine_model_init,Mine_model_body,Mine_model_Cls, testDataset_IvYGAP, gpuID,flag_IvYGAP=True)
    print('test in epoch: %d/%d, MGMT || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_MGMT[0], list_MGMT[1], list_MGMT[2], list_MGMT[6]))
    print('test in epoch: %d/%d, EGFR || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_EGFR[0], list_EGFR[1], list_EGFR[2], list_EGFR[6]))
    print('test in epoch: %d/%d, PTEN || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_PTEN[0], list_PTEN[1], list_PTEN[2], list_PTEN[6]))
    dataframe = pd.DataFrame(
        {'metrics': metrics,'MGMT': list_MGMT, 'EGFR': list_EGFR,'PTEN': list_PTEN})
    if epoch < 7:
        dataframe.to_excel('./logs/' + opt['name'] + '/IvYGAP-result-wsi.xlsx', sheet_name='epoch' + str(epoch),index=False)
    else:
        with pd.ExcelWriter('./logs/' + opt['name'] + '/IvYGAP-result-wsi.xlsx', mode='a', engine='openpyxl') as writer:
            dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)
    ######### Cam
    list_MGMT= test_allmarker_endtoend(opt, Mine_model_init,Mine_model_body,Mine_model_Cls, testDataset_cam, gpuID,flag_cam=True)
    print('test in epoch: %d/%d, MGMT || acc:%.3f,sen:%.3f,spec:%.3f, auc:%.3f' % (epoch, 70, list_MGMT[0], list_MGMT[1], list_MGMT[2], list_MGMT[6]))
    dataframe = pd.DataFrame(
        {'metrics': metrics,'MGMT': list_MGMT})
    if epoch < 7:
        dataframe.to_excel('./logs/' + opt['name'] + '/Cam-result-wsi.xlsx', sheet_name='epoch' + str(epoch),index=False)
    else:
        with pd.ExcelWriter('./logs/' + opt['name'] + '/Cam-result-wsi.xlsx', mode='a', engine='openpyxl') as writer:
            dataframe.to_excel(writer, sheet_name='epoch' + str(epoch), index=False)