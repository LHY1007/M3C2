import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.cuda
from torch.autograd import Variable
import basic_net as basic_net
import yaml
from yaml.loader import SafeLoader
from net import *
import copy
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from utils import FocalLoss
import scipy.sparse as sp
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def adjust_gradients(g1, g2, labels, lambda_reg=0.1):
    cos_sim = torch.cosine_similarity(g1.reshape(-1), g2.reshape(-1), dim=0)
    if cos_sim < 0:  # 检查 conflict
        num_labels = (labels == 1).sum().item()
        if num_labels > (len(labels) / 2):
            original_length = torch.norm(g1)
            g1 = g1 - lambda_reg * (torch.dot(g1.reshape(-1), g2.reshape(-1)) * g2 / torch.norm(g2))
            new_length = torch.norm(g1)
            g1 = g1 * (original_length / new_length)
        else:
            original_length = torch.norm(g2)
            g2 = g2 - lambda_reg * (torch.dot(g2.reshape(-1), g1.reshape(-1)) * g1 / torch.norm(g1))
            new_length = torch.norm(g2)
            g2 = g2 * (original_length / new_length)
    return g1, g2

def adjust_gradients_longth(g1, g2, labels, lambda_reg=0.1):
    cos_sim = torch.cosine_similarity(g1.reshape(-1), g2.reshape(-1), dim=0)
    if cos_sim < 0:  # 检查 conflict
        num_labels = (labels == 1).sum().item()
        if num_labels > (len(labels) / 2):
            g1 = g1 - lambda_reg * (torch.dot(g1.reshape(-1), g2.reshape(-1)) * g2 / torch.norm(g2))
        else:
            g2 = g2 - lambda_reg * (torch.dot(g2.reshape(-1), g1.reshape(-1)) * g1 / torch.norm(g1))
    return g1, g2

def adjust_gradients_old(g1, g2, labels, lambda_reg=0.1):
    cos_sim = torch.cosine_similarity(g1.reshape(-1), g2.reshape(-1), dim=0)
    if cos_sim < 0:  # 检查 conflict
            original_length = torch.norm(g2)
            g2 = g2 - lambda_reg * (torch.dot(g2.reshape(-1), g1.reshape(-1)) * g1 / torch.norm(g1))
            new_length = torch.norm(g2)
            g2 = g2 * (original_length / new_length)
    return g1, g2

class Mine_Task(nn.Module):
    def __init__(self, opt,input_dim=1024, output_dim=6):
        super(Mine_Task, self).__init__()
        self.fixdim = opt['fixdim']
        self.dim = 512
        self.criterion_ce_Task = nn.CrossEntropyLoss(
            weight=torch.from_numpy(np.array([1, 4.6, 4.1, 3.3])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])#1
        self.criterion_ce_His = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 4.6, 4.1, 3.3])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        #[1, 4.6, 4.1, 3.3]

        self.His = nn.Linear(512, self.dim)
        self.Mark = nn.Linear(3*512, self.dim)
        self.fc_out = nn.Linear(2*self.dim, 4)
    def forward(self, His, Mark):

        His = self.His(His) #[BS,512]
        Mark = self.Mark(Mark)#[BS,512]
        #2
        Task = torch.cat((His, Mark), dim=1)
        output = self.fc_out(Task)
        return output
    
    def calculateLoss_Task(self, his_mark,label):
        self.loss_Task = self.criterion_ce_Task(his_mark, label)
        return self.loss_Task

class CL(nn.Module):
    def __init__(self, input_dim=1024, common_dim=1024, private_dim=1024, num_layers=2):
        super(CL, self).__init__()
        self.fc_His = self._make_layers(input_dim, private_dim)
        self.fc_Mark = self._make_layers(input_dim, private_dim)
        self.fc_Pub_His = self._make_layers(input_dim, common_dim)
        self.fc_Pub_Mark = self._make_layers(input_dim, common_dim)

    def _make_layers(self, input_dim, output_dim):
        layers = []
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, High, Low,dis):
        features_His = torch.cat((Low['His'], High['His']), dim=1)
        features_Mark = torch.cat((Low['Mark'], High['Mark']), dim=1)
        features_Pub_His = torch.cat((Low['Pub_His'], High['Pub_His']), dim=1)
        features_Pub_Mark = torch.cat((Low['Pub_Mark'], High['Pub_Mark']), dim=1)

        His = self.fc_His(features_His)
        Mark = self.fc_Mark(features_Mark)
        Pub_His = self.fc_Pub_His(features_Pub_His)
        Pub_Mark = self.fc_Pub_Mark(features_Pub_Mark)
        if dis==False:
            His = torch.cat((Low['His'], High['Mark']), dim=1)
            Mark = torch.cat((Low['His'], High['Mark']), dim=1)
        return His, Mark, Pub_His, Pub_Mark


class Mine_init(nn.Module):
    def __init__(self, opt, vis=False,if_end2end=False):
        super(Mine_init, self).__init__()
        self.opt = opt
        self.vis = vis
        self.size = [1024, 512]
        self.if_end2end=if_end2end

        self.weights = nn.ParameterDict({#设置尺度权重[his_p,his_com,maker_com,maker_p]
            'low': nn.Parameter(torch.tensor([1.0, 0.5, 0.5, 0.0], requires_grad=True)),
            'high': nn.Parameter(torch.tensor([0.0, 0.5, 0.5, 1.0], requires_grad=True)),
        })

        opt['fixdim'] = 5000
        self.default_patchnum = opt['fixdim']

        self.position_embeddings1 = nn.Parameter(torch.zeros(1, self.default_patchnum, 512))
        self.position_embeddings2 = nn.Parameter(torch.zeros(1, self.default_patchnum, 512))
        fc = [nn.Linear(1024, self.size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.attention_init1 = nn.Sequential(*fc)
        self.attention_init2 = nn.Sequential(*fc)

        self.contrast = CL(1024, 1024, 1024, num_layers=3)  #subnet

    def forward(self,x20,x10,dis=True):

        weights_low = F.softmax(self.weights['low'], dim=0)
        weights_high = F.softmax(self.weights['high'], dim=0)
        # Apply learnable weights
        x10_dict = {
            'His': x10 * weights_low[0],
            'Pub_His': x10 * weights_low[1],
            'Pub_Mark': x10 * weights_low[2],
            'Mark': x10 * weights_low[3]
        }

        x20_dict = {
            'His': x20 * weights_high[0],
            'Pub_His': x20 * weights_high[1],
            'Pub_Mark': x20 * weights_high[2],
            'Mark': x20 * weights_high[3]

        }
        His, Mark, Pub_His, Pub_Mark = self.contrast(x20_dict, x10_dict,dis)


        if dis==False:
            His = His
            Mark = Mark
        else:
            His = His + Pub_His
            Mark = Mark + Pub_Mark
        features_task1 = self.attention_init1(His) + self.position_embeddings1  # [B, n, 512]
        features_task2 = self.attention_init2(Mark) + self.position_embeddings2  # [B, n, 512]
        
        return features_task1, features_task2, His, Mark, Pub_His, Pub_Mark
        # return hidden_states_stem

    def calculateLoss_init(self, His, Mark, Pub_His, Pub_Mark):
        # Calculate Euclidean distance
        pub_loss = F.mse_loss(Pub_His, Pub_Mark)
        his_loss = F.mse_loss(His, Pub_His)
        mark_loss = F.mse_loss(Mark, Pub_Mark)
        p_loss = F.mse_loss(His, Mark)
        # Calculate total loss
        loss = pub_loss / (
                    his_loss + mark_loss + p_loss + 1e-8)
        return loss
    def calculateLoss_init_1(self, His, Mark, Pub_His, Pub_Mark):
        # Calculate Euclidean distance
        pub_loss = F.mse_loss(Pub_His, Pub_Mark)
        his_loss = F.mse_loss(His, Pub_His)
        mark_loss = F.mse_loss(Mark, Pub_Mark)
        p_loss = F.mse_loss(His, Mark)
        # Calculate total loss
        loss = pub_loss / (
                    his_loss + mark_loss + 1e-8)
        return loss
    def calculateLoss_init_2(self, His, Mark, Pub_His, Pub_Mark):
        # Calculate Euclidean distance
        pub_loss = F.mse_loss(Pub_His, Pub_Mark)
        his_loss = F.mse_loss(His, Pub_His)
        mark_loss = F.mse_loss(Mark, Pub_Mark)
        p_loss = F.mse_loss(His, Mark)
        # Calculate total loss
        loss = pub_loss / (
                    p_loss + 1e-8)
        return loss

class Mine_IDH(nn.Module):
    def __init__(self, opt, vis=False):
        super(Mine_IDH, self).__init__()
        self.opt = opt
        self.vis = vis
        self.size = [1024, 512]
        self.n_classes_IDH = 2
        # Trans blocks
        self.layer_IDH = nn.ModuleList()
        for _ in range(self.opt['Network']['IDH_layers']):
            layer = Block(opt, self.size[1], vis)
            self.layer_IDH.append(copy.deepcopy(layer))
        self.encoder_norm_IDH = LayerNorm(self.size[1], eps=1e-6)

    def forward(self, hidden_states):
        attn_weights_IDH = []
        for layer_block in self.layer_IDH:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights_IDH.append(weights)

        encoded_IDH = self.encoder_norm_IDH(hidden_states)  # [B,2500,512]
        # encoded_IDH = torch.mean(encoded_IDH, dim=1)
        return hidden_states,encoded_IDH


class Mine_1p19q(nn.Module):
    def __init__(self, opt, vis=False):
        super(Mine_1p19q, self).__init__()
        self.opt = opt
        self.vis = vis
        self.size = [1024, 512]
        self.n_classes_1p19q = 2
        self.layer_1p19q = nn.ModuleList()
        for _ in range(self.opt['Network']['1p19q_layers']):
            layer = Block(opt, self.size[1], vis)
            self.layer_1p19q.append(copy.deepcopy(layer))
        self.encoder_norm_1p19q = LayerNorm(self.size[1], eps=1e-6)

    def forward(self,hidden_states):
        attn_weights_1p19q = []
        for layer_block in self.layer_1p19q:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights_1p19q.append(weights)


        encoded_1p19q = self.encoder_norm_1p19q(hidden_states)# [B,2500,512]
        # encoded_1p19q = torch.mean(encoded_1p19q, dim=1)

        return hidden_states,encoded_1p19q


class Mine_CDKN(nn.Module):
    def __init__(self, opt, vis=False):
        super(Mine_CDKN, self).__init__()
        self.opt = opt
        self.vis = vis
        self.size = [1024, 512]
        self.n_classes_CDKN = 2
        self.layer_CDKN = nn.ModuleList()
        for _ in range(self.opt['Network']['CDKN_layers']):
            layer = Block(opt, self.size[1], vis)
            self.layer_CDKN.append(copy.deepcopy(layer))
        self.encoder_norm_CDKN = LayerNorm(self.size[1], eps=1e-6)

    def forward(self, hidden_states):
        attn_weights_CDKN = []
        for layer_block in self.layer_CDKN:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights_CDKN.append(weights)


        encoded_CDKN = self.encoder_norm_CDKN(hidden_states)# [B,2500,512]
        # encoded_CDKN = torch.mean(encoded_CDKN, dim=1)
        return encoded_CDKN

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import CrossEntropyLoss, Dropout
class Label_correlation_Graph(nn.Module):
    def __init__(self, opt, vis=False):
        super(Label_correlation_Graph, self).__init__()
        self.opt = opt
        self.size = [1024, 512]
        self.adj=np.array([[1,0.4038,0.3035],[1,1,0.1263],[0.2595,0.0436,1]])

        self.alpha=self.opt['Network']['graph_alpha']
        self.n_classes_IDH=2
        self.n_classes_CDKN=2
        self.n_classes_1p19q=2


        self.criterion_ce_IDH = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        self.criterion_ce_1p19q = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 4.6])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        self.criterion_ce_CDKN = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.2, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])

        self.encoder_norm_IDH = LayerNorm(self.size[1], eps=1e-6)
        self.encoder_norm_1p19q = LayerNorm(self.size[1], eps=1e-6)
        self.encoder_norm_CDKN = LayerNorm(self.size[1], eps=1e-6)

        self.gc1 = GraphConvolution(self.size[1], self.size[1])
        self.gc2 = GraphConvolution(self.size[1], 2)
        self.dropout=Dropout(0.5)

        # self._fc2_IDH_1 = nn.Linear(self.opt['fixdim'], self.n_classes_IDH)
        # self._fc2_CDKN_1 = nn.Linear(self.opt['fixdim'], self.n_classes_CDKN)
        # self._fc2_1p19q_1 = nn.Linear(self.opt['fixdim'], self.n_classes_1p19q)
        # # atten
        self.attention_IDH = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.attention_1p19q = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.attention_CDKN = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.attention_V_IDH = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh()
        )
        self.attention_U_IDH = nn.Sequential(
            nn.Linear(512, 128),
            nn.Sigmoid()
        )
        self.attention_weights_IDH = nn.Linear(128, 1)
        self.attention_V_1p19q = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh()
        )
        self.attention_U_1p19q = nn.Sequential(
            nn.Linear(512, 128),
            nn.Sigmoid()
        )
        self.attention_weights_1p19q = nn.Linear(128, 1)
        self.attention_V_CDKN = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh()
        )
        self.attention_U_CDKN = nn.Sequential(
            nn.Linear(512, 128),
            nn.Sigmoid()
        )
        self.attention_weights_CDKN = nn.Linear(128, 1)

        self._fc2_IDH_1 = nn.Linear(self.size[1], self.n_classes_IDH)
        self._fc2_CDKN_1 = nn.Linear(self.size[1], self.n_classes_CDKN)
        self._fc2_1p19q_1 = nn.Linear(self.size[1], self.n_classes_1p19q)


    def forward(self, encoded_IDH,encoded_1p19q,encoded_CDKN):

        encoded_IDH = torch.unsqueeze(encoded_IDH, dim=3)  # [BS,2500,512,1]
        encoded_1p19q = torch.unsqueeze(encoded_1p19q, dim=3)
        encoded_CDKN = torch.unsqueeze(encoded_CDKN, dim=3)
        GCN_input = torch.cat((encoded_IDH, encoded_1p19q, encoded_CDKN), 3)  # [BS,2500,512,3 ]
        GCN_output = F.relu(self.gc1(GCN_input))  # [BS,2500,512,3 ]
        GCN_output = GCN_output * self.alpha + GCN_input * (1 - self.alpha)  # [BS,2500,512,3 ]

        ########################   IDH   ########################
        encoded_IDH = GCN_output[..., 0]  # [BS,2500,512]
        encoded_IDH = self.encoder_norm_IDH(encoded_IDH)  # [BS,2500,512]
        encoded_IDH_ori = encoded_IDH
        A_V_IDH = self.attention_V_IDH(encoded_IDH)  # BxNx128
        A_U_IDH = self.attention_U_IDH(encoded_IDH)  # BxNx128
        A_encoded_IDH = self.attention_weights_IDH(A_V_IDH * A_U_IDH)  # BxNx1
        A_encoded_IDH = F.softmax(A_encoded_IDH, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded_IDH.shape[0]):
            if i == 0:
                Final_con_layer = encoded_IDH[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_IDH[i], 1).expand(-1, encoded_IDH[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_IDH = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_IDH_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_IDH[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_IDH[i], 1).expand(-1, encoded_IDH[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_IDH = torch.cat((Final_con_layer_IDH, torch.unsqueeze(Final_con_layer, 0)),
                                                dim=0)  # BSxNx512
                encoded_IDH_new = torch.cat(
                    (encoded_IDH_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_IDH = encoded_IDH_new  # Bx512

        ########################   1p19q   ########################
        encoded_1p19q = GCN_output[..., 1]  # [BS,2500,512]
        encoded_1p19q = self.encoder_norm_1p19q(encoded_1p19q)  # [BS,2500,512]
        encoded_1p19q_ori = encoded_1p19q
        A_V_1p19q = self.attention_V_1p19q(encoded_1p19q)  # BxNx128
        A_U_1p19q = self.attention_U_1p19q(encoded_1p19q)  # BxNx128
        A_encoded_1p19q = self.attention_weights_1p19q(A_V_1p19q * A_U_1p19q)  # BxNx1
        A_encoded_1p19q = F.softmax(A_encoded_1p19q, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded_1p19q.shape[0]):
            if i == 0:
                Final_con_layer = encoded_1p19q[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_1p19q[i], 1).expand(-1, encoded_1p19q[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_1p19q = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_1p19q_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_1p19q[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_1p19q[i], 1).expand(-1, encoded_1p19q[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_1p19q = torch.cat((Final_con_layer_1p19q, torch.unsqueeze(Final_con_layer, 0)),
                                                  dim=0)  # BSxNx512
                encoded_1p19q_new = torch.cat(
                    (encoded_1p19q_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_1p19q = encoded_1p19q_new  # Bx512

        ########################   CDKN   ########################
        encoded_CDKN = GCN_output[..., 2]  # [BS,2500,512]
        encoded_CDKN = self.encoder_norm_CDKN(encoded_CDKN)  # [BS,2500,512]
        encoded_CDKN_ori = encoded_CDKN
        A_V_CDKN = self.attention_V_CDKN(encoded_CDKN)  # BxNx128
        A_U_CDKN = self.attention_U_CDKN(encoded_CDKN)  # BxNx128
        A_encoded_CDKN = self.attention_weights_CDKN(A_V_CDKN * A_U_CDKN)  # BxNx1
        A_encoded_CDKN = F.softmax(A_encoded_CDKN, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded_CDKN.shape[0]):
            if i == 0:
                Final_con_layer = encoded_CDKN[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_CDKN[i], 1).expand(-1, encoded_CDKN[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_CDKN = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_CDKN_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_CDKN[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_CDKN[i], 1).expand(-1, encoded_CDKN[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_CDKN = torch.cat((Final_con_layer_CDKN, torch.unsqueeze(Final_con_layer, 0)),
                                                 dim=0)  # BSxNx512
                encoded_CDKN_new = torch.cat(
                    (encoded_CDKN_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_CDKN = encoded_CDKN_new  # Bx512

        ####################saliency maps for IDH wt
        weight_IDH_wt = torch.unsqueeze(self._fc2_IDH_1.weight[0], dim=1)  # [512,1]
        saliency_IDH_wt = torch.matmul(Final_con_layer_IDH, weight_IDH_wt)[..., 0]  # [BSxN]
        if self._fc2_IDH_1.bias is not None:
            saliency_IDH_wt = saliency_IDH_wt + self._fc2_IDH_1.bias[0] / encoded_IDH_ori.shape[1]  # [BSxN]


        ####################saliency maps for IDH mut
        weight_IDH_mut = torch.unsqueeze(self._fc2_IDH_1.weight[1], dim=1)  # [512,1]
        saliency_IDH_mut = torch.matmul(Final_con_layer_IDH, weight_IDH_mut)[..., 0]  # [BSxN]
        if self._fc2_IDH_1.bias is not None:
            saliency_IDH_mut = saliency_IDH_mut + self._fc2_IDH_1.bias[1] / encoded_IDH_ori.shape[1]  # [BSxN]

        ####################saliency maps for 1p19q codel
        weight_1p19q_codel = torch.unsqueeze(self._fc2_1p19q_1.weight[1], dim=1)  # [512,1]
        saliency_1p19q_codel = torch.matmul(Final_con_layer_1p19q, weight_1p19q_codel)[..., 0]  # [BSxN]
        if self._fc2_1p19q_1.bias is not None:
            saliency_1p19q_codel = saliency_1p19q_codel + self._fc2_1p19q_1.bias[1] / encoded_1p19q_ori.shape[
                1]  # [BSxN]

        ####################saliency maps for 1p19q noncodel
        weight_1p19q_noncodel = torch.unsqueeze(self._fc2_1p19q_1.weight[0], dim=1)  # [512,1]
        saliency_1p19q_noncodel = torch.matmul(Final_con_layer_1p19q, weight_1p19q_noncodel)[..., 0]  # [BSxN]
        if self._fc2_1p19q_1.bias is not None:
            saliency_1p19q_noncodel = saliency_1p19q_noncodel + self._fc2_1p19q_1.bias[0] / encoded_1p19q_ori.shape[
                1]  # [BSxN]

        ####################saliency maps for CDKN HOMDEL
        weight_CDKN_HOMDEL = torch.unsqueeze(self._fc2_CDKN_1.weight[1], dim=1)  # [512,1]
        saliency_CDKN_HOMDEL = torch.matmul(Final_con_layer_CDKN, weight_CDKN_HOMDEL)[..., 0]  # [BSxN]
        if self._fc2_CDKN_1.bias is not None:
            saliency_CDKN_HOMDEL = saliency_CDKN_HOMDEL + self._fc2_CDKN_1.bias[1] / encoded_CDKN_ori.shape[
                1]  # [BSxN]

        ####################saliency maps for CDKN NonHOMDEL
        weight_CDKN_NonHOMDEL = torch.unsqueeze(self._fc2_CDKN_1.weight[0], dim=1)  # [512,1]
        saliency_CDKN_NonHOMDEL = torch.matmul(Final_con_layer_CDKN, weight_CDKN_NonHOMDEL)[..., 0]  # [BSxN]
        if self._fc2_CDKN_1.bias is not None:
            saliency_CDKN_NonHOMDEL = saliency_CDKN_NonHOMDEL + self._fc2_CDKN_1.bias[0] / encoded_CDKN_ori.shape[
                1]  # [BSxN]

        logits_CDKN = self._fc2_CDKN_1(encoded_CDKN)  # [BS,2]
        logits_IDH = self._fc2_IDH_1(encoded_IDH)  # [BS,2]
        logits_1p19q = self._fc2_1p19q_1(encoded_1p19q)  # [BS,2]
        mark_output = torch.cat((encoded_IDH, encoded_CDKN, encoded_1p19q), dim=1)
        results_dict = {'logits_IDH': logits_IDH, 'logits_1p19q': logits_1p19q, 'logits_CDKN': logits_CDKN}
        
        return results_dict, saliency_IDH_wt, saliency_IDH_mut, saliency_1p19q_codel, saliency_CDKN_HOMDEL, encoded_IDH, encoded_1p19q, encoded_CDKN,mark_output

    def calculateLoss_Graph(self,encoded_IDH,encoded_1p19q,encoded_CDKN):

        dis_IDH_IDH = F.cosine_similarity(encoded_IDH, encoded_IDH, dim=1)
        dis_IDH_1p19 = F.cosine_similarity(encoded_IDH, encoded_1p19q, dim=1)
        dis_IDH_CDKN = F.cosine_similarity(encoded_IDH, encoded_CDKN, dim=1)
        dis_1p19_IDH = F.cosine_similarity(encoded_1p19q, encoded_IDH, dim=1)
        dis_1p19_1p19 = F.cosine_similarity(encoded_1p19q, encoded_1p19q, dim=1)
        dis_1p19_CDKN = F.cosine_similarity(encoded_1p19q, encoded_CDKN, dim=1)
        dis_CDKN_IDH = F.cosine_similarity(encoded_CDKN, encoded_IDH, dim=1)
        dis_CDKN_1p19 = F.cosine_similarity(encoded_CDKN, encoded_1p19q, dim=1)
        dis_CDKN_CDKN = F.cosine_similarity(encoded_CDKN, encoded_CDKN, dim=1)

        cos_dis_matrix = [dis_IDH_IDH, dis_IDH_1p19, dis_IDH_CDKN, dis_1p19_IDH, dis_1p19_1p19, dis_1p19_CDKN,
                          dis_CDKN_IDH,
                          dis_CDKN_1p19, dis_CDKN_CDKN]

        adj_T = self.adj.T
        adj = (adj_T + self.adj) / 2
        adj = torch.from_numpy(np.array(adj)).float().cuda(self.opt['gpus'][0])
        adj = torch.unsqueeze(adj, dim=0)
        adj = adj.repeat(dis_IDH_IDH.detach().cpu().numpy().shape[0], 1, 1)

        dis_1p19_CDKN = dis_1p19_CDKN.detach().cpu().numpy()  # [BS]
        dis_1p19_CDKN_FLAG = np.ones(dis_IDH_IDH.detach().cpu().numpy().shape[0], dtype=float)
        for i in range(dis_IDH_IDH.detach().cpu().numpy().shape[0]):
            if dis_1p19_CDKN[i] < 0.1:
                dis_1p19_CDKN_FLAG[i] = 0
        dis_1p19_CDKN_FLAG = torch.from_numpy(np.array(dis_1p19_CDKN_FLAG)).float().cuda(self.opt['gpus'][0])

        dis_CDKN_1p19 = dis_CDKN_1p19.detach().cpu().numpy()  # [BS]
        dis_CDKN_1p19_FLAG = np.ones(dis_IDH_IDH.detach().cpu().numpy().shape[0], dtype=float)
        for i in range(dis_IDH_IDH.detach().cpu().numpy().shape[0]):
            if dis_CDKN_1p19[i] < 0.1:
                dis_CDKN_1p19_FLAG[i] = 0
        dis_CDKN_1p19_FLAG = torch.from_numpy(np.array(dis_CDKN_1p19_FLAG)).float().cuda(self.opt['gpus'][0])

        self.loss_Graph = (cos_dis_matrix[0] - adj[:, 0, 0]) ** 2 + (cos_dis_matrix[1] - adj[:, 0, 1]) ** 2 + (
                    cos_dis_matrix[2] - adj[:, 0, 2]) ** 2 \
                          + (cos_dis_matrix[3] - adj[:, 1, 0]) ** 2 + (
                                      cos_dis_matrix[4] - adj[:, 1, 1]) ** 2 + dis_1p19_CDKN_FLAG * (
                                      cos_dis_matrix[5] - adj[:, 1, 2]) ** 2 \
                          + (cos_dis_matrix[6] - adj[:, 2, 0]) ** 2 + dis_CDKN_1p19_FLAG * (
                                      cos_dis_matrix[7] - adj[:, 2, 1]) ** 2 + (cos_dis_matrix[8] - adj[:, 2, 2]) ** 2
        return torch.mean(self.loss_Graph)



    def calculateLoss_IDH(self, pred, label):
        self.loss_IDH = self.criterion_ce_IDH(pred, label)
        return self.loss_IDH

    def calculateLoss_1p19q(self, pred, label):
        self.loss_1p19q = self.criterion_ce_1p19q(pred, label)
        return self.loss_1p19q


    def calculateLoss_CDKN(self, pred, label):
        valid_indices = label != -1 
        if valid_indices.sum() == 0:
            return 0 
        valid_pred = pred[valid_indices]
        valid_label = label[valid_indices]
        self.loss_CDKN = self.criterion_ce_CDKN(valid_pred, valid_label)
        return self.loss_CDKN

class Mine_His(nn.Module):
    def __init__(self, opt, vis=False):
        super(Mine_His, self).__init__()
        self.opt = opt
        self.vis = vis
        self.size = [1024, 512]
        self.n_classes_His = 4
        self.layer_His = nn.ModuleList()
        for _ in range(self.opt['Network']['His_layers']):
            layer = Block(opt, self.size[1], vis)
            self.layer_His.append(copy.deepcopy(layer))
        self.encoder_norm_His = LayerNorm(self.size[1], eps=1e-6)
        self.criterion_ce_His = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([4, 3, 2.5, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        self.criterion_ce_His_2class = nn.CrossEntropyLoss().cuda(opt['gpus'][0])
        self.criterion_mse_diag = nn.MSELoss().cuda(opt['gpus'][0])
        self.criterion_ce_Grade = nn.CrossEntropyLoss(
            weight=torch.from_numpy(np.array([2.3, 2.3, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
    def forward(self, hidden_states):
        attn_weights_His = []
        count = 0
        for layer_block in self.layer_His:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights_His.append(weights)
        encoded_His = self.encoder_norm_His(hidden_states)  # [B,2500,512]

        return hidden_states, encoded_His
        # encoded_His = self.encoder_norm_His(hidden_states)
        # logits_His = self._fc2_His(torch.mean(encoded_His, dim=1))
        # results_dict = {'logits': logits_His}
        # return hidden_states,results_dict

    def calculateLoss_His(self, pred, label):
        self.loss_His = self.criterion_ce_His(pred, label)
        return self.loss_His
    def calculateLoss_His_2class(self, pred, label):
        self.loss_His_2_class = self.criterion_ce_His_2class(pred, label)
        return self.loss_His_2_class
    def calculateLoss_diag(self, pred, label):
        self.loss_diag = self.criterion_mse_diag(pred, label)
        return self.loss_diag
    def calculateLoss_Grade(self, pred, label):
        self.loss_Grade = self.criterion_ce_Grade(pred, label)
        return self.loss_Grade

class Mine_Grade(nn.Module):
    def __init__(self, opt, vis=False):
        super(Mine_Grade, self).__init__()
        self.opt = opt
        self.vis = vis
        self.size = [1024, 512]
        self.n_classes_Grade = 3
        self.layer_Grade = nn.ModuleList()
        for _ in range(self.opt['Network']['Grade_layers']):
            layer = Block(opt, self.size[1], vis)
            self.layer_Grade.append(copy.deepcopy(layer))
        self.encoder_norm_Grade = LayerNorm(self.size[1], eps=1e-6)
        self._fc2_Grade = nn.Linear(self.size[1], self.n_classes_Grade)
        self.criterion_ce_Grade = nn.CrossEntropyLoss(
            weight=torch.from_numpy(np.array([3.6, 4.8, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        self.criterion_focal_Grade = FocalLoss(alpha=1).cuda(opt['gpus'][0])


    def forward(self, hidden_states):
        attn_weights_Grade = []
        for layer_block in self.layer_Grade:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights_Grade.append(weights)
        encoded_Grade = self.encoder_norm_Grade(hidden_states)  # [B,2500,512]
        return encoded_Grade
        # encoded_Grade = self.encoder_norm_Grade(hidden_states)
        # logits_Grade = self._fc2_Grade(torch.mean(encoded_Grade, dim=1))
        # results_dict = {'logits': logits_Grade}
        # return results_dict

    def calculateLoss_Grade(self, pred, label):
        self.loss_Grade = self.criterion_ce_Grade(pred, label)
        return self.loss_Grade

class Cls_His_Grade(nn.Module):
    def __init__(self, opt, vis=False):
        super(Cls_His_Grade, self).__init__()
        self.opt = opt
        self.n_classes_Grade = 3
        self.n_classes_His = 4

        # self._fc2_His_1 = nn.Linear(self.opt['fixdim'], self.n_classes_His)
        # self._fc2_His_2class = nn.Linear(self.opt['fixdim'], 2)
        #### His
        self.attention_His = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.attention_V_His = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh()
        )
        self.attention_U_His = nn.Sequential(
            nn.Linear(512, 128),
            nn.Sigmoid()
        )
        self.attention_weights_His = nn.Linear(128, 1)

        #### His_2class
        self.attention_His_2class = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.attention_V_His_2class = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh()
        )
        self.attention_U_His_2class = nn.Sequential(
            nn.Linear(512, 128),
            nn.Sigmoid()
        )
        self.attention_weights_His_2class = nn.Linear(128, 1)

        #### Grade
        self.attention_Grade = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.attention_V_Grade = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh()
        )
        self.attention_U_Grade = nn.Sequential(
            nn.Linear(512, 128),
            nn.Sigmoid()
        )
        self.attention_weights_Grade = nn.Linear(128, 1)

        self._fc2_His_1 = nn.Linear(512, self.n_classes_His)
        self._fc2_His_2class = nn.Linear(512, 2)
        self._fc2_Grade = nn.Linear(512, self.n_classes_Grade)


    def forward(self, encoded_His):
        encoded_His_2class = encoded_His
        encoded_His_2class_ori = encoded_His_2class
        A_V_His_2class = self.attention_V_His_2class(encoded_His_2class)  # BxNx128
        A_U_His_2class = self.attention_U_His_2class(encoded_His_2class)  # BxNx128
        A_encoded_His_2class = self.attention_weights_His_2class(A_V_His_2class * A_U_His_2class)  # BxNx1
        A_encoded_His_2class = F.softmax(A_encoded_His_2class, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded_His_2class.shape[0]):
            if i == 0:
                Final_con_layer = encoded_His_2class[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_His_2class[i], 1).expand(-1, encoded_His_2class[i].shape[
                    1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_His_2class = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_His_2class_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_His_2class[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_His_2class[i], 1).expand(-1, encoded_His_2class[i].shape[
                    1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_His_2class = torch.cat(
                    (Final_con_layer_His_2class, torch.unsqueeze(Final_con_layer, 0)),
                    dim=0)  # BSxNx512
                encoded_His_2class_new = torch.cat(
                    (encoded_His_2class_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_His_2class = encoded_His_2class_new  # Bx512
        # encoded_His = torch.mean(encoded_His, dim=2)  # [BS,2500]
        # logits_His = self._fc2_His_1(encoded_His)  # [BS,2]
        # weight_His_GBM = self._fc2_His_1.weight[3]  # [2500]

        encoded_Grade = encoded_His
        encoded_Grade_ori = encoded_Grade
        A_V_Grade = self.attention_V_Grade(encoded_Grade)  # BxNx128
        A_U_Grade = self.attention_U_Grade(encoded_Grade)  # BxNx128
        A_encoded_Grade = self.attention_weights_Grade(A_V_Grade * A_U_Grade)  # BxNx1
        A_encoded_Grade = F.softmax(A_encoded_Grade, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded_Grade.shape[0]):
            if i == 0:
                Final_con_layer = encoded_Grade[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_Grade[i], 1).expand(-1, encoded_Grade[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_Grade = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_Grade_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_Grade[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_Grade[i], 1).expand(-1, encoded_Grade[i].shape[
                    1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_Grade = torch.cat(
                    (Final_con_layer_Grade, torch.unsqueeze(Final_con_layer, 0)),
                    dim=0)  # BSxNx512
                encoded_Grade_new = torch.cat(
                    (encoded_Grade_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_Grade = encoded_Grade_new  # Bx512

        encoded_His_ori = encoded_His
        A_V_His = self.attention_V_His(encoded_His)  # BxNx128
        A_U_His = self.attention_U_His(encoded_His)  # BxNx128
        A_encoded_His = self.attention_weights_His(A_V_His * A_U_His)  # BxNx1
        A_encoded_His = F.softmax(A_encoded_His, dim=1)[..., 0]  # BxN AMIL attention map
        for i in range(encoded_His.shape[0]):
            if i == 0:
                Final_con_layer = encoded_His[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_His[i], 1).expand(-1, encoded_His[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_His = torch.unsqueeze(Final_con_layer, 0)  # 1xNx512
                encoded_His_new = torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)  # 1x512
            else:
                Final_con_layer = encoded_His[i]  # Nx512
                saliency_map = torch.unsqueeze(A_encoded_His[i], 1).expand(-1, encoded_His[i].shape[1])  # Nx512
                Final_con_layer = Final_con_layer * saliency_map  # Nx512
                Final_con_layer_His = torch.cat((Final_con_layer_His, torch.unsqueeze(Final_con_layer, 0)),
                                                dim=0)  # BSxNx512
                encoded_His_new = torch.cat(
                    (encoded_His_new, torch.unsqueeze(torch.sum(Final_con_layer, dim=0), dim=0)), 0)
        encoded_His = encoded_His_new  # Bx512

        



        # logits_His_2class=self._fc2_His_2class(encoded_His)  # [BS,2]
        # weight_His_GBM_Cls2 = self._fc2_His_2class.weight[1]  # [2500]
        logits_His=self._fc2_His_1(encoded_His)
        logits_His_2class=self._fc2_His_2class(encoded_His_2class)
        logits_Grade=self._fc2_Grade(encoded_Grade)
        ####################saliency maps for his
        weight_A = torch.unsqueeze(self._fc2_His_1.weight[1], dim=1)  # [512,1]
        weight_O = torch.unsqueeze(self._fc2_His_1.weight[2], dim=1)  # [512,1]
        weight_GBM = torch.unsqueeze(self._fc2_His_1.weight[3], dim=1)  # [512,1]
        saliency_A = torch.matmul(Final_con_layer_His, weight_A)[..., 0]  # [BSxN]
        saliency_O = torch.matmul(Final_con_layer_His, weight_O)[..., 0]  # [BSxN]
        saliency_GBM = torch.matmul(Final_con_layer_His, weight_GBM)[..., 0]  # [BSxN]
        if self._fc2_His_1.bias is not None:
            saliency_A = saliency_A + self._fc2_His_1.bias[1] / encoded_His_ori.shape[1]  # [BSxN]
            saliency_O = saliency_O + self._fc2_His_1.bias[2] / encoded_His_ori.shape[1]  # [BSxN]
            saliency_GBM = saliency_GBM + self._fc2_His_1.bias[3] / encoded_His_ori.shape[1]  # [BSxN]

        ####################saliency maps for his_2class
        weight_GBM_2class = torch.unsqueeze(self._fc2_His_2class.weight[1], dim=1)  # [512,1]
        saliency_GBM_2class = torch.matmul(Final_con_layer_His_2class, weight_GBM_2class)[..., 0]  # [BSxN]
        if self._fc2_His_2class.bias is not None:
            saliency_GBM_2class = saliency_GBM_2class + self._fc2_His_2class.bias[1] / encoded_His_2class_ori.shape[1]  # [BSxN]
        results_dict = {'logits_His': logits_His,'logits_His_2class': logits_His_2class,'logits_Grade':logits_Grade}
        return  results_dict,saliency_GBM,saliency_GBM_2class,saliency_O,encoded_His_2class


    def Loss_mutual_correlation(self,weight_IDH_wt,weight_His_GBM,weight_1p19q_codel,weight_His_O,epoch,IDH_only=False):
        weight_IDH_wt = weight_IDH_wt.tolist()
        weight_His_GBM = weight_His_GBM.tolist()
        x = weight_IDH_wt
        b = sorted(enumerate(x), key=lambda x: x[1], reverse=True)
        Index_IDH_wt = [x[0] for x in b]
        x = weight_His_GBM
        b = sorted(enumerate(x), key=lambda x: x[1], reverse=True)
        Index_His_GBM = [x[0] for x in b]

        if not IDH_only:
            weight_1p19q_codel = weight_1p19q_codel.tolist()
            weight_His_O = weight_His_O.tolist()
            x = weight_1p19q_codel
            b = sorted(enumerate(x), key=lambda x: x[1], reverse=True)
            Index_1p19q_codel = [x[0] for x in b]
            x = weight_His_O
            b = sorted(enumerate(x), key=lambda x: x[1], reverse=True)
            Index_His_O = [x[0] for x in b]

        self.opt['Network']['top_K_patch'] = int(self.opt['fixdim'] / 3)
        top_K_patch = int(self.opt['Network']['top_K_patch'] * (0.85 ** (int(epoch / 10))))
        #### IDH-wt  **** GBM
        loss_IDH_GBM = 0
        for i in range(top_K_patch):
            index_patch_low = Index_IDH_wt[i]
            if i <= int(self.opt['Network']['top_K_patch'] / 2):
                target_low_index_list = Index_His_GBM[0:self.opt['Network']['top_K_patch']]
            else:
                target_low_index_list = Index_His_GBM[i - int(self.opt['Network']['top_K_patch'] / 2):i + int(
                    self.opt['Network']['top_K_patch'] / 2)]
            if not index_patch_low in target_low_index_list:
                loss_IDH_GBM += 1
        loss_IDH_GBM = loss_IDH_GBM / top_K_patch
        loss_GBM_IDH = 0
        for i in range(top_K_patch):
            index_patch_low = Index_His_GBM[i]
            if i <= int(self.opt['Network']['top_K_patch'] / 2):
                target_low_index_list = Index_IDH_wt[0:self.opt['Network']['top_K_patch']]
            else:
                target_low_index_list = Index_IDH_wt[i - int(self.opt['Network']['top_K_patch'] / 2):i + int(
                    self.opt['Network']['top_K_patch'] / 2)]
            if not index_patch_low in target_low_index_list:
                loss_GBM_IDH += 1
        loss_GBM_IDH = loss_GBM_IDH / top_K_patch
        loss_IDH_GBM = (loss_GBM_IDH + loss_IDH_GBM) / 2
        #### 1p19q codel  **** O
        if not IDH_only:
            loss_1p19q_O = 0
            for i in range(top_K_patch):
                index_patch_low = Index_1p19q_codel[i]
                if i <= int(self.opt['Network']['top_K_patch'] / 2):
                    target_low_index_list = Index_His_O[0:self.opt['Network']['top_K_patch']]
                else:
                    target_low_index_list = Index_His_O[i - int(self.opt['Network']['top_K_patch'] / 2):i + int(
                        self.opt['Network']['top_K_patch'] / 2)]
                if not index_patch_low in target_low_index_list:
                    loss_1p19q_O += 1
            loss_1p19q_O = loss_1p19q_O / top_K_patch
            loss_O_1p19q = 0
            for i in range(top_K_patch):
                index_patch_low = Index_His_O[i]
                if i <= int(self.opt['Network']['top_K_patch'] / 2):
                    target_low_index_list = Index_1p19q_codel[0:self.opt['Network']['top_K_patch']]
                else:
                    target_low_index_list = Index_1p19q_codel[i - int(self.opt['Network']['top_K_patch'] / 2):i + int(
                        self.opt['Network']['top_K_patch'] / 2)]
                if not index_patch_low in target_low_index_list:
                    loss_O_1p19q += 1
            loss_O_1p19q = loss_O_1p19q / top_K_patch
            loss_1p19q_O = (loss_O_1p19q + loss_1p19q_O) / 2

        if IDH_only:
            loss_mutual_correlation = loss_IDH_GBM
        else:
            loss_mutual_correlation = loss_IDH_GBM + loss_1p19q_O
        loss_mutual_correlation = torch.from_numpy(np.array([loss_mutual_correlation])).cuda(self.opt['gpus'][0])[0]
        return loss_mutual_correlation

import argparse, time, random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
if __name__ == "__main__":


    import argparse
    # import h5py
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--opt', type=str, default='./config/mine.yml')
    # args = parser.parse_args()
    # with open(args.opt) as f:
    #     opt = yaml.load(f, Loader=SafeLoader)
    # setup_seed(opt['seed'])
    # gpuID = opt['gpus']
    # res_init = Mine_init(opt).cuda(opt['gpus'][0])
    # res_IDH=Mine_IDH(opt).cuda(opt['gpus'][0])
    # res_1p19q = Mine_1p19q(opt).cuda(opt['gpus'][0])
    # res_CDKN = Mine_CDKN(opt).cuda(opt['gpus'][0])
    # res_Graph = Label_correlation_Graph(opt).cuda(opt['gpus'][0])
    # res_His = Mine_His(opt).cuda(opt['gpus'][0])
    # # res_Grade = Mine_Grade(opt).cuda(opt['gpus'][0])
    # res_Cls_His_Grade = Cls_His_Grade(opt).cuda(opt['gpus'][0])
    # #
    # init_weights(res_init, init_type='xavier', init_gain=1)
    # init_weights(res_IDH, init_type='xavier', init_gain=1)
    # init_weights(res_1p19q, init_type='xavier', init_gain=1)
    # init_weights(res_CDKN, init_type='xavier', init_gain=1)
    # init_weights(res_His, init_type='xavier', init_gain=1)
    # # init_weights(res_Grade, init_type='xavier', init_gain=1)
    # device = torch.device('cuda:{}'.format(opt['gpus'][0]))
    # res_init = torch.nn.DataParallel(res_init, device_ids=gpuID)
    # res_IDH = torch.nn.DataParallel(res_IDH, device_ids=gpuID)
    # res_1p19q = torch.nn.DataParallel(res_1p19q, device_ids=gpuID)
    # res_CDKN = torch.nn.DataParallel(res_CDKN, device_ids=gpuID)
    # res_Graph = torch.nn.DataParallel(res_Graph, device_ids=gpuID)
    # res_His = torch.nn.DataParallel(res_His, device_ids=gpuID)
    # res_Cls_His_Grade = torch.nn.DataParallel(res_Cls_His_Grade, device_ids=gpuID)
    # # res_His.to(device)
    # # res_Grade.to(device)
    # # res_Cls_His_Grade.to(device)
    # #
    # input1 = torch.ones((4, 2500,1024)).cuda(opt['gpus'][0])
    # # root = opt['dataDir'] + 'Res50_feature_2500_fixdim0/'
    # # # root=r'D:\PhD\Project_WSI\data\Res50_feature_2500/'
    # # patch_all0 =torch.from_numpy(np.array(h5py.File(root + 'TCGA-DU-A5TY-01Z-00-DX1.h5')['Res_feature'][:])).float().cuda(opt['gpus'][0])# (1,N,1024)
    # # patch_all1=torch.from_numpy(np.array(h5py.File(root + 'TCGA-HT-8104-01A-01-TS1.h5')['Res_feature'][:])).float().cuda(opt['gpus'][0])# (1,N,1024)
    # # patch_all2 = torch.from_numpy(np.array(h5py.File(root + 'TCGA-CS-6188-01A-01-BS1.h5')['Res_feature'][:])).float().cuda(opt['gpus'][0])  # (1,N,1024)
    # # patch_all3 = torch.from_numpy(np.array(h5py.File(root + 'TCGA-DU-7010-01Z-00-DX1.h5')['Res_feature'][:])).float().cuda(opt['gpus'][0])  # (1,N,1024)
    # # input1 = torch.cat((patch_all0, patch_all1, patch_all2,patch_all3), 0)  # [4,N,1024]
    # #
    # hidden_states_init = res_init(input1)
    # #
    # hidden_states, encoded_IDH=res_IDH(hidden_states_init)
    # hidden_states, encoded_1p19q = res_1p19q(hidden_states)
    # encoded_CDKN = res_CDKN(hidden_states)
    # # # a_max = np.max(hidden_states.detach().numpy()[0])
    # # # a_min = np.min(hidden_states.detach().numpy()[0])
    # #
    # out,weight_IDH_wt,weight_IDH_mut,weight_1p19q_codel,weight_CDKN_HOMDEL,encoded_IDH0, encoded_1p19q0, encoded_CDKN0 = res_Graph(encoded_IDH,encoded_1p19q,encoded_CDKN)
    # # loss_IDH = res_Graph.calculateLoss_IDH(out['logits_IDH'], torch.from_numpy(np.array([1,1,1,1])).cuda(opt['gpus'][0]))
    # loss_Graph=res_Graph.module.calculateLoss_Graph(encoded_IDH0, encoded_1p19q0, encoded_CDKN0)
    # hidden_states, encoded_His = res_His(hidden_states_init)
    # # encoded_Grade=res_Grade(hidden_states)
    # out,weight_His_GBM,weight_His_O=res_Cls_His_Grade(encoded_His)
    # weight_IDH_wt=weight_IDH_wt[0:int(weight_IDH_wt.detach().cpu().numpy().shape[0]/len(gpuID))]
    # weight_1p19q_codel = weight_1p19q_codel[0:int(weight_1p19q_codel.detach().cpu().numpy().shape[0] / len(gpuID))]
    # weight_His_GBM = weight_His_GBM[0:int(weight_His_GBM.detach().cpu().numpy().shape[0] / len(gpuID))]
    # loss_mutual_correlation=res_Cls_His_Grade.module.Loss_mutual_correlation(weight_IDH_wt, weight_1p19q_codel, weight_His_GBM, 0)
    # a=1
































