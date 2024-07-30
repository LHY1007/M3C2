import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import basic_net as basic_net
from torch.nn.modules.utils import _pair
from scipy import ndimage


def init_weights(net, init_type='xavier', init_gain=1):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
def to_device(net, device='cpu', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(device)
        # net = torch.nn.DataParallel(net, device_ids=gpu_ids)
        # net = torch.nn.parallel.DistributedDataParallel(net)  # multi-GPUs
    # init_weights(net, init_type, init_gain=init_gain)
    return net


def get_scheduler(optimizer, n_ep, n_ep_decay, decayType='step', cur_ep=-1):
    '''
    :param optimizer:
    :param decayType:
    :param n_ep: number of ep for training
    :param n_ep_decay: #ep for begining schduling
    :param cur_ep:
    :return:
    '''
    lr_policy = decayType
    if lr_policy == 'linear':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - n_ep_decay) / float(n_ep - n_ep_decay + 1)
            return lr_l
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
    elif lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_ep_decay, gamma=0.5, last_epoch=cur_ep)
    elif lr_policy == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96, last_epoch=cur_ep)
    else:
        return NotImplementedError('no such learn rate policy')
    return scheduler
from torch.optim.lr_scheduler import LambdaLR
import math
class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

def set_requires_grad(nets, requires_grad=False):
    """Set requires_grad=False for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def save(self, filename, ep, total_it):
    state = {
        'precodec': self.precodec.state_dict(),
        'precodec_opt': self.precodec_opt.state_dict(),
        'ep': ep,
        'total_it': total_it
    }
    torch.save(state, filename)

class Res50_pretrain(nn.Module):
    def __init__(self):
        super(Res50_pretrain, self).__init__()
        self.resNet1 = basic_net.resnet50(pretrained=True)
        self.resNet = list(self.resNet1.children())[:-3]
        self.features = nn.Sequential(*self.resNet)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for param in self.parameters():
            param.requires_grad = False
    def forward(self,x):
        """
                x: shape (N, 3 ,224, 224)
                """
        x = self.features(x)
        x = self.avgpool(x)  # (N/n, 1024 ,1, 1)
        x = torch.squeeze(x)# (N/n, 1024)
        return x

class Res34_pretrain(nn.Module):
    def __init__(self):
        super(Res34_pretrain, self).__init__()
        self.resNet1 = basic_net.resnet34(pretrained=True)
        self.resNet = list(self.resNet1.children())[:-2]
        self.features = nn.Sequential(*self.resNet)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for param in self.parameters():
            param.requires_grad = False
    def forward(self,x):
        """
                x: shape (N, 3 ,224, 224)
                """
        x = self.features(x)
        x = self.avgpool(x)  # (N, 3 ,224, 224)
        x = torch.squeeze(x)
        # x = torch.unsqueeze(x, dim=0)
        return x

class Res18_pretrain(nn.Module):
    def __init__(self):
        super(Res18_pretrain, self).__init__()
        self.resNet1 = basic_net.resnet18(pretrained=True)
        self.resNet = list(self.resNet1.children())[:-2]
        self.features = nn.Sequential(*self.resNet)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for param in self.parameters():
            param.requires_grad = False
    def forward(self,x):
        """
                x: shape (N, 3 ,224, 224)
                """
        x = self.features(x)
        x = self.avgpool(x)  # (N, 3 ,224, 224)
        x = torch.squeeze(x)
        # x = torch.unsqueeze(x, dim=0)
        return x
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}
class Mlp(nn.Module):
    def __init__(self,opt,hidden_size):
        super(Mlp, self).__init__()
        self.fc1 = Linear(hidden_size, hidden_size*4)
        self.fc2 = Linear(hidden_size*4, hidden_size)
        self.act_fn = ACT2FN["relu"]
        self.dropout = Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, opt, hidden_size,vis):
        super(Block, self).__init__()
        self.opt=opt
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(opt,hidden_size)
        self.attn = basic_net.NystromAttention(
            dim=hidden_size,
            dim_head=hidden_size // 16,
            heads=16,
            num_landmarks=hidden_size // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        h = x

        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h
        if self.opt['Network']['Trans_block']=='simple':
            return x, None
        else:
            h = x
            x = self.ffn_norm(x)
            x = self.ffn(x)
            x = x + h
            return x, None

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.adj = np.array([[1, 0.4038, 0.3035], [1, 1, 0.1263], [0.2595, 0.0436, 1]])
        self.adj_1 = self.adj + np.multiply(self.adj.T, self.adj.T > self.adj) - np.multiply(self.adj,self.adj.T > self.adj)
        self.adj_1 = Parameter(torch.from_numpy(np.array(self.adj_1)).float(),requires_grad=False)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """
       input #[BS,2500,512,3 ]
       adj #[3,3 ]
        """
        input = torch.transpose(input, 0, 3)#[3,2500,512,BS ]
        input = torch.transpose(input, 2, 3)  # [3,2500,BS,512 ]
        support = torch.matmul(input, self.weight)# [3,2500,BS,512 ]

        support= torch.transpose(support, 0, 3)  # [512,2500,BS,3 ]
        output = torch.matmul( support,self.adj_1  )# [512,2500,BS,3  ]
        output= torch.transpose(output, 0, 3) # [3,2500,BS,512 ]
        if self.bias is not None:
            output = output + self.bias
            output = torch.transpose(output, 0, 2)  # [BS,2500,3,512 ]
            output = torch.transpose(output, 2, 3)  # [BS,2500,512,3 ]
            return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

class basic_model_stem(nn.Module):
    def __init__(self, opt):
        super(basic_model_stem, self).__init__()
        self.opt = opt
        self.model_general = self.opt['name'].split('_')[0]
        self.model_specific = self.opt['name'].split('_')[3]
        self.task = self.opt['name'].split('_')[1]

        if self.model_specific == 'res':
            self.resNet1 = basic_net.resnet18_stem(pretrained=True)
            self.resNet = list(self.resNet1.children())
            self.features = nn.Sequential(*self.resNet)

    def forward(self, x):
        if self.model_specific=='res':
            x = self.features(x) #512

        return x


class basic_model(nn.Module):
    def __init__(self, opt):
        super(basic_model, self).__init__()
        self.opt=opt
        self.model_general=self.opt['name'].split('_')[0]
        self.model_specific= self.opt['name'].split('_')[3]
        self.task = self.opt['name'].split('_')[1]
        if self.task=='IDH' or self.task=='1p19q' or self.task=='CDKN':
            self.n_classes = 2
        elif self.task=='Diag':
            self.n_classes = 4
        elif self.task=='His':
            self.n_classes = 2
        if self.task == '1p19q':
            self.criterion_ce = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 8])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        elif self.task == 'IDH':
            self.criterion_ce = nn.CrossEntropyLoss().cuda(opt['gpus'][0])
        elif self.task == 'CDKN':
            self.criterion_ce = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        elif self.task == 'Diag':
            self.criterion_ce = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([1, 4.19, 2.88, 3.64])).float().cuda(opt['gpus'][0])).cuda(
                opt['gpus'][0])
        elif self.task == 'Grade':
            self.criterion_ce = nn.CrossEntropyLoss(
                weight=torch.from_numpy(np.array([3.6, 4.8, 1])).float().cuda(opt['gpus'][0])).cuda(opt['gpus'][0])
        elif self.task == 'His':
            self.criterion_ce = nn.CrossEntropyLoss().cuda(opt['gpus'][0])

        if self.model_specific=='res':
            self.resNet1 = basic_net.resnet18(pretrained=True)
            for name, p in self.resNet1.named_parameters():
                if "conv1" in name :
                    p.requires_grad = False
            self.resNet = list(self.resNet1.children())[:-2]
            self.features = nn.Sequential(*self.resNet)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, self.n_classes)
            self.encoder_norm = LayerNorm(512, eps=1e-6)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, mode="fan_out",nonlinearity="sigmoid")
                    nn.init.constant_(m.bias, 0)
        elif self.model_specific=='incep':
            self.inception1 = basic_net.inception_v3(pretrained=True)
            for name, p in self.inception1.named_parameters():
                if "Conv2d_1a_3x3" in name :
                    p.requires_grad = False
            self.inception = list(self.inception1.children())[:-1]
            self.fc = nn.Linear(2048, self.n_classes)
            self.encoder_norm = LayerNorm(2048, eps=1e-6)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="sigmoid")
                    nn.init.constant_(m.bias, 0)
        elif self.model_specific=='dense':
            self.densenet1 = basic_net.densenet121(pretrained=True)
            for name, p in self.densenet1.named_parameters():
                if "conv0" in name or "norm0" in name or "relu0" in name or "pool0" in name:
                    p.requires_grad = False
            self.densenet = list(self.densenet1.children())[:-1]
            self.fc = nn.Linear(1024, self.n_classes)
            self.encoder_norm = LayerNorm(1024, eps=1e-6)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="sigmoid")
                    nn.init.constant_(m.bias, 0)
        elif self.model_specific == 'mna':
            self.mnasnet1 = basic_net.mnasnet0_5(pretrained=True)
            self.mnasnet = list(self.mnasnet1.children())[:-1]
            self.fc = nn.Sequential(nn.Dropout(p=0.2, inplace=True),nn.Linear(1280, self.n_classes))
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, mode="fan_out",nonlinearity="sigmoid")
                    nn.init.constant_(m.bias, 0)
        elif self.model_specific == 'alex':
            self.alex1 = basic_net.alexnet(pretrained=True)
            self.alex = list(self.alex1.children())[:-1]
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, self.n_classes),
            )


    def forward(self, x):
        if self.model_specific=='res':
            x = self.features(x) #512
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)# [N,512]
            x = self.encoder_norm(x)
            out = self.fc(x)# [N,2]
            out=torch.unsqueeze(torch.mean(out,dim=0), dim=0)
            a=1

        elif self.model_specific == 'incep':
            x = self.inception[0](x)
            x = self.inception[1](x)
            x = self.inception[2](x)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            x = self.inception[3](x)
            x = self.inception[4](x)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            x = self.inception[5](x)
            x = self.inception[6](x)
            x = self.inception[7](x)
            x = self.inception[8](x)
            x = self.inception[9](x)
            x = self.inception[10](x)
            x = self.inception[11](x)
            x = self.inception[12](x)
            x = self.inception[14](x)
            x = self.inception[15](x)
            x = self.inception[16](x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = F.dropout(x, training=self.training)
            x = x.view(x.size(0), -1)
            x = self.encoder_norm(x)
            out = self.fc(x)
            out = torch.unsqueeze(torch.mean(out, dim=0), dim=0)
            a = 1

        elif self.model_specific == 'dense':
            features = self.densenet[0](x)  # 512
            out = F.relu(features, inplace=True) # [N,512]
            out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
            out = self.encoder_norm(out)
            out = self.fc(out)
            out = torch.unsqueeze(torch.mean(out, dim=0), dim=0)
            a = 1

        elif self.model_specific == 'mna':
            x=self.mnasnet[0](x)
            x = x.mean([2, 3])
            out=self.fc(x)
            out = torch.unsqueeze(torch.mean(out, dim=0), dim=0)
            a = 1

        elif self.model_specific == 'alex':
            x = self.alex[0](x)  # 512
            x = self.avgpool(x)
            x = x.view(x.size(0),  256 * 6 * 6)  # [N,512]
            out = self.fc(x)  # [N,2]
            out = torch.unsqueeze(torch.mean(out, dim=0), dim=0)
            a = 1
        results_dict = {'logits': out}
        return results_dict

    def calculateLoss(self,pred,label):
        self.loss_ce = self.criterion_ce(pred, label)
        return self.loss_ce

if __name__ == "__main__":
    import argparse
    import yaml
    from yaml.loader import SafeLoader

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='./config/miccai.yml')
    args = parser.parse_args()
    with open(args.opt) as f:
        opt = yaml.load(f, Loader=SafeLoader)
    gpuID = opt['gpus']
    device = torch.device('cuda:{}'.format(gpuID[0])) if gpuID else torch.device('cpu')
    res = basic_model(opt).cuda(gpuID[0])

    res.to(device)
    input1 = torch.ones((8,200, 3, 224, 224)).cuda(gpuID[0])
    out = res(input1)