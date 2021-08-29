import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from torch.utils import data
from torch.nn.parameter import Parameter
from torch.autograd import Variable


class ArcFaceNetloss(nn.Module):
    def __init__(self, feature_dim, cls_num):
        super(ArcFaceNetloss, self).__init__()
        self.W = nn.Parameter(torch.randn(feature_dim, cls_num))  # (V,C)
        # self.ce = nn.CrossEntropyLoss()

    def forward(self, feature, label=None, m=0.1, s=1):
        # 特征向量x与权重向量w 归一化
        x = F.normalize(feature, dim=1)  # x是（N，V）结构   # N是batchsize，V是特征的维度，C是代表类别数
        w = F.normalize(self.W, dim=0)   # W是（V,C）结构
        # 特征向量与参数向量的夹角theta，分子numerator，分母denominator
        cos = torch.matmul(x, w) / 10  # （N,C）   # /10防止下溢
        theta = torch.acos(cos)  # (N,C)
        numerator = torch.exp(s * torch.cos(theta + m))  # (N,C)
        denominator = torch.sum(torch.exp(s * torch.cos(theta)), dim=1, keepdim=True) - torch.exp(
            s * torch.cos(theta)) + numerator  # 第一项(N,1)  keepdim=True保持形状不变.这是我们原有的softmax的分布。第二项(N,C),最后结果是(N,C)
        out = torch.log(numerator /  denominator)  # (N,C)
        # loss = nn.NLLLoss(out, label)
        return out

# Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)
class AAMsoftmaxloss(nn.Module):
    def __init__(self, feature_dim, cls_num, margin=0.1, scale=50, easy_margin=False, **kwargs):
        super(AAMsoftmaxloss, self).__init__()
        self.test_normalize = True
        self.in_feats = feature_dim
        self.m = margin
        self.s = scale
        self.ce = nn.CrossEntropyLoss()
        self.weight = torch.nn.Parameter(torch.FloatTensor(cls_num, feature_dim), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        # print('Initialised AAMSoftmax margin %.3f scale %.3f'%(self.m,self.s))

    def forward(self, x, label=None):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        loss = self.ce(output, label)
        # prec1   = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss

# Adapted from https://github.com/CoinCheung/pytorch-loss (MIT License)
class AMsoftmaxloss(nn.Module):
    def __init__(self,feature_dim, cls_num, margin=0.1, scale=70, **kwargs):
        super(AMsoftmaxloss, self).__init__()
        self.test_normalize = True
        self.in_feats = feature_dim
        self.W = torch.nn.Parameter(torch.randn(feature_dim, cls_num), requires_grad=True)
        self.m = margin
        self.s = scale
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)
        # print('Initialised AMSoftmax m=%.3f s=%.3f'%(self.m,self.s))

    def forward(self, x, label=None):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)
        if label_view.is_cuda: label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss    = self.ce(costh_m_s, label)
        # prec1   = accuracy(costh_m_s.detach(), label.detach(), topk=(1,))[0]
        return loss

class AngularProtoLoss(nn.Module):
    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
        super(AngularProtoLoss, self).__init__()
        self.test_normalize = True
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion  = torch.nn.CrossEntropyLoss()
        print('Initialised AngleProto')

    def forward(self, x, label=None):
        assert x.size()[1] >= 2
        out_anchor      = torch.mean(x[:,1:,:],1)
        out_positive    = x[:,0,:]
        stepsize        = out_anchor.size()[0]
        cos_sim_matrix  = F.cosine_similarity(out_positive.unsqueeze(-1), out_anchor.unsqueeze(-1).transpose(0,2))
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        label   = torch.from_numpy(np.asarray(range(0,stepsize))).cuda()
        nloss   = self.criterion(cos_sim_matrix, label)
        # prec1   = accuracy(cos_sim_matrix.detach(), label.detach(), topk=(1,))[0]
        return nloss

class CenterLoss(nn.Module):
    def __init__(self, feature_dim, cls_num):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(cls_num, feature_dim))

    def forward(self, features, labels, reduction='mean'):
        # 特征向量归一化
        _features = F.normalize(features)
        centers_batch = self.centers.index_select(dim=0, index=labels.long())
        # 根据论文《A Discriminative Feature Learning Approach for Deep Face Recognition》修改如下
        if reduction == 'sum':  # 返回loss的和
            return torch.sum(torch.pow(_features - centers_batch, 2)) / 2
        elif reduction == 'mean':  # 返回loss和的平均值，默认为mean方式
            return torch.sum(torch.pow(_features - centers_batch, 2)) / 2 / len(features)
        else:
            raise ValueError("ValueError: {0} is not a valid value for reduction".format(reduction))

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.pdist = nn.PairwiseDistance()

    def forward(self, positive_pairs, negative_pairs, margin):
        ## POSITIVE PART
        faces1, voices1 = positive_pairs
        dists_pos = self.pdist(faces1, voices1)
        pos_part = dists_pos ** 2
        # NEGATIVE PART
        faces2, voices2 = negative_pairs
        dists_neg = self.pdist(faces2, voices2)
        neg_part = (margin - dists_neg).clamp(0) ** 2

        # TBoard.add_scalar('Train/pos_part_sum', pos_part.sum().item(), step_num)
        # TBoard.add_scalar('Train/neg_part_sum', neg_part.sum().item(), step_num)
        # TBoard.add_scalar('Train/dists_neg_mean', dists_neg.mean().item(), step_num)
        # TBoard.add_scalar('Train/dists_pos_mean', dists_pos.mean().item(), step_num)
        # TBoard.add_scalar('Train/faces2_mean', faces2.mean().item(), step_num)
        # TBoard.add_scalar('Train/voices2_mean', voices2.mean().item(), step_num)
        # TBoard.add_scalar('Train/faces1_mean', faces1.mean().item(), step_num)
        # TBoard.add_scalar('Train/voices1_mean', voices1.mean().item(), step_num)
        # TBoard.add_scalar('Train/faces2_voices2_mean', (faces2 - voices2).mean().item(), step_num)
        # TBoard.add_scalar('Train/voices1_voices2_mean', (voices1 - voices2).mean().item(), step_num)

        ## CALCULATE LOSS
        B, D = faces1.size()
        batch_loss = pos_part.sum() + neg_part.sum()
        batch_loss /= B + B
        return batch_loss