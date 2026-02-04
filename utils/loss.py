import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import math
import warnings
from torch.nn.modules import Module
from torch.nn import _reduction as _Reduction
# Recommend
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, ignore_index=ignore_index,
                                   reduction='mean')

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=0, weight=None, size_average=True, ignore_index=-1):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input1, target1):
        if input1.dim()>2:
            input1 = input1.contiguous().view(input1.size(0), input1.size(1), -1)
            input1 = input1.transpose(1,2)
            input1 = input1.contiguous().view(-1, input1.size(2)).squeeze()
        if target1.dim()==4:
            target1 = target1.contiguous().view(target1.size(0), target1.size(1), -1)
            target1 = target1.transpose(1,2)
            target1 = target1.contiguous().view(-1, target1.size(2)).squeeze()
        elif target1.dim()==3:
            target1 = target1.view(-1)
        else:
            target1 = target1.view(-1, 1)

        logpt = -F.cross_entropy(input1, target1, ignore_index=self.ignore_index)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1-pt)**self.gamma)*logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def weighted_BCE_logits1(logit_pixel, truth_pixel,weight_pos=0.25, weight_neg=0.75):
    logit = logit_pixel.view(-1)
    truth = truth_pixel.view(-1)
    assert (logit.shape == truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

    pos = (truth > 0.5).float()
    neg = (truth < 0.5).float()
    pos_num = pos.sum().item() + 1e-12
    neg_num = neg.sum().item() + 1e-12
    loss = (weight_pos * pos * loss / pos_num + weight_neg * neg * loss / neg_num).sum()

    return loss

def weighted_BCE_logits2(logit_pixel, truth_pixel,weight_pos=0.75, weight_neg=0.25):
    logit = logit_pixel.view(-1)
    truth = truth_pixel.view(-1)
    assert (logit.shape == truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

    pos = (truth > 0.5).float()
    neg = (truth < 0.5).float()
    pos_num = pos.sum().item() + 1e-12
    neg_num = neg.sum().item() + 1e-12
    loss = (weight_pos * pos * loss / pos_num + weight_neg * neg * loss / neg_num).sum()

    return loss

class FocalLoss2d_main(nn.Module):
    def __init__(self, gamma=0, weight=None, size_average=True,scale_factor=2, ignore_index=-1):
        super(FocalLoss2d_main, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        # EFL损失函数的超参数
        self.scale_factor = scale_factor

    def forward(self, input1, target1):
        if input1.dim()>2:
            input1 = input1.contiguous().view(input1.size(0), input1.size(1), -1)
            input1 = input1.transpose(1,2)
            input1 = input1.contiguous().view(-1, input1.size(2)).squeeze()
        if target1.dim()==4:
            target1 = target1.contiguous().view(target1.size(0), target1.size(1), -1)
            target1 = target1.transpose(1,2)
            target1 = target1.contiguous().view(-1, target1.size(2)).squeeze()
        elif target1.dim()==3:
            target1 = target1.view(-1)
        else:
            target1 = target1.view(-1, 1)

        # compute the negative likelyhood
        # weight = Variable(self.weight)

        pos1 = ((0.5 < target1) & (1.5 > target1)).float()
        pos2 = ((1.5 < target1) & (2.5 > target1)).float()
        pos3 = ((2.5 < target1) & (3.5 > target1)).float()
        pos4 = ((3.5 < target1) & (4.5 > target1)).float()
        pos5 = ((4.5 < target1) & (5.5 > target1)).float()
        pos6 = ((5.5 < target1) & (6.5 > target1)).float()

        pos = (target1 > 0.5).float()
        neg = (target1 < 0.5).float()

        pos1_num = pos1.sum().item()+ 1e-12
        pos2_num = pos2.sum().item()+ 1e-12
        pos3_num = pos3.sum().item()+ 1e-12
        pos4_num = pos4.sum().item()+ 1e-12
        pos5_num = pos5.sum().item()+ 1e-12
        pos6_num = pos6.sum().item()+ 1e-12
        pos_num = pos.sum().item()+ 1e-12
        neg_num = neg.sum().item()+ 1e-12
        all = pos_num+neg_num

        pos1_neg = (abs(neg_num-pos1_num) / (all+pos1_num))
        pos2_neg = (abs(neg_num-pos2_num) / (all+pos2_num))
        pos3_neg = (abs(neg_num-pos3_num) / (all+pos3_num))
        pos4_neg =  (abs(neg_num-pos4_num) / (all+pos4_num))
        pos5_neg = (abs(neg_num-pos5_num) / (all+pos5_num))
        pos6_neg =  (abs(neg_num-pos6_num) / (all+pos6_num))
        # neg_w = (abs(neg_num-pos_num) / (all+neg_num))
        map_val = torch.tensor([1,pos1_neg ,pos2_neg,pos3_neg,pos4_neg,pos5_neg,pos6_neg]).cuda()

        dy_gamma = self.gamma

        logpt = -F.cross_entropy(input1, target1,ignore_index=self.ignore_index,weight=map_val)
        pt = torch.exp(logpt)

        # compute the loss
        loss = (-((1-pt)**dy_gamma)*logpt)
        # print(loss)

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()



class ChangeSimilarity(nn.Module):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """
    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)

        # self.loss_f = nn.TripletMarginLoss(margin=0., reduction=reduction)
    def forward(self, x1, x2, label_change):
        """
                x1, x2: [B, C, H, W] 模型输出的 logits (未经过 softmax)
                label_change: [B, H, W] 变化标签 (1为变化, 0为未变化)
                """
        b, c, h, w = x2.size()
        p1 = F.softmax(x1, dim=1)
        p2 = F.softmax(x2, dim=1)

        p1_class1 = p1[:, 1:2, :, :]
        p1_class2 = p1[:, 2:3, :, :]
        p1_class3 = p1[:, 3:4, :, :]



        x2_change = p1[:, 1:4, :, :].sum(dim=1,keepdim=True)
        x2_nonchange = p1[:, 0:1, :, :]

        p1 = torch.cat([x2_nonchange, x2_change], dim=1)
        p1_class1 = torch.cat([x2_nonchange, p1_class1], dim=1)
        p1_class2 = torch.cat([x2_nonchange, p1_class2], dim=1)
        p1_class3 = torch.cat([x2_nonchange, p1_class3], dim=1)


        ent1 = -torch.sum(p1 * torch.log(p1 + 1e-6), dim=1)
        ent2 = -torch.sum(p2 * torch.log(p2 + 1e-6), dim=1)
        uncertainty = (ent1 + ent2) / 2.0  # [B, H, W]


        feat1 = p1.permute(0, 2, 3, 1).reshape(-1, c)
        feat2 = p2.permute(0, 2, 3, 1).reshape(-1, c)

        feat1_class1 = p1_class1.permute(0, 2, 3, 1).reshape(-1, c)
        feat1_class2 = p1_class2.permute(0, 2, 3, 1).reshape(-1, c)
        feat1_class3 = p1_class3.permute(0, 2, 3, 1).reshape(-1, c)


        cos_sim = F.cosine_similarity(feat1, feat2, dim=1)
        cos_sim12 = F.cosine_similarity(feat1_class3, feat2, dim=1)

        cos_sim = cos_sim.view(b, h, w)
        cos_sim12 = cos_sim12.view(b, h, w)

        mask_change = label_change.float()
        mask_unchange = 1.0 - mask_change

        loss_unchange = (1.0 - cos_sim) * mask_unchange

        loss_change = F.relu(cos_sim - cos_sim12) * mask_change

        total_loss = loss_unchange + loss_change

        return total_loss.mean()


class cs_loss(nn.Module):
    def __init__(self):
        super(cs_loss, self).__init__()

        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction='mean')
    def forward(self, feat1, feat2, mask,label_change):

        f1 = feat1  # [N, 2]
        f2 = feat2  # [N, 2]
        mask_flat = mask.view(-1)  # [B*H*W] = [N]
        label_change = label_change.view(-1)

        valid = label_change > 0.5  # [N], bool
        mask_flat = mask_flat[valid]
        label_change = label_change[valid]

        mask_conchange = torch.zeros_like(label_change)
        mask_only0 = (label_change == 1)
        mask_only1 = (mask_flat == 1)
        mask_conchange [mask_only0] = 0
        mask_conchange [mask_only1] = 1

        f1 = f1[valid]  # [N_valid, 2]
        f2 = f2[valid]  # [N_valid, 2]

        loss = self.loss_f(f1, f2, mask_conchange)

        return loss
class ChangeSimilarity2(nn.Module):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """
    def __init__(self, reduction='mean'):
        super(ChangeSimilarity2, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)
        self.dicee = cs_loss()

    def forward(self, x1, x2, label_temporal, label_change):

        l1_class1 = torch.zeros_like(label_temporal)
        l1_class2 = torch.zeros_like(label_temporal)
        l1_class3 = torch.zeros_like(label_temporal)
        mask12_only = (label_temporal == 1)
        mask23_only = (label_temporal == 2)
        mask123_only = (label_temporal == 3)
        l1_class1[mask12_only] = 1
        l1_class2[mask23_only] = 1
        l1_class3[mask123_only] = 1


        b, c, h, w = x2.size()

        p1 = F.softmax(x1, dim=1)
        p2 = F.softmax(x2, dim=1)

        p1_class3 = p1[:, 3:4, :, :]
        p1_change = p1[:, 1:4, :, :].sum(dim=1,keepdim=True)
        p2_nonchange = p1[:, 0:1, :, :]

        p1_binary = torch.cat([p2_nonchange, p1_change], dim=1)
        p1_class3 = torch.cat([p2_nonchange, p1_class3], dim=1)

        feat1 = p1_binary.permute(0, 2, 3, 1)
        feat2 = p2.permute(0, 2, 3, 1)
        feat1_class3 = p1_class3.permute(0, 2, 3, 1)
        feat2_class3 = p2.permute(0, 2, 3, 1)

        feat1 = torch.reshape(feat1, [b * h * w, c])
        feat2 = torch.reshape(feat2, [b * h * w, c])
        feat1_class3 = torch.reshape(feat1_class3, [b * h * w, c])
        feat2_class3 = torch.reshape(feat2_class3, [b * h * w, c])

        labels_unchange = ~label_change.bool()
        target_unchange = labels_unchange.float()
        target_unchange = target_unchange-labels_unchange.float()
        target_unchange = torch.reshape(target_unchange,[b*h*w])

        cos_sim2 = self.loss_f(feat1, feat2, target_unchange)
        cos_change3 = self.dicee(feat1_class3, feat2_class3,l1_class3,label_change)

        total_loss = (cos_sim2+cos_change3)/2

        return total_loss


class ChangeSimilarity_multi(nn.Module):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """
    def __init__(self, reduction='mean'):
        super(ChangeSimilarity_multi, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)

        # self.loss_f = nn.TripletMarginLoss(margin=0., reduction=reduction)
    def forward(self, x_temporal, x1,x2, labels12,labels23):
        b, c1, h, w = x1.size()
        b, c2, h, w = x2.size()

        x_temporal = F.softmax(x_temporal, dim=1)
        C0 = x_temporal[:, 0:1, :, :]
        C1 = x_temporal[:, 1:2, :, :]
        C2 = x_temporal[:, 2:3, :, :]
        C3 = x_temporal[:, 3:4, :, :]
        F_A = (C1 + C3)
        F_B = (C2 + C3)
        F_C = C3

        P_A = torch.cat([C0, F_A], dim=1)
        P_B = torch.cat([C0, F_B], dim=1)
        P_C = torch.cat([C0, F_C], dim=1)
        P_A = P_A.permute(0,2,3,1)
        P_B = P_B.permute(0,2,3,1)
        P_C = P_C.permute(0, 2, 3, 1)

        P_A = torch.reshape(P_A, [b * h * w, c1])
        P_B = torch.reshape(P_B, [b * h * w, c2])
        P_C = torch.reshape(P_C, [b * h * w, c2])

        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)

        # 对应元素相乘
        intersection = x1 * x2

        # 重新归一化
        x3 = F.softmax(intersection, dim=1)

        x1 = x1.permute(0,2,3,1)
        x2 = x2.permute(0,2,3,1)
        x3 = x3.permute(0, 2, 3, 1)

        x1 = torch.reshape(x1,[b * h * w,c1])
        x2 = torch.reshape(x2,[b * h * w,c2])
        x3 = torch.reshape(x3, [b * h * w, c2])

        labels12_unchange = ~labels12.bool()
        target12 = labels12_unchange.float()
        target12 = target12-labels12.float()
        target12 = torch.reshape(target12,[b*h*w])

        labels23_unchange = ~labels23.bool()
        target23 = labels23_unchange.float()
        target23 = target23 - labels23.float()
        target23 = torch.reshape(target23, [b * h * w])


        labels13  = labels12*labels23
        labels13_unchange = ~labels13.bool()
        target13 = labels13_unchange.float()
        target13 = target13 - labels13.float()
        target13 = torch.reshape(target13, [b * h * w])

        loss1 = self.loss_f(x1, P_A, target12)
        loss2 = self.loss_f(x2, P_B, target23)
        loss3 = self.loss_f(x3, P_C, target13)
        loss = (loss1+loss2)+loss3
        return loss











