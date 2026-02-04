import os
import time

import torch.nn as nn
import torch.autograd
from skimage import io
from torch import optim

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

working_path = os.path.dirname(os.path.abspath(__file__))

from utils.loss import FocalLoss2d,ChangeSimilarity2
from utils.utils import accuracy,  AverageMeter,SCDD_eval
from tqdm import tqdm
###############################################
from datasets import RS_ST as RS
from models.Network import CSTMNet as Net

NET_NAME = 'Loss'
DATA_NAME = 'DATA'
###############################################

# Training options
###############################################
args = {
    'train_batch_size':8,
    'val_batch_size': 8,
    'lr': 0.0001,
    'epochs': 50,
    'gpu': True,
    'lr_decay_power': 0.9,
    'betas': (0.9, 0.999),
    'weight_decay': 0.05,
    'alpha': 0.99,
    'eps': 1e-8,
    'momentum': 0.9,
    'print_freq': 20,
    'predict_step': 1,
    'pred_dir': os.path.join(working_path, 'results', DATA_NAME),
    'chkpt_dir': os.path.join(working_path, 'checkpoints4', DATA_NAME),
    'log_dir': os.path.join(working_path, 'logs', DATA_NAME, NET_NAME),
    # 'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME, 'pretrained.pth')
}
###############################################

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
writer = SummaryWriter(args['log_dir'])

def main():
    net = Net(in_channels=3,num_classes=RS.num_classes).cuda()

    train_set = RS.Data('train', random_flip=True)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=2, shuffle=True)
    val_set = RS.Data('test')
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=2, shuffle=False)

    criterion = FocalLoss2d(ignore_index=-1).cuda()
    # optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'], weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'],weight_decay=args['weight_decay'], betas=args['betas'], eps=args['eps'], amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95, last_epoch=-1)
    train(train_loader, net, criterion,optimizer, scheduler, val_loader)
    writer.close()
    print('Training finished.')


def train(train_loader, net, criterion, optimizer, scheduler, val_loader):
    bestaccT = 0
    bestFscdV = 0.0
    bestloss = 1.0
    begin_time = time.time()
    all_iters = float(len(train_loader) * args['epochs'])
    criterion_sc = ChangeSimilarity2()
    curr_epoch = 0
    while True:
        torch.cuda.empty_cache()
        net.train()
        # freeze_model(NCNet)
        start = time.time()
        acc_meter_all = AverageMeter()

        train_loss = AverageMeter()

        curr_iter = curr_epoch * len(train_loader)
        for i, data in enumerate(tqdm(train_loader)):
            running_iter = curr_iter + i + 1
            adjust_lr(optimizer, running_iter, all_iters)
            imgs_1, imgs_2, imgs_3, labels_12,labels_23,labels = data
            if args['gpu']:
                imgs_1 = imgs_1.cuda().float()
                imgs_2 = imgs_2.cuda().float()
                imgs_3 = imgs_3.cuda().float()

                labels12 = labels_12.cuda().long()
                labels23 = labels_23.cuda().long()

                labels_temporal = torch.zeros_like(labels12)
                mask12_only = (labels12 == 1) & (labels23 == 0)
                labels_temporal[mask12_only] = 1
                mask23_only = (labels12 == 0) & (labels23 == 1)
                labels_temporal[mask23_only] = 2
                mask_both = (labels12 == 1) & (labels23 == 1)
                labels_temporal[mask_both] = 3
                labels_binary = (labels_temporal>0).cuda().long()

            optimizer.zero_grad()
            change_temporal,change_binary = net(imgs_1, imgs_2, imgs_3)

            loss1 =  criterion(change_temporal, labels_temporal)+ criterion(change_binary, labels_binary) + criterion_sc(change_temporal,change_binary,labels_temporal,labels_binary)
            loss = loss1
            loss.backward()
            optimizer.step()


            train_loss.update(loss.cpu().detach().numpy())
            curr_time = time.time() - start

            if (i + 1) % args['print_freq'] == 0:
                print('[epoch %d] [iter %d / %d %.1fs]' % (curr_epoch, i + 1, len(train_loader), curr_time))

        F_a,F_b = validate(val_loader, net, criterion, curr_epoch)


        torch.save(net.state_dict(), os.path.join(args['chkpt_dir'],NET_NAME + '_%de_F1_%.2f_IoU%.2f.pth' % (curr_epoch, F_a* 100, F_b* 100)))
        # print('Total time: %.1fs Best rec: Train acc %.2f, Val Fscd %.2floss %.4f' % (time.time() - begin_time, bestaccT * 100, bestFscdV * 100, bestloss))
        curr_epoch += 1
        # scheduler.step()
        if curr_epoch >= args['epochs']:
            return


def validate(val_loader, net, criterion, curr_epoch):

    net.eval()

    torch.cuda.empty_cache()
    start = time.time()
    val_loss = AverageMeter()
    acc_meter_all = AverageMeter()
    acc_meter_bi = AverageMeter()
    acc_meter_T2 = AverageMeter()
    acc_meter_T3 = AverageMeter()
    acc_meter_T4 = AverageMeter()
    preds_temporal_val = []
    preds_binary_val = []
    preds_T2_val = []
    preds_T3_val = []
    preds_T4_val = []
    labels_temporal_val = []
    labels_binary_val = []
    labels_T2_val = []
    labels_T3_val = []
    labels_T4_val = []
    for vi, data in enumerate(tqdm(val_loader)):
        imgs_1, imgs_2, imgs_3, labels_12,labels_23,labels = data
        if args['gpu']:
            imgs_1 = imgs_1.cuda().float()
            imgs_2 = imgs_2.cuda().float()
            imgs_3 = imgs_3.cuda().float()

            labels12 = labels_12.cuda().long()
            labels23 = labels_23.cuda().long()

            labels_temporal = torch.zeros_like(labels12)
            mask12_only = (labels12 == 1) & (labels23 == 0)
            labels_temporal[mask12_only] = 1
            mask23_only = (labels12 == 0) & (labels23 == 1)
            labels_temporal[mask23_only] = 2
            mask_both = (labels12 == 1) & (labels23 == 1)
            labels_temporal[mask_both] = 3

            labels_binary = (labels_temporal>0).cuda().long()

        with torch.no_grad():
            change_temporal, change_binary = net(imgs_1, imgs_2, imgs_3)
            loss = criterion(change_temporal, labels_temporal) + criterion(change_binary,labels_binary)
        val_loss.update((loss).cpu().detach().numpy())
        labels12 = labels12.cpu().detach().numpy()
        labels23 = labels23.cpu().detach().numpy()
        labels_binary = labels_binary.cpu().detach().numpy()
        labels_temporal = labels_temporal.cpu().detach().numpy()

        change_temporal = torch.softmax(change_temporal,dim=1).detach()
        change_temporal = torch.argmax(change_temporal, dim=1).cpu().numpy()

        preds_temporal = change_temporal


        for (pred_temporal, label_temporal) in zip(preds_temporal, labels_temporal):

            preds_temporal_val.append(pred_temporal)
            labels_temporal_val.append(label_temporal)


        if curr_epoch % args['predict_step'] == 0 and vi == 0:
            print('Prediction saved!')


    OA_b, P_b , R_b , F_b , IoU_b = SCDD_eval(preds_temporal_val, labels_temporal_val, 4)

    curr_time = time.time() - start
    print('%.1fs F_t: %.2f M_t: %.2f ' % (curr_time, F_b * 100,  IoU_b * 100))

    return F_b,IoU_b


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def adjust_lr(optimizer, curr_iter, all_iter, init_lr=args['lr']):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** args['lr_decay_power'])
    running_lr = init_lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


if __name__ == '__main__':
    main()