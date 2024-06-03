import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from dataset import get_testing_data, get_training_data
from model import ViTModel, CNNModel
from utils import AverageMeter, get_model_dir


import pdb

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name',
                        default=None,
                        required=True,
                        type=str,
                        help='experiment name')
    # 数据处理超参
    parser.add_argument('--video_path',
                        default=None,
                        required=True,
                        help='Directory path of videos')
    parser.add_argument(
        '--dataset',
        default='dataset0420',
        type=str,
        help='Used dataset (dataset0420 | hdd)')
    parser.add_argument(
        '--n_classes',
        default=7,
        type=int,
        help=
        'Number of classes (dataset0420: 7, hdd: xxx)'
    )
    parser.add_argument('--pretrain_path',
                        default=None,
                        help='Pretrained model path (.pth).')
    parser.add_argument('--sample_size',
                        default=224,
                        type=int,
                        help='Height and width of inputs')
    parser.add_argument('--sample_duration',
                        default=5,
                        type=int,
                        help='Temporal duration of inputs')
    parser.add_argument(
        '--sample_t_stride',
        default=1,
        type=int,
        help='If larger than 1, input frames are subsampled with the stride.')
    parser.add_argument(
        '--train_crop',
        default='random',
        type=str,
        help=('Spatial cropping method in training. '
              'random is uniform. '
              'corner is selection from 4 corners and 1 center. '
              '(random | corner | center)'))
    parser.add_argument(
        '--value_scale',
        default=1,
        type=int,
        help=
        'If 1, range of inputs is [0-1]. If 255, range of inputs is [0-255].')
    parser.add_argument('--train_crop_min_scale',
                        default=0.25,
                        type=float,
                        help='Min scale for random cropping in training')
    parser.add_argument('--train_crop_min_ratio',
                        default=0.75,
                        type=float,
                        help='Min aspect ratio for random cropping in training')
    parser.add_argument('--no_hflip',
                        action='store_true',
                        help='If true holizontal flipping is not performed.')
    parser.add_argument('--colorjitter',
                        action='store_true',
                        help='If true colorjitter is performed.')
    parser.add_argument('--train_t_crop',
                        default='random',
                        type=str,
                        help=('Temporal cropping method in training. '
                              'random is uniform. '
                              '(random | center)'))
   
    # 训练超参
    parser.add_argument('--backbone', default='vit', type=str, help='backbone type')
    parser.add_argument('--nepoch', default=300, type=int, help='epoch number')
    parser.add_argument('--weight_decay', default=4e-5, type=float, help='weight decay')
    parser.add_argument('--lr',
                        default=1e-3,
                        type=float,
                        help=('Initial learning rate'))
    parser.add_argument('--min_lr',
                    default=1e-4,
                    type=float,
                    help=('min learning rate'))
    parser.add_argument('--warmup_epoch',default=10,type=int,help='warmup epoch')
    
    parser.add_argument('--batch_size',
                        default=64,
                        type=int,
                        help='Batch Size')

    args = parser.parse_args()

    return args

def set_random_seeds(seed):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(trainloader, epoch, model, optimizer, criterion, writer):
    model.train()
    loss_meter = AverageMeter("Loss", ":.4e")
    acc_meter = AverageMeter("Accuracy", ":.4e")
    batch_time = AverageMeter("Time", ":6.3f")
    minibatch_count = len(trainloader)
    end = time.time()

    for batch_idx, (data, target) in enumerate(trainloader):
        
        learning_rate = optimizer.param_groups[0]['lr']        
        
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        pred = output.data.max(1)[1]
        correct = pred.eq(target).sum().item()
        acc = correct / data.shape[0]
        
        loss_meter.update(loss.item(), data.shape[0])
        acc_meter.update(acc, data.shape[0])
        
        eta = batch_time.avg * minibatch_count - batch_time.sum
        if batch_idx % 5 == 0:
            outputs = (
                ["e:{},{}/{}".format(epoch, batch_idx, minibatch_count), "{:.2g} mb/s".format(1.0 / batch_time.avg),]
                + [
                    "passed:{:.2f}".format(batch_time.sum),
                    "eta:{:.2f}".format(eta),
                    "lr:{:.5f}".format(learning_rate),
                    "acc:{:.3f}".format(acc),
                ]
                + ["loss:{:.6f}".format(loss.item()),]
            )
            
            print(" ".join(outputs))
            writer.add_scalar('Training Loss', loss.item(), epoch * len(trainloader) + batch_idx)
            writer.add_scalar('Training Acc', acc_meter.avg , epoch * len(trainloader) + batch_idx)
            writer.add_scalar('lr', learning_rate, epoch * len(trainloader) + batch_idx)
    print(f'Epoch: {epoch}, Train Loss: {loss_meter.avg}, Accuracy: {acc_meter.avg}')


@torch.no_grad()
def test(testloader, epoch, model, criterion, writer):
    model.eval()
    loss_meter = AverageMeter("Loss", ":.4e")
    acc_meter = AverageMeter("Accuracy", ":.4e")
    for batch_idx, (data, target) in enumerate(testloader):
        
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        output = model(data)
        loss = criterion(output, target)
        pred = output.data.max(1)[1]
        correct = pred.eq(target).sum().item()
        acc = correct / data.shape[0]

        loss_meter.update(loss.item(), data.shape[0])
        acc_meter.update(acc, data.shape[0])
    
    print(f'Epoch: {epoch}, Test Loss: {loss_meter.avg}, Accuracy: {acc_meter.avg}')
    writer.add_scalar('Test Loss', loss_meter.avg, epoch)
    writer.add_scalar('Test Accuracy', acc_meter.avg, epoch)

def main():
    
    args = get_args()
    set_random_seeds(529)
    if not os.path.exists('ckpts'):
        os.makedirs('ckpts')

    trainloader = get_training_data(args)
    testloader = get_testing_data(args)

    if args.backbone == 'vivit':
        model = ViTModel(backbone='vivit', class_num=args.n_classes, pretrain=False)
    if args.backbone == 'timesformer':
        model = ViTModel(backbone='timesformer', class_num=args.n_classes, pretrain=False)
    if args.backbone == 'videomae':
        model = ViTModel(backbone='videomae', class_num=args.n_classes, pretrain=False)
    elif args.backbone == 'resnet':
        model = CNNModel(backbone='resnet18', class_num=args.n_classes)
    else:
        raise("unsupported backbone")
 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    writer = SummaryWriter(os.path.join('train_log/', datetime.now().strftime('%Y-%m-%d-%H-%M') + '_' + args.exp_name))

    # 创建warmup调度器：LinearLR
    warmup_scheduler = LinearLR(optimizer, start_factor=args.min_lr/args.lr, end_factor=1.0, total_iters=args.warmup_epoch)
    # 创建余弦退火调度器：CosineAnnealingLR, 注意这里total_epochs需要减去warmup_epochs
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.nepoch - args.warmup_epoch)
    # SequentialLR组合使用两个调度器
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epoch])


    start_epoch = 0
    if args.pretrain_path:
        checkpoints = torch.load(args.pretrain_path, map_location='cpu')
        model.load_state_dict(checkpoints['net'])
        start_epoch = checkpoints['epoch']
        optimizer.load_state_dict(checkpoints['optimizer'])
    
    if not os.path.exists(get_model_dir()):
        os.makedirs(get_model_dir())
    
    if torch.cuda.is_available():
        model.cuda()
        model = torch.nn.DataParallel(model)

    for epoch in range(start_epoch, args.nepoch):
        train(trainloader, epoch, model, optimizer, criterion, writer)
        test(testloader, epoch, model, criterion, writer)
        scheduler.step()

        # if (epoch + 1) % 5 == 0:
        #     save_info ={
        #         'net':model.module.state_dict(),
        #         'optimizer':optimizer.state_dict(),
        #         'epoch':epoch
        #     }
        #     torch.save(save_info, os.path.join(get_model_dir(), f'model_{(epoch + 1)}.pth'))


if __name__ == "__main__":

    main()
