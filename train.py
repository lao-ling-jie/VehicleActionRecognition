import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from dataset import get_testing_data, get_training_data
from model import VideoModel
from utils import AverageMeter, get_model_dir

def get_args():
    parser = argparse.ArgumentParser()

    # 数据处理超参
    parser.add_argument('--video_path',
                        default=None,
                        help='Directory path of videos')
    parser.add_argument(
        '--dataset',
        default='dataset0420',
        type=str,
        help='Used dataset (dataset0420 | hdd)')
    parser.add_argument(
        '--n_classes',
        default=19,
        type=int,
        help=
        'Number of classes (dataset0420: 19, hdd: xxx)'
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
    parser.add_argument('--nepoch', default=300, type=int, help='epoch number')
    parser.add_argument('--lr',
                        default=1e-3,
                        type=float,
                        help=('Initial learning rate'))
    parser.add_argument('--min_lr',
                    default=1e-3,
                    type=float,
                    help=('min learning rate'))
    parser.add_argument('--warmup_epoch',default=10,type=int,help='warmup epoch')
    
    parser.add_argument('--batch_size',
                        default=128,
                        type=int,
                        help='Batch Size')

    args = parser.parse_args()

    return args

def set_random_seeds(seed):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torchvision.utils.set_random_seed(seed)


def train(trainloader, epoch, model, optimizer, criterion, writer):
    model.train()
    loss_meter = AverageMeter("Loss", ":.4e")
    correct_meter = AverageMeter("Accuracy", ":.4e")
    for batch_idx, (data, target) in enumerate(trainloader):

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target).sum().item()

        loss_meter.update(loss.item(), data.shape[0])
        correct_meter.update(correct.item(), data.shape[0])
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()} Acc: {correct_meter.avg}')
            writer.add_scalar('Training Loss', loss.item(), epoch * len(trainloader) + batch_idx)
            writer.add_scalar('Training Acc', correct_meter.avg, epoch * len(trainloader) + batch_idx)

@torch.no_grad()
def test(testloader, epoch, model, criterion, writer):
    model.eval()
    loss_meter = AverageMeter("Loss", ":.4e")
    correct_meter = AverageMeter("Accuracy", ":.4e")
    for batch_idx, (data, target) in enumerate(testloader):
        
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        
        output = model(data)
        loss = criterion(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target).sum().item()

        loss_meter.update(loss.item(), data.shape[0])
        correct_meter.update(correct.item(), data.shape[0])
    
    print(f'Epoch: {epoch}, Test Loss: {loss_meter.avg}, Accuracy: {correct_meter.avg}%')
    writer.add_scalar('Test Loss', loss_meter.avg, epoch)
    writer.add_scalar('Test Accuracy', correct_meter.avg, epoch)

def main():
    
    args = get_args()
    set_random_seeds(529)
    if not os.path.exists('ckpts'):
        os.makedirs('ckpts')

    trainloader = get_training_data(args)
    testloader = get_testing_data(args)

    model = VideoModel(backbone='vivit', class_num=args.n_classes, pretrain=True)
 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_deacy)
    writer = SummaryWriter('train_log/')

    # 创建warmup调度器：LinearLR
    warmup_scheduler = LinearLR(optimizer, start_factor=args.min_lr/args.lr, end_factor=1.0, total_iters=args.warmup_epoch)
    # 创建余弦退火调度器：CosineAnnealingLR, 注意这里total_epochs需要减去warmup_epochs
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.nepochs - args.warmup_epoch)
    # SequentialLR组合使用两个调度器
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_epoch])


    start_epoch = 0
    if args.pretrain_path:
        checkpoints = torch.load(args.pretrain_path, map_location='cpu')
        model.load_state_dict(checkpoints['net'])
        start_epoch = checkpoints['epoch']
        optimizer.load_state_dict(checkpoints['optimizer'])
    
    if os.path.exists(get_model_dir()):
        os.makedirs(get_model_dir())
    
    if torch.cuda.is_available():
        model.cuda()
        model = torch.nn.parallel(model)

    for epoch in range(start_epoch, args.nepoch):
        train(trainloader, epoch, model, optimizer, criterion, writer)
        test(testloader, epoch, model, criterion, writer)
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            save_info ={
                'net':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'epoch':epoch
            }
            torch.save(save_info, os.path.join(get_model_dir(), f'model_{(epoch + 1)}.pth'))


if __name__ == "__main__":

    main()
