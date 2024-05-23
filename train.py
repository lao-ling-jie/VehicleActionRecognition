import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import torchvision
from transformers import VivitModel
from torch.utils.tensorboard import SummaryWriter
from dataset import get_testing_data, get_training_data
from model import VideoModel


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
    parser.add_argument('--learning_rate',
                        default=0.1,
                        type=float,
                        help=('Initial learning rate'
                              '(divided by 10 while training by lr scheduler)'))
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight_decay',
                        default=4e-5,
                        type=float,
                        help='Weight Decay')
    parser.add_argument('--nesterov',
                        action='store_true',
                        help='Nesterov momentum')

    parser.add_argument('--lr_scheduler',
                        default='multistep',
                        type=str,
                        help='Type of LR scheduler (multistep | plateau)')
    parser.add_argument(
        '--multistep_milestones',
        default=[50, 100, 150],
        type=int,
        nargs='+',
        help='Milestones of LR scheduler. See documentation of MultistepLR.')
    parser.add_argument(
        '--overwrite_milestones',
        action='store_true',
        help='If true, overwriting multistep_milestones when resuming training.'
    )
    parser.add_argument(
        '--plateau_patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )
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

def train():

    return

@torch.no_grad()
def test():
    return 

def main():
    
    args = get_args()
    set_random_seeds(529)
    if not os.path.exists('ckpts'):
        os.makedirs('ckpts')

    trainloader = get_training_data(args)
    testloader = get_testing_data(args)

    model = VideoModel(backbone='vivit', class_num=args.n_classes, pretrain=True)
 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_deacy)
    writer = SummaryWriter('train_log/')

    start_epoch = 0
    if args.pretrain_path:
        checkpoints = torch.load(args.pretrain_path, map_location='cpu')
        model.load_state_dict(checkpoints['net'])
        start_epoch = checkpoints['epoch']
        optimizer.load_state_dict(checkpoints['optimizer'])

    for epoch in range(start_epoch, args.nepoch):
        train()
        test()

        if (epoch + 1) % 5 == 0:
            save_info ={
                'net':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'epoch':epoch
            }
            torch.save(save_info, f'model_{(epoch + 1)}.pth')


if __name__ == "__main__":

    main()
