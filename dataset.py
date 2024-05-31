from datasets import VideoDataset
from datasets.spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                        CornerCrop, MultiScaleCornerCrop,
                        RandomResizedCrop, RandomHorizontalFlip,
                        ToTensor, ScaleValue, ColorJitter,
                        PickFirstChannels)
from datasets.temporal_transforms import (LoopPadding, TemporalRandomCrop,
                        TemporalCenterCrop, TemporalEvenCrop,
                        SlidingWindow, TemporalSubsampling)
from datasets.temporal_transforms import Compose as TemporalCompose
from datasets.loader import ImageLoader, VideoLoaderAVI
from torch.utils.data import DataLoader



def get_training_transform(opt):
    assert opt.train_crop in ['random', 'corner', 'center']
    spatial_transform = []
    if opt.train_crop == 'random':
        spatial_transform.append(
            RandomResizedCrop(
                opt.sample_size, (opt.train_crop_min_scale, 1.0),
                (opt.train_crop_min_ratio, 1.0 / opt.train_crop_min_ratio)))
    elif opt.train_crop == 'corner':
        scales = [1.0]
        scale_step = 1 / (2**(1 / 4))
        for _ in range(1, 5):
            scales.append(scales[-1] * scale_step)
        spatial_transform.append(MultiScaleCornerCrop(opt.sample_size, scales))
    elif opt.train_crop == 'center':
        spatial_transform.append(Resize(opt.sample_size))
        spatial_transform.append(CenterCrop(opt.sample_size))
    if opt.colorjitter:
        spatial_transform.append(ColorJitter())
    if not opt.no_hflip:
        spatial_transform.append(RandomHorizontalFlip())
    
    spatial_transform.append(ToTensor())
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    spatial_transform.append(ScaleValue(opt.value_scale))
    spatial_transform.append(normalize)
    spatial_transform = Compose(spatial_transform)

    assert opt.train_t_crop in ['random', 'center']
    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    if opt.train_t_crop == 'random':
        temporal_transform.append(TemporalRandomCrop(opt.sample_duration))
    elif opt.train_t_crop == 'center':
        temporal_transform.append(TemporalCenterCrop(opt.sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)

    return spatial_transform, temporal_transform
    
def get_testing_transform(opt):
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    spatial_transform = [
        Resize(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor()
    ]
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    # temporal_transform.append(
    #     TemporalEvenCrop(opt.sample_duration, opt.n_val_samples))
    temporal_transform = TemporalCompose(temporal_transform)

    return spatial_transform, temporal_transform


def get_training_data(opt):
    dataset_name = opt.dataset
    video_path = opt.video_path
    spatial_transform, temporal_transform = get_testing_transform(opt)

    assert dataset_name in ['dataset0420', 'hdd']

    if dataset_name == 'dataset0420':
        training_data = VideoDataset(video_path,
                                     loader=ImageLoader(),
                                     spatial_transform=spatial_transform,
                                     temporal_transform=temporal_transform,
                                     is_train=True)
    else:
        return

    tainloader = DataLoader(training_data, batch_size=opt.batch_size, num_workers=16, shuffle=True)

    return tainloader

def get_testing_data(opt):
    dataset_name = opt.dataset
    video_path = opt.video_path
    spatial_transform, temporal_transform = get_testing_transform(opt)

    assert dataset_name in ['dataset0420', 'hdd']

    if dataset_name == 'dataset0420':
        testing_data = VideoDataset(video_path,
                                     loader=ImageLoader(),
                                     spatial_transform=spatial_transform,
                                     temporal_transform=temporal_transform,
                                     is_train=False)
    else:
        return
    
    testloader = DataLoader(testing_data, batch_size = opt.batch_size, num_workers=8, shuffle=True)

    return testloader

if __name__ == "__main__":

    from train import get_args
    args = get_args()
    dataloader = get_training_data(args)
    

    for i, (clip, label) in enumerate(dataloader):
        print(i, clip.shape)
