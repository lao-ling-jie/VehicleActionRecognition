import os
from pathlib import Path
import torch
import torch.utils.data as data

from .loader import VideoLoaderAVI


class VideoDataset(data.Dataset):

    def __init__(self, 
                 root_path,
                 spatial_transform = None,
                 temporal_transform = None,
                 sample_number = 5 ):
        
        self.loader = VideoLoaderAVI()
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.sample_number = sample_number

        self.data_path, self.label, self.idx2label = self.__make_dataset(root_path)

    def __make_dataset(self, root_path):

        dirs = os.listdir(root_path)
        label2idx = {d.split('_')[1]: int(d.split('_')[0]) for d in dirs}
        idx2label = {int(d.split('_')[0]): d.split('_')[1] for d in dirs}

        data_path = [file.as_posix() for file in Path(root_path).rglob('*.avi')]
        label = [label2idx[path.split('/')[-1].split('.avi')[0].split(' ')[-1]] for path in data_path]

        return data_path, label, idx2label

    def __loading(self, path):
        
        clip = self.loader(path, self.sample_number)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        return clip
    
    def __getitem__(self, index):

        path = self.data_path[index]
        label = self.label[index]

        clip = self.__loading(path)

        return clip, label



