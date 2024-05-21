import os
import torch
from torch.utils.data import Dataset, random_split
from pathlib import Path
import numpy as np
from .loader import VideoLoaderAVI

import pdb



class VideoDataset(Dataset):

    def __init__(self, 
                 root_path,
                 spatial_transform=None,
                 temporal_transform=None,
                 is_train=True,
                 sample_number=5):
        
        self.loader = VideoLoaderAVI()
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.sample_number = sample_number
        self.is_train = is_train

        self.data_path, self.label, self.idx2label = self.__make_dataset(root_path)

    def __len__(self):
        
        return len(self.data_path)

    def __make_dataset(self, root_path):

        dirs = [name for name in os.listdir(root_path)
                   if os.path.isdir(os.path.join(root_path, name))]
        idx2label = {int(d.split('_')[0]): d.split('_')[1] for d in dirs}

        data_path = []
        label = []
        if self.is_train:
            filename = os.path.join(root_path, 'train_path.txt')
        else:
            filename = os.path.join(root_path, 'test_path.txt')
        with open(filename, 'r') as f:
            for line in f:
                data_path.append(line.split(' ')[0])
                label.append(int(line.split(' ')[1]))

        return data_path, label, idx2label

    def __loading(self, path):
        
        clip = self.loader(path, self.sample_number)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
            clip = torch.stack(clip, 0)
        else:
            clip = [torch.from_numpy(np.array(c)) for c in clip]
            clip = torch.stack(clip, 0).permute(0, 3, 2, 1)

        return clip
    
    def __getitem__(self, index):

        path = self.data_path[index]
        label = self.label[index]

        clip = self.__loading(path)

        return clip, label
