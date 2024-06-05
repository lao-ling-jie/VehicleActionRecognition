import os
import torch
from torch.utils.data import Dataset
import numpy as np
from .loader import VideoLoaderAVI

import pdb



class VideoTransformerDataset(Dataset):

    def __init__(self, 
                 root_path,
                 image_processor,
                 loader = VideoLoaderAVI(),
                 is_train=True,
                 sample_number=5):
        
        self.loader = loader
        self.processor = image_processor
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
            filename = os.path.join(root_path, 'train.txt')
        else:
            filename = os.path.join(root_path, 'test.txt')
        with open(filename, 'r') as f:
            for line in f:
                data_path.append(line.split(' ')[0])
                label.append(int(line.split(' ')[1]))

        return data_path, label, idx2label

    def __loading(self, path):
        
        clip = self.loader(path, self.sample_number)
        clip = self.processor(clip, return_tensors="pt")        

        return clip
    
    def __getitem__(self, index):

        path = self.data_path[index]
        label = torch.tensor(self.label[index])
        clip = self.__loading(path)

        return clip, label
