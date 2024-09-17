# reference https://github.com/xumingze0308/TRN.pytorch/blob/master/lib/datasets/hdd_data_layer.py

import os.path as osp

import torch
import torch.utils.data as data
import numpy as np
import json
from PIL import Image
import pdb

class TRNHDDDataLayer(data.Dataset):
    def __init__(self, data_root, enc_steps, spatial_transform=None, temporal_transform=None, phase='train'):
        self.data_root = data_root
        self.enc_steps = enc_steps
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.training = phase=='train'

        with open(osp.join(self.data_root, 'dataset.json'), 'r') as f:
            self.sessions = json.load(f)[f'{phase}_session_set']

        self.inputs = []
        for session in self.sessions:
            sensor = np.load(osp.join(self.data_root, 'sensor', session+'.npy'))
            target = np.load(osp.join(self.data_root, 'target', session+'.npy'))
            seed = np.random.randint(self.enc_steps) + 1 if self.training else 90
            for start, end in zip(
                range(seed, target.shape[0], self.enc_steps),
                range(seed + self.enc_steps, target.shape[0], self.enc_steps)):
                enc_target = target[start:end]
                self.inputs.append([
                    session, start, end, sensor[start:end],enc_target,
                ])

    def get_dec_target(self, target_vector):
        target_matrix = np.zeros((self.enc_steps, self.dec_steps))
        for i in range(self.enc_steps):
            for j in range(self.dec_steps):
                target_matrix[i,j] = target_vector[i+j]
        return target_matrix

    def get_clip(self, session, start, end):

        clip = []
        for idx in range(start, end):
            img_path = osp.join(self.data_root, 'camera', session,  f"{idx:05d}.jpg")
            image = Image.open(img_path)
            if self.spatial_transform:
                image = self.spatial_transform(image)
            clip.append(image)
        clip = torch.stack(clip)
        
        if self.temporal_transform:
            clip = self.temporal_transform(clip)
        
        return clip

    def __getitem__(self, index):
        session, start, end, sensor_inputs, enc_target = self.inputs[index]
        camera_inputs = self.get_clip(session, start, end)
        sensor_inputs = torch.as_tensor(sensor_inputs.astype(np.float32))
        enc_target = torch.as_tensor(enc_target.astype(np.int64))

        return camera_inputs, sensor_inputs, enc_target

    def __len__(self):
        return len(self.inputs)
    
if __name__ == "__main__":
    data_root = "/data/others/ChangeLineRecognition/dataset/hdd_data"
    from torchvision import transforms

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = TRNHDDDataLayer(data_root, 16, spatial_transform=train_transform, temporal_transform=None, phase='train')
    sample = dataset[0]
    print(sample[0].shape)