import os
import cv2
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the videos.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.video_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_folder = os.path.join(self.root_dir, self.video_list[idx])
        video_files = [f for f in os.listdir(video_folder) if f.endswith('.avi')]
        video_path = os.path.join(video_folder, video_files[0])  # Assuming one .avi file per folder
        
        # Read the video
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
        cap.release()
        
        # Convert frames to tensor
        frames_tensor = self.transform(frames) if self.transform else frames
        
        # Extract label from folder name
        label = int(video_folder.split('_')[-1])
        
        return frames_tensor, label