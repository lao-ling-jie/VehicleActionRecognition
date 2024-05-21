from datasets import VideoDataset

def get_training_data(video_path,
                      spatial_transform,
                      ):

    return

if __name__ == "__main__":

    root_path = "/data/others/ChangeLineRecognition/dataset/dataset0420"
    dataset = VideoDataset(root_path)
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for (clip, label) in dataloader:
        print(clip.shape)
        print(label)
        break
