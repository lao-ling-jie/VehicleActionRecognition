import cv2
import os
import json
import random
from pathlib import Path
import matplotlib.pyplot as plt

def change_file_name(data_root):

    dir_paths = os.listdir(data_root)
    for i, dir_path in enumerate(dir_paths):
        filename = os.listdir(os.path.join(data_root, dir_path))[0]
        label = filename.split(".avi")[0].split(' ')[-1]
        print(os.path.join(data_root, str(i) + '_' + label))
        os.rename(os.path.join(data_root, dir_path), os.path.join(data_root, str(i) + '_' + label))

def sample_frames_from_videos(root_path, frame_num):
    video_files_dict = {}

    # 根据路径遍历所有.avi视频文件
    for video_file in Path(root_path).rglob('*.avi'):
        # 获取视频文件的绝对路径
        video_path = str(video_file.resolve())

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Unable to open video file {video_path}")
            continue
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数
        if frame_num > total_frames:
            print(f"Warning: Requested more frames than available in {video_path}")
            error_cnt += 1
            continue

        frame_indices = []
        
        # 计算采样间隔
        interval = total_frames // frame_num

        # 采样帧的索引
        for i in range(frame_num):
            # 计算每个所需帧的帧索引
            index = i * interval
            if index < total_frames:
                frame_indices.append(index)
        
        # 关闭视频
        cap.release()
        
        # 把该视频文件的帧索引列表添加到字典中
        video_files_dict[video_path] = frame_indices
        print(f"{video_path} has done")
    
    # 将字典保存到JSON文件
    with open(os.path.join(root_path, 'frame_indices.json'), 'w') as f:
        json.dump(video_files_dict, f)
    print(f"Data has been successfully saved to {os.path.join(root_path, 'frame_indices.json')}")
    
def count_frames_and_plot_histogram(root_path):
    frame_counts = []  # 存储所有视频的帧数

    # 遍历所有.avi视频文件
    for video_file in Path(root_path).rglob('*.avi'):
        # 获取视频文件的绝对路径
        video_path = str(video_file.resolve())
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Unable to open video file {video_path}")
            continue
        
        # 获取视频总帧数并添加到列表
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_counts.append(total_frames)
        
        # 关闭视频文件
        cap.release()

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(frame_counts, bins=30, color='blue', alpha=0.7)  # 可以调整bins的数量以改变直方图的粒度
    plt.title('Histogram of Frame Counts')
    plt.xlabel('Frame Count')
    plt.ylabel('Number of Videos')
    plt.grid(True)
    plt.show()

def split_videos(root_dir, train_ratio=0.9):
    label_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    train_filename = os.path.join(root_dir, 'train_path.txt')
    test_filename = os.path.join(root_dir, 'test_path.txt')

    train_file = open(train_filename, 'w')
    test_file = open(test_filename, 'w')

    # 遍历每一个类别目录
    for label_dir in label_dirs:
        videos = list(Path(root_dir).joinpath(label_dir).rglob('*.avi'))
        if len(videos) == 0:
            continue
        
        label = label_dir.split("_")[0]
        # 确保至少有一个视频用于测试
        random.shuffle(videos)
        num_test = max(1, int(len(videos) * (1 - train_ratio)))  # 至少1个测试文件
        test_videos = videos[:num_test]
        train_videos = videos[num_test:]

        # 测试集
        for video in test_videos:
            test_file.write(str(video) + ' ' + label + '\n')

        # 训练集
        for video in train_videos:
            train_file.write(str(video) + ' ' + label + '\n')
    
    train_file.close()
    test_file.close()

def unify_filename(root_dir):
    
    data_path = [file.as_posix() for file in Path(root_dir).rglob('*.avi')]
    for path in data_path:
        os.rename(path, path.replace(" ", "_"))
    print("done!")

if __name__ == "__main__":

    data_root = "C:\汽车运动视频检测\dataset0420"
    # change_file_name("/data/others/ChangeLineRecognition/dataset/dataset0420")
    split_videos(data_root)