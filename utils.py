import cv2
import os
import json
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

if __name__ == "__main__":

    # change_file_name("dataset0420")
    count_frames_and_plot_histogram("dataset0420")