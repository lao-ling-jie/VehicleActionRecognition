import cv2
import os
import json
import random
from pathlib import Path
import matplotlib.pyplot as plt

import pdb

def get_model_dir():
    
    return os.path.join(os.getcwd(), 'ckpts/')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

def change_file_name(data_root):

    dir_paths = os.listdir(data_root)
    for i, dir_path in enumerate(dir_paths):
        if not os.path.isdir(os.path.join(data_root, dir_path)):
            continue
        filename = os.listdir(os.path.join(data_root, dir_path))[0]
        label = filename.split(".avi")[0].split('_')[-1]
        os.rename(os.path.join(data_root, dir_path), os.path.join(data_root, str(i) + '_' + label))
        print(f"{os.path.join(data_root, dir_path)} ===> {os.path.join(data_root, str(i) + '_' + label)}")

def unify_filename(root_dir):
    
    data_path = [file.as_posix() for file in Path(root_dir).rglob('*.avi')]
    for path in data_path:
        os.rename(path, path.replace(" ", "_"))
    print("done!")

def get_video_info(root_path):
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
        
        label = int(str(video_file).split("/")[-2].split("_")[0])
        label_name = str(video_file).split("/")[-2].split("_")[1]
        info = {
            'frame_num': total_frames,
            'label' : label,
            'label_name' : label_name
        }
        # 关闭视频
        cap.release()
        
        # 把该视频文件的帧索引列表添加到字典中
        video_files_dict[video_path] = info
        print(f"{video_path} has done")
    
    # 将字典保存到JSON文件
    with open(os.path.join(root_path, 'video_info.json'), 'w') as f:
        json.dump(video_files_dict, f)
    print(f"Data has been successfully saved to {os.path.join(root_path, 'video_info.json')}")
    
def count_frames_and_plot_histogram(root_path):
    frame_counts = {}  # 存储所有视频的帧数

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
        frame_counts[video_path] = total_frames
        
        # 关闭视频文件
        cap.release()

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(frame_counts, bins=30, color='blue', alpha=0.7)  # 可以调整bins的数量以改变直方图的粒度
    plt.title('Histogram of Frame Counts')
    plt.xlabel('Frame Count')
    plt.ylabel('Number of Videos')
    plt.grid(True)
    plt.savefig(os.path.join(root_path, 'frame_counts.png'), dpi=300)

    return frame_counts

def split_train_test(json_file, frame_num, ratio, per_label_num):
    
    data_root = '/'.join(json_file.split("/")[:-1])

    # 加载 JSON 文件
    with open(json_file, 'r') as file:
        data = json.load(file)

    # 准备数据
    labeled_videos = {}
    for video_path, info in data.items():
        label = info['label']
        frame_count = info['frame_num']
        label_name = info['label_name']
        if frame_count >= frame_num:
            if label not in labeled_videos:
                labeled_videos[label] = []
            labeled_videos[label].append((video_path, label_name))
    
    # 检查每个类别的文件数量是否满足要求，并去除不满足条件的标签
    to_remove = [label for label, videos in labeled_videos.items() if len(videos) < per_label_num]
    for label in to_remove:
        print(f"标签 '{label}' 的视频数量不足，已从划分中移除。")
        del labeled_videos[label]


    # 重新排序标签，从0到n
    sorted_labels = sorted(labeled_videos.keys())
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_labels)}
    labeled_videos = {label_mapping.get(old_label): videos for old_label, videos in labeled_videos.items()}
    
    # 划分训练集和测试集
    train_set = {}
    test_set = {}
    for label, videos in labeled_videos.items():
        random.shuffle(videos)
        test_size = max(int(len(videos) * ratio), 1)  # 确保至少有一个测试视频
        test_set[label] = videos[:test_size]
        train_set[label] = videos[test_size:]

    # 写入训练集和测试集到文本文件
    with open(os.path.join(data_root, 'train.txt'), 'w') as train_file, open(os.path.join(data_root, 'test.txt'), 'w') as test_file:
        for label, videos in train_set.items():
            for video_path, label_name in videos:
                train_file.write(f"{video_path} {label} {label_name}\n")
        for label, videos in test_set.items():
            for video_path, label_name in videos:
                test_file.write(f"{video_path} {label} {label_name}\n")

    print("split done!")

def save_video2img(root_dir, save_dir, num_frames_to_extract=5):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for video_file in Path(root_dir).rglob('*.avi'):
        
        video_path = str(video_file.resolve())
        save_video_dir = str(video_file).replace(root_dir, save_dir).split(".avi")[0]
        
        if not os.path.exists(save_video_dir):
            os.makedirs(save_video_dir)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < num_frames_to_extract:
            continue
        
        # 计算帧抽取间隔
        interval = max(1, total_frames // num_frames_to_extract)
        
        current_frame_index = 0
        
        # 提取帧直到达到需求的数量
        for i in range(num_frames_to_extract):
            if current_frame_index < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
                ret, frame = cap.read()
                if ret:
                    # 保存
                    cv2.imwrite(os.path.join(save_video_dir, str(i) + '.png'), frame)
                current_frame_index += interval
            else:
                break

        # 完成视频文件操作后释放资源
        cap.release()

if __name__ == "__main__":

    data_root = "/data/others/ChangeLineRecognition/dataset/dataset0420"
    # unify_filename(data_root)
    # change_file_name(data_root)
    # get_video_info(data_root)

    video_info_path = "/data/others/ChangeLineRecognition/dataset/dataset0420/video_info.json"
    split_train_test(video_info_path, 5, 0.1, 45)

    # save_video2img(data_root, data_root.replace('dataset0420', 'dataset0420_f5'), num_frames_to_extract=5)
    