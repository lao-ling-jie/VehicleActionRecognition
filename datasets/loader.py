import cv2
from PIL import Image

class VideoLoaderAVI(object):
    """类用于加载.avi视频文件，并按固定间隔提取指定数量的帧。"""
    
    def __call__(self, video_path, num_frames_to_extract):
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: Cannot open video file.")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 计算帧抽取间隔
        interval = max(1, total_frames // num_frames_to_extract)
        
        video = []
        current_frame_index = 0
        
        # 提取帧直到达到需求的数量
        for _ in range(num_frames_to_extract):
            if current_frame_index < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_index)
                ret, frame = cap.read()
                if ret:
                    # 将读取的帧由BGR转换为RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 使用PIL生成图片
                    video.append(Image.fromarray(frame_rgb))
                current_frame_index += interval
            else:
                break

        # 完成视频文件操作后释放资源
        cap.release()
        
        return video
