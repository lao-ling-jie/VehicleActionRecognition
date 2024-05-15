import cv2

# 视频文件的路径
video_path = r'dataset0420\1.5\scene-000753+2+1.5 ChangingTurnRight.avi'

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 获取视频的基本信息
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video Information:")
print(f"Width: {width}")
print(f"Height: {height}")
print(f"FPS: {fps}")
print(f"Total Frames: {frame_count}")

# 读取并保存前五帧
frame_number = 0
while frame_number < 5:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # 保存帧为图片
    cv2.imwrite(f'frame_{frame_number}.jpg', frame)
    frame_number += 1

# 释放视频文件
cap.release()
