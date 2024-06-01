gpu = 1
cpu = 16
memory = 60000
group = fingerprint

train_cnn:
	rlaunch -P1 --charged-group=${group} --preemptible=no --negative-tags 1080ti --negative-tags p40 --cpu=${cpu} --gpu=${gpu} --memory=${memory} -- python3 train.py --batch_size=16 --video_path=../dataset/dataset0420 --backbone=cnn
train_vit:
	rlaunch -P1 --charged-group=${group} --preemptible=no --negative-tags 1080ti --negative-tags p40 --cpu=${cpu} --gpu=${gpu} --memory=${memory} -- python3 train.py --batch_size=16 --video_path=../dataset/dataset0420 --backbone=vit
