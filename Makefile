gpu = 1
cpu = 16
memory = 60000
group = fingerprint
exp_name = debug

train_cnn:
	rlaunch -P1 --charged-group=${group} --preemptible=no --negative-tags 1080ti --negative-tags p40 --cpu=${cpu} --gpu=${gpu} --memory=${memory} -- \
	python3 train.py --batch_size=16 --video_path=../dataset/dataset0420 --backbone=cnn --exp_name ${exp_name} --backbone resnet

train_vivit:
	rlaunch -P1 --charged-group=${group} --preemptible=no --negative-tags 1080ti --negative-tags p40 --cpu=${cpu} --gpu=${gpu} --memory=${memory} -- \
	python3 train.py --batch_size=16 --video_path=../dataset/dataset0420 --backbone=vit --exp_name ${exp_name} --backbone vivit

train_timesformer:
	rlaunch -P1 --charged-group=${group} --preemptible=no --negative-tags 1080ti --negative-tags p40 --cpu=${cpu} --gpu=${gpu} --memory=${memory} -- \
	python3 train.py --batch_size=16 --video_path=../dataset/dataset0420 --backbone=vit --exp_name ${exp_name} --backbone timesformer

train_videomae:
	rlaunch -P1 --charged-group=${group} --preemptible=no --negative-tags 1080ti --negative-tags p40 --cpu=${cpu} --gpu=${gpu} --memory=${memory} -- \
	python3 train.py --batch_size=16 --video_path=../dataset/dataset0420 --backbone=vit --exp_name ${exp_name} --backbone videomae