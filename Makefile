gpu = 8
cpu = 16
memory = 60000
group = fingerprint
video_path = /data/others/ChangeLineRecognition/dataset/hdd_data
exp_name = debug
python3 = python3

train_resnet18:
	rlaunch -P1 --charged-group=${group} --preemptible=yes --negative-tags 1080ti --negative-tags p40 --cpu=${cpu} --gpu=${gpu} --memory=${memory} -- \
	${python3} train.py --batch_size=128 --video_path=${video_path} --backbone=resnet18 --exp_name=${exp_name} --nsample=8

train_resnet50:
	rlaunch -P1 --charged-group=${group} --preemptible=yes --negative-tags 1080ti --negative-tags p40 --cpu=${cpu} --gpu=${gpu} --memory=${memory} -- \
	${python3} train.py --batch_size=128 --video_path=${video_path} --backbone=resnet50 --exp_name=${exp_name} --nsample=8

train_vivit:
	rlaunch -P1 --charged-group=${group} --preemptible=yes --negative-tags 1080ti --negative-tags p40 --cpu=${cpu} --gpu=${gpu} --memory=${memory} -- \
	${python3} train.py --batch_size=64 --video_path=${video_path} --backbone=vivit --exp_name=${exp_name} --nsample=8

train_timesformer:
	rlaunch -P1 --charged-group=${group} --preemptible=yes --negative-tags 1080ti --negative-tags p40 --cpu=${cpu} --gpu=${gpu} --memory=${memory} -- \
	${python3} train.py --batch_size=128 --video_path=${video_path} --backbone=timesformer --exp_name=${exp_name} --nsample=8

train_videomae:
	rlaunch -P1 --charged-group=${group} --preemptible=yes --negative-tags 1080ti --negative-tags p40 --cpu=${cpu} --gpu=${gpu} --memory=${memory} -- \
	${python3} train.py --batch_size=128 --video_path=${video_path} --backbone=videomae --exp_name=${exp_name} --nsample=8

debug:
	${python3} train.py --batch_size=64 --video_path=${video_path} --backbone=vivit --exp_name=debug --nsample=8
