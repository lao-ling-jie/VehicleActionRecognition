gpu = 2
cpu = 16
memory = 60000
group = fingerprint
exp_name = debug
python3 = python3

train_resnet18:
	rlaunch -P1 --charged-group=${group} --preemptible=yes --negative-tags 1080ti --negative-tags p40 --cpu=${cpu} --gpu=${gpu} --memory=${memory} -- \
	${python3} train.py --batch_size=16 --video_path=../dataset/dataset0420 --backbone=resnet18 --exp_name=${exp_name}

train_resnet50:
	rlaunch -P1 --charged-group=${group} --preemptible=yes --negative-tags 1080ti --negative-tags p40 --cpu=${cpu} --gpu=${gpu} --memory=${memory} -- \
	${python3} train.py --batch_size=16 --video_path=../dataset/dataset0420 --backbone=resnet50 --exp_name=${exp_name}

train_vivit:
	rlaunch -P1 --charged-group=${group} --preemptible=yes --negative-tags 1080ti --negative-tags p40 --cpu=${cpu} --gpu=${gpu} --memory=${memory} -- \
	${python3} train.py --batch_size=64 --video_path=../dataset/dataset0420 --backbone=vivit --exp_name=${exp_name}

train_timesformer:
	rlaunch -P1 --charged-group=${group} --preemptible=yes --negative-tags 1080ti --negative-tags p40 --cpu=${cpu} --gpu=${gpu} --memory=${memory} -- \
	${python3} train.py --batch_size=16 --video_path=../dataset/dataset0420 --backbone=timesformer --exp_name=${exp_name}

train_videomae:
	rlaunch -P1 --charged-group=${group} --preemptible=yes --negative-tags 1080ti --negative-tags p40 --cpu=${cpu} --gpu=${gpu} --memory=${memory} -- \
	${python3} train.py --batch_size=16 --video_path=../dataset/dataset0420 --backbone=videomae --exp_name=${exp_name}
