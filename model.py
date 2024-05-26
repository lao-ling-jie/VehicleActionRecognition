import torch.nn as nn
from transformers import VivitModel
from transformers.models.vivit import VivitConfig

class VideoModel(nn.Module):
    def __init__(self, backbone='vivit', class_num=19, pretrain=True):
        super(VideoModel, self).__init__()
        
        if backbone == 'vivit':
            self.backbone = VivitModel(VivitConfig())
        else:
            raise("unsuported backbone")
        
        if pretrain:
            self.backbone.from_pretrained("google/vivit-b-16x2-kinetics400")
        
        self.classifier = nn.Linear(768, class_num)
    
    def forward(self, x):

        outputs = self.backbone(x)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output[:, 0, :])
        return logits