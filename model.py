import torch.nn as nn
from transformers import VivitModel
from transformers.models.vivit import VivitConfig
from torchvision.models import resnet18


import pdb
class ViTModel(nn.Module):
    def __init__(self, backbone='vivit', class_num=19, pretrain=True):
        super(ViTModel, self).__init__()
        
        if backbone == 'vivit':
            config = VivitConfig(image_size=224, num_frames=5, num_hidden_layers=6)
            self.backbone = VivitModel(config)
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

class CNNModel(nn.Module):
    def __init__(self, backbone='resnet18', class_num=19):
        super(CNNModel, self).__init__()
        
        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=True)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        else:
            raise("unsuported backbone")
        
        self.lstm = nn.LSTM(input_size=512, hidden_size=512, batch_first=True)
        self.classifier = nn.Linear(768, class_num)
    
    def forward(self, x):
       pdb.set_trace()
       b, t, c, h, w = x.shape
       x = x.reshape(b*t, c, h, w)
       features = self.backbone(x)
       
       features = features.reshape(b, t, )

if __name__ == "__main__":

    import torch
    x = torch.randn(4, 5, 3, 224, 224)
    model = CNNModel()
    output = model(x)