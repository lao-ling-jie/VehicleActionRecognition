import torch.nn as nn
from transformers import VivitModel, VivitConfig, \
                    TimesformerModel, TimesformerConfig, VideoMAEModel, VideoMAEConfig
from torchvision.models import resnet18


import pdb
class ViTModel(nn.Module):
    def __init__(self, backbone='vivit', class_num=19, pretrain=True):
        super(ViTModel, self).__init__()
        
        if backbone == 'vivit':
            config = VivitConfig(image_size=224, num_frames=5, num_hidden_layers=3)
            self.backbone = VivitModel(config)
            if pretrain:
                self.backbone.from_pretrained("google/vivit-b-16x2-kinetics400")
        elif backbone == 'timesformer':
            config = TimesformerConfig(num_frames=5, num_hidden_layers=3)
            self.backbone = TimesformerModel(config)
            if pretrain:
                self.backbone.from_pretrained("facebook/timesformer-base-finetuned-k400")
        elif backbone == 'videomae':
            config = VideoMAEConfig(num_frames=5, num_hidden_layers=3)
            self.backbone = VideoMAEModel(config)
            if pretrain:
                self.backbone.from_pretrained("MCG-NJU/videomae-base")
        else:
            raise("unsuported backbone")
        
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
        
        self.lstm = nn.LSTM(input_size=512, hidden_size=512, num_layers=3, dropout=0.2, batch_first=True)
        self.classifier = nn.Linear(512, class_num)
    
    def forward(self, x):

       b, t, c, h, w = x.shape
       x = x.reshape(b*t, c, h, w)
       features = self.backbone(x)

       c = features.shape[1]
       features = features.reshape(b, t, c)
       
       t_features, _ = self.lstm(features)
       out = t_features[:, -1, :]
       out = self.classifier(out)

       return out

if __name__ == "__main__":

    import torch
    x = torch.randn(4, 5, 3, 224, 224)
    model = ViTModel(backbone='timesformer', pretrain=False)
    output = model(x)
    print(output.shape)