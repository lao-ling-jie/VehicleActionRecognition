import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VivitModel, VivitConfig, \
                    TimesformerModel, TimesformerConfig, VideoMAEModel, VideoMAEConfig, VideoClassificationPipeline
from torchvision.models import resnet18, resnet50


import pdb

class ViTModel(nn.Module):
    def __init__(self, backbone='vivit', n_frame=8, class_num=19, pretrain=True):
        super(ViTModel, self).__init__()
        
        if backbone == 'vivit':
            config = VivitConfig(image_size=224, 
                                num_frames=n_frame,
                                num_hidden_layers=6,
                                hidden_size=384,
                                num_attention_heads=12,
                                intermediate_size=3072,
                                attention_probs_dropout_prob=0.2)
            self.backbone = VivitModel(config)
            if pretrain:
                self.backbone.from_pretrained("google/vivit-b-16x2-kinetics400")
        elif backbone == 'timesformer':
            config = TimesformerConfig(num_frames=n_frame, num_hidden_layers=2)
            self.backbone = TimesformerModel(config)
            if pretrain:
                self.backbone.from_pretrained("facebook/timesformer-base-finetuned-k400")
        elif backbone == 'videomae':
            config = VideoMAEConfig(num_frames=n_frame, num_hidden_layers=2)
            self.backbone = VideoMAEModel(config)
            if pretrain:
                self.backbone.from_pretrained("MCG-NJU/videomae-base")
        else:
            raise("unsuported backbone")
        
        self.classifier = nn.Linear(384 // n_frame, class_num)
    
    def forward(self, x):
        b, t, c, h, w = x.shape
        outputs = self.backbone(x)
        sequence_output = outputs[1]
        b, c = sequence_output.shape
        tfeats = sequence_output.reshape(b, t , c // t)
        logits = []
        for i in range(t):
            logits.append(self.classifier(tfeats[:, i]))
        logits = torch.stack(logits, dim=1)
        
        return logits

class CNNModel(nn.Module):
    def __init__(self, backbone='resnet18', class_num=12):
        super(CNNModel, self).__init__()
        
        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=True)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            feature_size = 512
        elif backbone == "resnet50":
            self.backbone = resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            feature_size = 2048
        else:
            raise("unsuported backbone")
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=feature_size, num_layers=3, dropout=0.2, batch_first=True)
        self.classifier = nn.Linear(feature_size, class_num)
        
    def forward(self, x):
    
        b, t, c, h, w = x.shape
        x = x.reshape(b*t, c, h, w)
        features = self.backbone(x)

        c = features.shape[1]
        features = features.reshape(b, t, c)
        self.lstm.flatten_parameters()
        t_features, _ = self.lstm(features)
        logits = []
        for i in range(t):
            logits.append(self.classifier(t_features[:, i]))
        logits = torch.stack(logits, dim=1)
        return logits

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

if __name__ == "__main__":

    import torch
    x = torch.randn(4, 8, 3, 224, 224)
    # model = ViTModel(backbone='vivit', n_frame=8, class_num=12)
    # output = model(x)
    # print(output.shape)

    model = CNNModel(backbone='resnet18')
    output = model(x)
    print(output.shape)