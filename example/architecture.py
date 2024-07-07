import torch
import torch.nn as nn
from transformers import VivitModel, VivitConfig, \
                    TimesformerModel, TimesformerConfig, VideoMAEModel, VideoMAEConfig
from vit_pytorch.vit import Transformer
from einops import repeat


class Framework(nn.Module):
    def __init__(self, decoder_dim, backbone='vivit', class_num=19, pretrain=True):
        super(Framework, self).__init__()
        
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

        self.to_patch = self.backbone.embeddings
        num_patches, encoder_dim, pixel_values_per_patch  = 392, 768, 16
        self.masking_ratio = 0


        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim) if encoder_dim != decoder_dim else nn.Identity()
        self.decoder = Transformer(dim = decoder_dim, depth = 3, heads = 8, dim_head = 64, mlp_dim = decoder_dim * 4)
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, x):
        
        outputs = self.backbone(x)
        encoded_tokens  = outputs[0]
        patches = self.to_patch(x)
        
        # cls head
        logits = self.classifier(encoded_tokens[:, 0, :])

        # decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens[:, 1:])        
        pred_pixel_values = self.to_pixels(decoder_tokens)


        return {
            'logits': logits,
            'decoer_output': pred_pixel_values,
            'encoder_patches': patches[:, 1:]
        }



if __name__ == "__main__":

    x = torch.randn(4, 5, 3, 224, 224)
    model = Framework(backbone='vivit', decoder_dim=768, pretrain=False)
    output = model(x)
    for key, val in output.items():
        print(val.shape)
