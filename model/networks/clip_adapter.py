import torch
import torch.nn as nn
from torch.nn import functional as F
import clip
from typing import List
from torch.nn import init

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
    
class TextEncoder(nn.Module):

    def __init__(self, cfg, clip_model, prompts: List[str]):
        super().__init__()
        self.cfg = cfg
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.prompts = prompts
    
    def forward(self):
        """Return text features of CLIP model

        Args:
            prompts (List[str]): prompt + class_names

        Returns:
            _type_: _description_
        """
        prompts = torch.cat([clip.tokenize(p) for p in self.prompts])
        prompts = prompts.to('cuda')
        text_features = self.clip_model.encode_text(prompts)
        x = text_features
        return x


class CustomCLIP(nn.Module):

    def __init__(self, cfg, clip_model, ratio, prompts: List[str]):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, clip_model, prompts)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.adapter = Adapter(512, 4) # .to(clip_model.dtype)
        self.ratio = ratio
        
    def print_network(self):
        pass
    
    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.adapter.apply(init_func)

            
    def forward(self, image: torch.Tensor):
        image_features = self.image_encoder(image.type(self.dtype))
        # x = self.adapter(image_features)

        # image_features = self.ratio * x + (1 - self.ratio) * image_features # clip adapter added in the image encoder of the CLIP

        text_features = self.text_encoder()
        x = self.adapter(text_features.type(self.adapter.fc._modules['0'].weight.dtype))
        text_features = self.ratio * x + (1 - self.ratio) * text_features.float()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits