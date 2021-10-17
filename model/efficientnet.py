import torch
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import MemoryEfficientSwish
from torch import nn
from torch.nn import functional as F
from utils import freeze_bn, freeze_module

def get_model(name, num_classes):
    # encoder
    model = EfficientNet.from_pretrained(name, num_classes)
    return model


model_dict = {
    'efficientnet-b0': 1280,
    'efficientnet-b2': 1408
}


class SupConEfficient(nn.Module):
    def __init__(self, name='efficientnet-b0', head='mlp', feat_dim=128, pretrained=True, **kwargs):
        super().__init__()
        if pretrained:
            self.encoder = EfficientNet.from_pretrained(name, include_top=False)
        else:
            self.encoder = EfficientNet.from_name(name, include_top=False)
        freeze_bn(self.encoder)
        # prj head
        in_feature = model_dict[name]
        if head == 'linear':
            self.head = nn.Linear(in_feature, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(in_feature, in_feature),
                nn.ReLU(inplace=True),
                nn.Linear(in_feature, feat_dim)
            )
        # Classify layer
        num_classes = kwargs.get('num_classes')
        self._dropout = nn.Dropout(0.5)
        self._fc = nn.Linear(in_feature, num_classes)

    def forward(self, x, contrastive=False):
        feat = self.encoder(x)
        if contrastive:
            x = feat.flatten(start_dim=1)
            x = self.head(x)
            x = F.normalize(x, dim=1)
        else:
            x = feat.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)
        return x

    def freeze_encoder(self):
        freeze_module(self.encoder, trainable=False)

    def unfreeze_encoder(self):
        freeze_module(self.encoder, trainable=True)
