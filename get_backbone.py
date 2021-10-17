import numpy as np
import torch.nn as nn
import torch
from model.efficientnet import SupConEfficient
from utils import save_checkpoint

arch_name = 'efficientnet-b2'
checkpoint_path = 'runs/default/model_best.pth.tar'

labels = np.arange(13)
model = SupConEfficient(arch_name, num_classes=len(labels))
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])

save_checkpoint({
        'epoch': 0,
        'state_dict': model.encoder.state_dict(),
        'best_err1': 0
    }, True, expname='supcon_backbone')

