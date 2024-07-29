import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models._utils import IntermediateLayerGetter
from core.models.Mit_Moudle.Mit import MIT_Backbone


class Feature_extractor_ours_mit(nn.Module):
    def __init__(self):
        super(Feature_extractor_ours_mit, self).__init__()
        self.feature_extractor = MIT_Backbone()

    def forward(self, x):
        out = self.feature_extractor(x)
        return out