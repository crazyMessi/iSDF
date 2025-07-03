import torch
import torch.nn as nn
import torch.nn.functional as F
from data_util.dtype_utils import get_torch_dtype
import spconv.pytorch as spconv

class ShapeFeatureExtractor(nn.Module):
    def __init__(self, input_channels=6, output_dim=1024):
        super(ShapeFeatureExtractor, self).__init__()
        self.layer1 = nn.Conv1d(input_channels, 64, 1)
        self.layer2 = nn.Conv1d(64, 128, 1)
        self.layer3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.activation = nn.Sigmoid()
        
        
    def forward(self, x):
        x = self.activation(self.bn1(self.layer1(x)))
        x = self.activation(self.bn2(self.layer2(x)))
        x = self.activation(self.bn3(self.layer3(x)))
        
        x = torch.max(x, dim=2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x


class SDFDecoder(nn.Module):
    def __init__(self, input_dim=1024):
        super(SDFDecoder, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        x = F.sigmoid(x)
        return x
