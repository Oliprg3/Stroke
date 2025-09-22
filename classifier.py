import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SliceEncoder2p5D(nn.Module):
    """Encode a k-slice stack via channel concatenation into a 2D backbone."""
    def __init__(self, backbone='resnet34', k=5, pretrained=True, out_dim=256):
        super().__init__()
        if backbone == 'resnet34':
            net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            in_planes = 64
            self.stem = nn.Sequential(
                nn.Conv2d(k, 64, kernel_size=7, stride=2, padding=3, bias=False),
                net.bn1, net.relu, net.maxpool
            )
            self.enc = nn.Sequential(net.layer1, net.layer2, net.layer3, net.layer4)
            self.pool = nn.AdaptiveAvgPool2d(1)
            feat = net.fc.in_features
        else:
            raise NotImplementedError(backbone)
        self.head = nn.Sequential(
            nn.Linear(feat, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(self, x):
    
        x = self.stem(x)
        x = self.enc(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return x  

class AttentionAggregator(nn.Module):
    def __init__(self, in_dim=256, hidden=256, num_classes=2):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, bidirectional=True, batch_first=True)
        self.attn = nn.Linear(hidden*2, 1)
        self.cls = nn.Linear(hidden*2, num_classes)

    def forward(self, feats, lengths):
      
        packed = nn.utils.rnn.pack_padded_sequence(feats, lengths=lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        w = self.attn(out)  
        mask = torch.arange(out.size(1), device=out.device)[None, :] >= torch.tensor(lengths, device=out.device)[:, None]
        w.masked_fill_(mask.unsqueeze(-1), float('-inf'))
        a = torch.softmax(w, dim=1) 
        ctx = (a * out).sum(dim=1)  
        logits = self.cls(ctx)
        return logits, a.squeeze(-1)
