import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers import DropPath

NAMES = ("efficientnet_b4", "resnet50", "vit_small_patch14_dinov2.lvd142m")
CHANNELS = (1792, 2048, 384)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Encoder(nn.Module):
    def __init__(self, name, pretrained):
        super().__init__()
        kw = {"img_size": 224} if "vit" in name else {}
        self.net = timm.create_model(name, pretrained=pretrained, in_chans=6,
                                     num_classes=0, global_pool="", **kw)
        self.vit = "vit" in name

    def forward(self, x):
        f = self.net.forward_features(x)
        if self.vit:
            f = f[:, self.net.num_prefix_tokens:]
            n = int(f.shape[1] ** 0.5)
            f = F.adaptive_avg_pool2d(f.transpose(1, 2).reshape(f.shape[0], -1, n, n), 7)
        return f


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.1, layer_scale=0.1):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        y = self.norm(self.dw(x).permute(0, 2, 3, 1))
        y = self.fc2(F.gelu(self.fc1(y))) * self.gamma
        return x + self.drop_path(y.permute(0, 3, 1, 2))


class PVTMoE(nn.Module):
    def __init__(self, pretrained=True, width=100, n_classes=2):
        super().__init__()
        dim = len(NAMES) * width
        self.enc = nn.ModuleList(Encoder(n, pretrained) for n in NAMES)
        self.proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, width, 1), LayerNorm2d(width)) for c in CHANNELS)
        self.gate = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, len(NAMES)))
        self.block = ConvNeXtBlock(dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(0.2)
        self.cls = nn.Linear(dim, n_classes)
        self.proj_head = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 128))

    def train(self, mode=True):
        super().train(mode)
        for m in self.enc.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
        return self

    def forward(self, x):
        p = [proj(enc(x)) for enc, proj in zip(self.enc, self.proj)]
        w = self.gate(torch.cat([q.mean((2, 3)) for q in p], 1)).float().softmax(1)
        z = torch.cat([len(NAMES) * w[:, i, None, None, None] * q for i, q in enumerate(p)], 1)
        h = self.norm(self.block(z).mean((2, 3)))
        return self.cls(self.drop(h)), F.normalize(self.proj_head(h).float(), dim=1), w
