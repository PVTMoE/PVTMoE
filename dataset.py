import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset

LO, HI, SIZE = -125, 225, 224


def phase_stats(df, root):
    s = np.zeros((2, 3))
    for i in df["id"]:
        v = np.clip(np.load(Path(root) / f"{i}.npy").astype(np.float64), LO, HI)
        s += [[v[p].size, v[p].sum(), (v[p] ** 2).sum()] for p in range(2)]
    mean = s[:, 1] / s[:, 0]
    return mean.tolist(), np.sqrt(s[:, 2] / s[:, 0] - mean ** 2).tolist()


def augment(x):
    a = np.deg2rad(np.random.uniform(-15, 15))
    s = np.random.uniform(0.9, 1.1)
    t = np.random.uniform(-8, 8, 2) * 2 / SIZE
    theta = torch.tensor([[np.cos(a) / s, -np.sin(a) / s, t[0]],
                          [np.sin(a) / s, np.cos(a) / s, t[1]]], dtype=torch.float32)[None]
    grid = F.affine_grid(theta, (1, 6, SIZE, SIZE), align_corners=False)
    x = F.grid_sample(x[None], grid, padding_mode="border", align_corners=False)[0]
    return x * float(np.random.uniform(0.9, 1.1)) + 0.02 * torch.randn_like(x)


class PVTDataset(Dataset):
    def __init__(self, df, root, mean, std, train=False):
        self.df, self.root, self.train = df.reset_index(drop=True), Path(root), train
        self.mean = np.asarray(mean, np.float32).reshape(2, 1, 1, 1)
        self.std = np.asarray(std, np.float32).reshape(2, 1, 1, 1)

    def __len__(self):
        return len(self.df)

    def triplet(self, vol, c, cy, cx):
        idx = np.clip([c - 1, c, c + 1], 0, vol.shape[1] - 1)
        v = np.pad(vol[:, idx], ((0, 0), (0, 0), (112, 112), (112, 112)), constant_values=LO)
        v = v[:, :, cy:cy + SIZE, cx:cx + SIZE].astype(np.float32)
        v = (np.clip(v, LO, HI) - self.mean) / self.std
        return torch.from_numpy(v.reshape(6, SIZE, SIZE))

    def __getitem__(self, i):
        r = self.df.iloc[i]
        vol = np.load(self.root / f"{r['id']}.npy", mmap_mode="r")
        cz, cy, cx = int(r["cz"]), int(r["cy"]), int(r["cx"])
        if self.train:
            return augment(self.triplet(vol, cz + np.random.randint(-2, 3), cy, cx)), int(r["label"])
        x = torch.stack([self.triplet(vol, cz + d, cy, cx) for d in range(-2, 3)])
        return x, int(r["label"])

