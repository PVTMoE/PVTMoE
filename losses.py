import torch
import torch.nn.functional as F


def balance_loss(w):
    m = w.mean(0)
    return m.var(unbiased=False) / m.mean() ** 2


def supcon_loss(z, y, tau=0.1):
    eye = torch.eye(len(y), dtype=torch.bool, device=z.device)
    with torch.autocast(z.device.type, enabled=False):
        logp = (z.float() @ z.float().T / tau).masked_fill(eye, -1e9)
    logp = logp - logp.logsumexp(1, keepdim=True)
    pos = (y[:, None] == y[None]) & ~eye
    n = pos.sum(1)
    ok = n > 0
    if not ok.any():
        return z.sum() * 0
    return -((logp * pos).sum(1)[ok] / n[ok]).mean()


def total_loss(logits, z, w, y, lam=0.5, beta=0.01):
    ce = F.cross_entropy(logits, y, label_smoothing=0.05)
    return ce + beta * balance_loss(w) + lam * supcon_loss(z, y)
