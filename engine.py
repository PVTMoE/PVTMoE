import torch
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    P, Y = [], []
    for x, y in loader:
        b, k = x.shape[:2]
        logits = model(x.flatten(0, 1).to(device))[0].view(b, k, -1).mean(1)
        P.append(logits.softmax(1)[:, 1].cpu())
        Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def metrics(p, y, thr=0.5):
    assert len(set(y)) == 2, "metrics need both classes"
    pred = p >= thr
    tp, tn = (pred & (y == 1)).sum(), (~pred & (y == 0)).sum()
    fp = (pred & (y == 0)).sum()
    prec, sens = tp / max(tp + fp, 1), tp / (y == 1).sum()
    return dict(acc=float((tp + tn) / len(y)), sens=float(sens), spec=float(tn / (y == 0).sum()),
                prec=float(prec), f1=float(2 * prec * sens / max(prec + sens, 1e-9)),
                auc=float(roc_auc_score(y, p)))
