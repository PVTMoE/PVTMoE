# PVT-MoE

Example code for a mixture-of-experts classifier of portal vein thrombosis (bland vs. neoplastic) on two-phase CT.

| File | Content |
|---|---|
| `model.py` | `PVTMoE` |
| `losses.py` | training losses |
| `dataset.py` | cropping, augmentation, dataset |
| `engine.py` | prediction and metrics |
| `train.ipynb` | training, writes `model.pt` |
| `predict.ipynb` | validation prediction, writes `predictions.csv` |

## Data format

- `ROOT/{id}.npy`: float array `(2, D, H, W)` in HU, phases `[arterial, portal]`.
- `ROOT/table.csv`: columns `id, label, cz, cy, cx, split`; `label` 1 = neoplastic, `(cz, cy, cx)` = box centre in voxels, `split` = `train` / `val`.

## Usage

```bash
pip install -r requirements.txt
jupyter lab
```

Set `ROOT` in `train.ipynb` and `predict.ipynb`, then run them in that order. `PRETRAINED = False` skips weight download.
