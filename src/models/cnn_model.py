"""The 1D convolutional network: architecture, training and prediction (PyTorch)."""
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

EPOCHS = 30
BATCH = 64
LEARNING_RATE = 1e-3
SEED = 42


def make_branch(pool):
    """Three convolution blocks that turn one view into a short feature vector."""
    layers, channels = [], 1
    for width in (16, 32, 64):
        layers += [nn.Conv1d(channels, width, kernel_size=5, padding=2),
                   nn.BatchNorm1d(width), nn.ReLU(), nn.MaxPool1d(pool)]
        channels = width
    layers += [nn.AdaptiveAvgPool1d(8), nn.Flatten()]
    return nn.Sequential(*layers)


class TwoBranchCNN(nn.Module):
    """One branch reads the global view (2001 bins), the other the local view (201 bins)."""

    def __init__(self):
        super().__init__()
        self.global_branch = make_branch(pool=4)
        self.local_branch = make_branch(pool=2)
        self.head = nn.Sequential(
            nn.Dropout(0.4), nn.Linear(2 * 64 * 8, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, global_view, local_view):
        joined = torch.cat([self.global_branch(global_view),
                            self.local_branch(local_view)], dim=1)
        return self.head(joined).squeeze(1)


def predict_proba(model, xg, xl):
    """Probability of the planet class for arrays of shape (n, 2001) and (n, 201)."""
    model.eval()
    xg = torch.from_numpy(xg).float().unsqueeze(1)
    xl = torch.from_numpy(xl).float().unsqueeze(1)
    out = []
    with torch.no_grad():
        for s in range(0, len(xg), 256):
            out.append(torch.sigmoid(model(xg[s:s + 256], xl[s:s + 256])).numpy())
    return np.concatenate(out).astype(float)


def train_model(xg, xl, y, epochs=EPOCHS, seed=SEED, val=None):
    """Train from scratch. val = (xg, xl, y) is only used to log a learning curve.

    Returns (model, history) where history has one (train_loss, val_pr_auc) per epoch.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TwoBranchCNN()
    n_pos = float(np.sum(y))
    pos_weight = torch.tensor((len(y) - n_pos) / n_pos, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    g_all = torch.from_numpy(xg).float().unsqueeze(1)
    l_all = torch.from_numpy(xl).float().unsqueeze(1)
    y_all = torch.from_numpy(np.asarray(y)).float()
    n = len(y_all)
    history = []
    for _ in range(epochs):
        model.train()
        order = torch.from_numpy(rng.permutation(n))
        total = 0.0
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            if len(idx) < 2:  # BatchNorm needs at least two samples
                continue
            g, l, t = g_all[idx], l_all[idx], y_all[idx]
            # Augmentation: a transit is symmetric, so mirror both views together,
            # and add a little noise.
            flip = torch.from_numpy(rng.random(len(idx)) < 0.5)[:, None, None]
            g = torch.where(flip, g.flip(-1), g) + 0.05 * torch.randn_like(g)
            l = torch.where(flip, l.flip(-1), l) + 0.05 * torch.randn_like(l)
            optimizer.zero_grad()
            loss = loss_fn(model(g, l), t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(idx)
        scheduler.step()
        val_pr = np.nan
        if val is not None:
            val_pr = average_precision_score(val[2], predict_proba(model, val[0], val[1]))
        history.append((total / n, val_pr))
    return model, history


def save_model(model, path):
    torch.save(model.state_dict(), path)