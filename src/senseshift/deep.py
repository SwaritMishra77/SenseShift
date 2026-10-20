"""Compact raw-waveform baselines. Torch is an optional runtime dependency."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from sklearn.metrics import f1_score


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Install with `pip install -e '.[deep]'` to run CNN/transformer") from exc
    return torch, nn


def make_model(kind: str, classes: int):
    torch, nn = _torch()

    class SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 32, 7, padding=3, stride=2), nn.BatchNorm1d(32), nn.GELU(),
                nn.Conv1d(32, 64, 5, padding=2, stride=2), nn.BatchNorm1d(64), nn.GELU(),
                nn.Conv1d(64, 64, 3, padding=1, stride=2), nn.BatchNorm1d(64), nn.GELU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, classes),
            )

        def forward(self, x):
            return self.net(x[:, None, :])

    class PatchTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch = nn.Conv1d(1, 48, kernel_size=10, stride=10)
            self.position = nn.Parameter(torch.zeros(1, 21, 48))
            self.cls = nn.Parameter(torch.zeros(1, 1, 48))
            encoder = nn.TransformerEncoderLayer(
                d_model=48, nhead=4, dim_feedforward=96, dropout=0.1,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder, num_layers=2, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(48)
            self.head = nn.Linear(48, classes)

        def forward(self, x):
            patches = self.patch(x[:, None, :]).transpose(1, 2)
            cls = self.cls.expand(len(x), -1, -1)
            z = torch.cat([cls, patches], dim=1) + self.position
            return self.head(self.norm(self.encoder(z)[:, 0]))

    if kind == "cnn":
        return SmallCNN()
    if kind == "transformer":
        return PatchTransformer()
    raise ValueError(kind)


def fit_deep(kind: str, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray, *, seed: int,
             max_epochs: int = 12, batch_size: int = 256, patience: int = 3):
    torch, nn = _torch()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Normalize using only the training subjects of the source device.
    mu, sigma = float(X_train.mean()), max(float(X_train.std()), 1e-6)
    xtr = ((X_train - mu) / sigma).astype(np.float32)
    xval = ((X_val - mu) / sigma).astype(np.float32)
    model = make_model(kind, int(y_train.max()) + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(xtr), torch.from_numpy(y_train.astype(np.int64)))
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    history, best_score, best_state, stale = [], -1.0, None, 0
    for epoch in range(max_epochs):
        model.train()
        losses = []
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        logits = predict_deep(model, xval, already_scaled=True)
        score = float(f1_score(y_val, logits.argmax(1), average="macro", zero_division=0))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)),
                        "validation_macro_f1": score})
        if score > best_score + 1e-4:
            best_score, best_state, stale = score, deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, {"mean": mu, "std": sigma, "best_validation_macro_f1": best_score,
                   "history": history}


def predict_deep(model, X: np.ndarray, *, already_scaled: bool = False,
                 mean: float = 0.0, std: float = 1.0) -> np.ndarray:
    torch, _ = _torch()
    model.eval()
    if not already_scaled:
        X = ((X - mean) / std).astype(np.float32)
    output = []
    with torch.inference_mode():
        for start in range(0, len(X), 512):
            output.append(model(torch.from_numpy(X[start:start + 512])).numpy())
    return np.vstack(output)


def save_deep(model, path, metadata):
    torch, _ = _torch()
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
