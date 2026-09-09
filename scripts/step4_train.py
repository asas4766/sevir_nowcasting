"""
STEP 4: TRAINING LOOP
==================================================================
Trains the U-Net on the 20 (or 10, if you kept the smaller subset)
train events for 5 epochs, batch size 2.

KEY ISSUE THIS ADDRESSES: VIL is extremely skewed -- the vast
majority of pixels in any frame are 0 (no rain). A model trained
with plain MSE/L1 loss can achieve a deceptively low loss just by
predicting all zeros everywhere, since that's "close enough" for
90%+ of pixels. To prevent this collapse, we weight the loss so
pixels with higher (normalized) VIL intensity count more.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from step2_dataset import SEVIRNowcastDataset
from step3_unet import UNet


class WeightedMSELoss(nn.Module):
    """
    MSE loss where each pixel is weighted by its TARGET intensity.
    Since targets are normalized VIL in [0,1] (0 = no rain, 1 = max VIL
    byte value), weight = 1 + rain_weight * target gives heavy-rain
    pixels up to (1 + rain_weight)x the influence of a no-rain pixel,
    directly countering the class imbalance that causes all-zero
    predictions.
    """

    def __init__(self, rain_weight: float = 9.0):
        super().__init__()
        self.rain_weight = rain_weight

    def forward(self, pred, target):
        weights = 1.0 + self.rain_weight * target
        loss = weights * (pred - target) ** 2
        return loss.mean()


def train(train_npz_path: str, epochs: int = 5, batch_size: int = 2,
          lr: float = 1e-3, rain_weight: float = 9.0, save_path: str = "unet_sevir.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    npz = np.load(train_npz_path)
    train_ds = SEVIRNowcastDataset(npz["train_vil"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Training samples: {len(train_ds)} | Batches per epoch: {len(train_loader)}")

    model = UNet(in_channels=4, out_channels=4, base_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = WeightedMSELoss(rain_weight=rain_weight)

    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        history.append(avg_loss)
        print(f"Epoch {epoch}/{epochs} | Weighted MSE Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")

    # Sanity check: is the model actually predicting non-trivial rain, or collapsing to all-zero?
    model.eval()
    with torch.no_grad():
        x, y = next(iter(train_loader))
        pred = model(x.to(device)).cpu()
    print(f"\nSanity check on one training batch:")
    print(f"  Target mean/max: {y.mean():.4f} / {y.max():.4f}")
    print(f"  Pred   mean/max: {pred.mean():.4f} / {pred.max():.4f}")
    if pred.max() < 0.05 and y.max() > 0.2:
        print("  WARNING: prediction max is near-zero while target has real signal -- "
              "possible collapse to all-zero. Consider increasing rain_weight or epochs.")
    else:
        print("  Looks reasonable -- model is producing non-trivial spatial variation.")

    return model, history


if __name__ == "__main__":
    # NOTE: update this path to wherever your Step 1 output actually lives, e.g.
    # "/content/sevir_data/sevir_vil_subset.npz" in Colab.
    model, history = train("/content/sevir_data/sevir_vil_subset.npz", epochs=5, batch_size=2)

    # Loss should trend downward across epochs (won't be perfectly monotonic with only
    # a handful of events and batches, but the overall trend should decrease).
    print(f"\nLoss history: {[f'{l:.6f}' for l in history]}")
    if history[-1] < history[0]:
        print("Loss decreased from first to last epoch.")
    else:
        print("WARNING: loss did not decrease -- check learning rate / data before proceeding.")
