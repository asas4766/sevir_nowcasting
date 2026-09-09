"""
STEP 2: PREPROCESSING - PyTorch Dataset for SEVIR VIL nowcasting
==================================================================
Turns each (384, 384, 49) storm event into multiple (4-in, 4-out)
training samples via a sliding window, resizes frames to 128x128,
and applies a log1p transform + normalization to tame VIL's skew
(VIL is mostly 0 with rare large spikes during storms).
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# log1p(255) is the max possible value after log-transforming a uint8 byte (0-255).
# We divide by this constant so normalized values land roughly in [0, 1].
LOG_NORM_CONST = math.log1p(255.0)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Raw VIL byte values (0-255) -> log1p -> scaled to ~[0,1]."""
    return torch.log1p(x) / LOG_NORM_CONST


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Inverse of normalize(): back to raw VIL byte-value scale (0-255).
    Needed later in Step 5/6 to compute CSI/POD/FAR and plot GIFs in
    physically meaningful units instead of the normalized log space."""
    return torch.expm1(x * LOG_NORM_CONST)


class SEVIRNowcastDataset(Dataset):
    """
    vil_array: numpy array, shape (N_events, H, W, T), dtype uint8
               (this is exactly what Step 1 saved as train_vil / test_vil)

    Each event has T=49 frames (5-min cadence, ~4 hours). We slide an
    8-frame window (4 input + 4 target) across each event's timeline
    with the given stride, producing several samples per event so 10-25
    events still yield a reasonable-sized training set.
    """

    def __init__(self, vil_array: np.ndarray, seq_in: int = 4, seq_out: int = 4,
                 crop_size: int = 128, stride: int = 4):
        assert vil_array.ndim == 4, f"expected (N,H,W,T), got {vil_array.shape}"
        self.data = vil_array
        self.seq_in = seq_in
        self.seq_out = seq_out
        self.seq_total = seq_in + seq_out
        self.crop_size = crop_size

        n_events, H, W, T = vil_array.shape
        self.samples = []
        for e in range(n_events):
            last_start = T - self.seq_total
            if last_start < 0:
                continue  # event too short for even one full window (shouldn't happen with T=49)
            for start in range(0, last_start + 1, stride):
                self.samples.append((e, start))

        if len(self.samples) == 0:
            raise ValueError(
                f"No samples could be built: each event needs at least "
                f"{self.seq_total} frames, but T={T}."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        event_idx, start = self.samples[i]
        # (H, W, seq_total) -> (seq_total, H, W)
        seq = self.data[event_idx, :, :, start:start + self.seq_total]
        seq = np.transpose(seq, (2, 0, 1)).astype(np.float32)
        seq_t = torch.from_numpy(seq)  # (seq_total, H, W)

        # Resize every frame to crop_size x crop_size in one batched call.
        # interpolate expects (N, C, H, W); treat each frame as its own "batch" item with 1 channel.
        seq_t = seq_t.unsqueeze(1)  # (seq_total, 1, H, W)
        seq_t = F.interpolate(seq_t, size=(self.crop_size, self.crop_size),
                               mode="bilinear", align_corners=False)
        seq_t = seq_t.squeeze(1)  # (seq_total, crop_size, crop_size)

        seq_t = normalize(seq_t)

        x = seq_t[:self.seq_in]   # (4, 128, 128) - input frames
        y = seq_t[self.seq_in:]   # (4, 128, 128) - target frames
        return x, y


if __name__ == "__main__":
    # Quick self-test against the real Step 1 output file.
    npz = np.load("sevir_vil_subset.npz")
    train_ds = SEVIRNowcastDataset(npz["train_vil"])
    test_ds = SEVIRNowcastDataset(npz["test_vil"])

    print(f"Train samples: {len(train_ds)} (from {npz['train_vil'].shape[0]} events)")
    print(f"Test samples:  {len(test_ds)} (from {npz['test_vil'].shape[0]} events)")

    x, y = train_ds[0]
    print(f"x shape: {tuple(x.shape)}, dtype: {x.dtype}")
    print(f"y shape: {tuple(y.shape)}, dtype: {y.dtype}")
    print(f"x value range (normalized): [{x.min():.4f}, {x.max():.4f}]")

    # Round-trip check: normalize -> denormalize should recover the original byte values (within float error)
    raw = torch.from_numpy(npz["train_vil"][0, :, :, 0].astype(np.float32))
    recovered = denormalize(normalize(raw))
    max_err = (raw - recovered).abs().max().item()
    print(f"normalize/denormalize round-trip max error: {max_err:.6f} (should be ~0)")
