"""
STEP 6: VISUALIZATION (CRITICAL FOR CV)
==================================================================
Generates a single GIF for the single most dramatic storm in the test
set: 3 columns (Ground Truth | Persistence | U-Net), 4 frames animating
through the 4 predicted lead times (t+1 .. t+4), dark background,
'turbo' colormap.

"Most dramatic" = the test sample with the highest total rain-pixel
intensity in its target frames -- i.e. the storm with the strongest
actual precipitation signal, which makes for the most visually
convincing (and most meteorologically meaningful) comparison.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
import imageio.v2 as imageio
import io

from step2_dataset import SEVIRNowcastDataset, denormalize
from step3_unet import UNet

matplotlib.use("Agg")  # headless rendering, safe for both scripts and notebooks


def find_most_dramatic_sample(test_ds, device):
    """Returns the index (into test_ds) of the sample with the highest total
    target intensity -- i.e. the storm with the most actual rain to show off."""
    best_idx, best_score = 0, -1.0
    for i in range(len(test_ds)):
        _, y = test_ds[i]
        score = denormalize(y).sum().item()
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx, best_score


def make_comparison_gif(test_npz_path: str, model_path: str, output_path: str = "nowcast_comparison.gif",
                         vmax_byte: float = 180.0, fps: int = 2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    npz = np.load(test_npz_path)
    test_ds = SEVIRNowcastDataset(npz["test_vil"])

    model = UNet(in_channels=4, out_channels=4, base_channels=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Scanning test set for the most dramatic storm...")
    idx, score = find_most_dramatic_sample(test_ds, device)
    print(f"Selected test sample {idx} (total target intensity: {score:.0f})")

    x, y = test_ds[idx]  # x: (4,128,128) input, y: (4,128,128) target -- normalized [0,1]

    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device)).squeeze(0).cpu()

    # Persistence: last input frame repeated across all 4 lead times
    persistence = x[-1:].repeat(4, 1, 1)

    # De-normalize everything back to the raw 0-255 VIL byte scale for physically meaningful plotting
    gt_byte = denormalize(y).numpy()
    persist_byte = denormalize(persistence).numpy()
    pred_byte = denormalize(pred).numpy()

    cmap = plt.get_cmap("turbo").copy()
    cmap.set_under("black")  # true zero-rain pixels render as pure black on the dark background

    frames = []
    n_lead = gt_byte.shape[0]
    for t in range(n_lead):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), facecolor="black")
        panels = [
            (gt_byte[t], "Ground Truth"),
            (persist_byte[t], "Persistence"),
            (pred_byte[t], "U-Net Prediction"),
        ]
        for ax, (frame, title) in zip(axes, panels):
            ax.set_facecolor("black")
            im = ax.imshow(frame, cmap=cmap, vmin=0.5, vmax=vmax_byte)
            ax.set_title(title, color="white", fontsize=13, fontweight="bold")
            ax.axis("off")

        fig.suptitle(f"Precipitation Nowcasting -- Lead time t+{t+1} (SEVIR VIL)",
                     color="white", fontsize=14, y=1.02)
        cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label("VIL (byte scale, 0-255)", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="black", bbox_inches="tight", dpi=110)
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    # Hold the last frame a bit longer so the final comparison is easy to study when the GIF loops
    frames_with_hold = frames + [frames[-1]] * 2

    imageio.mimsave(output_path, frames_with_hold, fps=fps, loop=0)
    print(f"GIF saved to {output_path} ({len(frames)} unique frames, {fps} fps)")
    return output_path


if __name__ == "__main__":
    make_comparison_gif(
        test_npz_path="/content/sevir_data/sevir_vil_subset.npz",
        model_path="/content/unet_sevir.pt",
        output_path="/content/nowcast_comparison.gif",
    )
