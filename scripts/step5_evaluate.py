"""
STEP 5: METEOROLOGICAL EVALUATION
==================================================================
Evaluates the trained U-Net on the 5 held-out test events using the
standard categorical verification metrics from meteorology:

  POD (Probability of Detection) = hits / (hits + misses)
      -- of all the times it actually rained, how often did we catch it?
  FAR (False Alarm Ratio)        = false_alarms / (hits + false_alarms)
      -- of all the times we predicted rain, how often were we wrong?
  CSI (Critical Success Index)   = hits / (hits + misses + false_alarms)
      -- overall skill score, punishes both misses AND false alarms
         (unlike accuracy, which is misleadingly high for rare events
         just by predicting "no rain" everywhere)

Threshold: byte value >= 74 (out of 0-255) is used as the SEVIR-standard
convective rain/no-rain threshold (this is what the SEVIR benchmark
papers use -- NOT the physical "VIL > 1.0 kg/m^2" from the original
naive plan, since VIL is byte-encoded, not stored in physical units
directly).

Baseline: "Persistence" simply copies the LAST INPUT FRAME forward as
the prediction for all 4 future frames. It's the standard baseline in
nowcasting literature -- a good model must beat it, since persistence
is a genuinely strong baseline for very-short-lead-time forecasts.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from step2_dataset import SEVIRNowcastDataset, denormalize
from step3_unet import UNet

RAIN_THRESHOLD_BYTE = 74.0  # SEVIR convective-storm threshold, on the 0-255 raw byte scale


def compute_confusion_counts(pred_byte: torch.Tensor, target_byte: torch.Tensor, threshold: float):
    """
    pred_byte, target_byte: tensors of any shape, already de-normalized to the
    0-255 raw VIL byte scale. Returns pooled hit/miss/false_alarm/correct_negative
    counts across every pixel and every sample passed in.
    """
    pred_rain = pred_byte >= threshold
    target_rain = target_byte >= threshold

    hits = (pred_rain & target_rain).sum().item()
    misses = (~pred_rain & target_rain).sum().item()
    false_alarms = (pred_rain & ~target_rain).sum().item()
    correct_neg = (~pred_rain & ~target_rain).sum().item()
    return hits, misses, false_alarms, correct_neg


def scores_from_counts(hits, misses, false_alarms, eps=1e-8):
    pod = hits / (hits + misses + eps)
    far = false_alarms / (hits + false_alarms + eps)
    csi = hits / (hits + misses + false_alarms + eps)
    return {"POD": pod, "FAR": far, "CSI": csi, "hits": hits, "misses": misses, "false_alarms": false_alarms}


def evaluate(test_npz_path: str, model_path: str, threshold: float = RAIN_THRESHOLD_BYTE,
             batch_size: int = 4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    npz = np.load(test_npz_path)
    test_ds = SEVIRNowcastDataset(npz["test_vil"])
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = UNet(in_channels=4, out_channels=4, base_channels=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    n_lead_times = None
    unet_counts = None       # pooled, overall
    persist_counts = None
    unet_counts_by_lead = None   # per-lead-time (t+1, t+2, t+3, t+4) breakdown
    persist_counts_by_lead = None

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            if n_lead_times is None:
                n_lead_times = y.shape[1]
                unet_counts = np.zeros(3)
                persist_counts = np.zeros(3)
                unet_counts_by_lead = np.zeros((n_lead_times, 3))
                persist_counts_by_lead = np.zeros((n_lead_times, 3))

            # --- U-Net prediction ---
            pred = model(x)
            pred_byte = denormalize(pred.cpu())
            target_byte = denormalize(y.cpu())

            h, m, fa, _ = compute_confusion_counts(pred_byte, target_byte, threshold)
            unet_counts += np.array([h, m, fa])

            # --- Persistence baseline: repeat the LAST input frame 4 times ---
            last_input_frame = x[:, -1:, :, :].cpu()             # (B, 1, 128, 128)
            persistence_pred = last_input_frame.repeat(1, y.shape[1], 1, 1)  # (B, 4, 128, 128)
            persistence_byte = denormalize(persistence_pred)

            h_p, m_p, fa_p, _ = compute_confusion_counts(persistence_byte, target_byte, threshold)
            persist_counts += np.array([h_p, m_p, fa_p])

            # --- Per-lead-time breakdown (this is the informative view) ---
            for t in range(n_lead_times):
                h_t, m_t, fa_t, _ = compute_confusion_counts(pred_byte[:, t], target_byte[:, t], threshold)
                unet_counts_by_lead[t] += np.array([h_t, m_t, fa_t])

                h_pt, m_pt, fa_pt, _ = compute_confusion_counts(persistence_byte[:, t], target_byte[:, t], threshold)
                persist_counts_by_lead[t] += np.array([h_pt, m_pt, fa_pt])

    unet_scores = scores_from_counts(*unet_counts)
    persist_scores = scores_from_counts(*persist_counts)

    print(f"Rain/no-rain threshold: VIL byte value >= {threshold}")
    print(f"Test samples evaluated: {len(test_ds)}\n")

    print("=== Overall (pooled across all 4 lead times) ===")
    print(f"{'Metric':<8}{'U-Net':>12}{'Persistence':>14}")
    for key in ["POD", "FAR", "CSI"]:
        print(f"{key:<8}{unet_scores[key]:>12.4f}{persist_scores[key]:>14.4f}")

    print("\n=== CSI by lead time (this is the metric that matters most) ===")
    print(f"{'Lead time':<12}{'U-Net CSI':>12}{'Persist. CSI':>14}{'U-Net wins?':>14}")
    unet_wins_any = False
    for t in range(n_lead_times):
        u_csi = scores_from_counts(*unet_counts_by_lead[t])["CSI"]
        p_csi = scores_from_counts(*persist_counts_by_lead[t])["CSI"]
        wins = u_csi > p_csi
        unet_wins_any = unet_wins_any or wins
        print(f"t+{t+1:<10}{u_csi:>12.4f}{p_csi:>14.4f}{'YES' if wins else 'no':>14}")

    print()
    if unet_scores["CSI"] > persist_scores["CSI"]:
        improvement = (unet_scores["CSI"] - persist_scores["CSI"]) / (persist_scores["CSI"] + 1e-8) * 100
        print(f"U-Net beats persistence on OVERALL CSI by {improvement:.1f}% relative improvement.")
    elif unet_wins_any:
        print("U-Net does not beat persistence overall, but DOES beat it at longer lead times "
              "(see table above) -- this is the expected, publishable pattern in nowcasting: "
              "persistence is strong immediately but degrades as the storm evolves, while the "
              "learned model's relative advantage grows with lead time.")
    else:
        print("U-Net did not beat persistence at any lead time. With this little training data "
              "(10-20 events, 5 epochs) that can happen. Try: more epochs, more real training "
              "events (closer to the original 20), or a higher rain_weight in Step 4. This is "
              "still worth reporting honestly -- an interviewer will respect an honest limitations "
              "section far more than an inflated claim.")

    return unet_scores, persist_scores, unet_counts_by_lead, persist_counts_by_lead


if __name__ == "__main__":
    results = evaluate(
        test_npz_path="/content/sevir_data/sevir_vil_subset.npz",
        model_path="/content/unet_sevir.pt",
    )
