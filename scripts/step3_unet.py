"""
STEP 3: THE U-NET MODEL
==================================================================
Standard U-Net: 4 input channels (4 historical VIL frames) ->
4 output channels (4 future VIL frames), with skip connections
between the encoder and decoder at each resolution level.

Sized for 128x128 inputs, small enough to train a few epochs on
CPU or a single GPU within a day's time budget.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) x2 -- the standard U-Net building block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downscaling: maxpool then DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """Upscaling then concat with the matching encoder skip connection, then DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # in_channels here is the number of channels coming up from below (already halved from
        # the transpose conv); after concatenating the skip connection it doubles back before DoubleConv.
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle any off-by-one size mismatch from odd input dimensions (not an issue at 128x128,
        # but keeps the model robust if someone changes crop_size later).
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                                       diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for precipitation nowcasting.

    Input:  (B, in_channels=4, 128, 128)  -- 4 historical VIL frames stacked as channels
    Output: (B, out_channels=4, 128, 128) -- 4 predicted future VIL frames
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 4, base_channels: int = 32):
        super().__init__()
        c = base_channels  # 32 by default -- kept small so training is fast on CPU/single GPU

        self.inc = DoubleConv(in_channels, c)          # 128x128, c
        self.down1 = Down(c, c * 2)                    # 64x64,   2c
        self.down2 = Down(c * 2, c * 4)                # 32x32,   4c
        self.down3 = Down(c * 4, c * 8)                # 16x16,   8c
        self.down4 = Down(c * 8, c * 16)               # 8x8,     16c (bottleneck)

        self.up1 = Up(c * 16, c * 8)                   # -> 16x16, 8c
        self.up2 = Up(c * 8, c * 4)                    # -> 32x32, 4c
        self.up3 = Up(c * 4, c * 2)                    # -> 64x64, 2c
        self.up4 = Up(c * 2, c)                        # -> 128x128, c

        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)

        # Predicting VIL, which is non-negative after our log-normalization (~[0,1]).
        # Sigmoid keeps outputs bounded in [0,1], matching the normalized target range.
        self.out_activation = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)       # skip 1
        x2 = self.down1(x1)    # skip 2
        x3 = self.down2(x2)    # skip 3
        x4 = self.down3(x3)    # skip 4
        x5 = self.down4(x4)    # bottleneck

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.outc(x)
        return self.out_activation(x)


if __name__ == "__main__":
    model = UNet(in_channels=4, out_channels=4, base_channels=32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameter count: {n_params:,}")

    dummy_x = torch.randn(2, 4, 128, 128)  # batch of 2, matching Step 4's batch size
    out = model(dummy_x)
    print(f"Input shape:  {tuple(dummy_x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    assert out.shape == (2, 4, 128, 128), "Output shape mismatch!"
    print(f"Output value range: [{out.min().item():.4f}, {out.max().item():.4f}] (should be within [0,1] due to sigmoid)")
    print("\n✅ U-Net forward pass verified.")
