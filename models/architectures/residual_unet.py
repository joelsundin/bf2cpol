"""
This architecture is an adapted and modified representation of the workflow described in:

Keikhosravi et al. (2020), "Non-disruptive collagen characterization in clinical histopathology using cross-modality image synthesis."
Original implementation:
https://github.com/uw-loci/he_shg_synth_workflow
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
Processed skip connection

    Conv(k=3,s=1) -> BNorm -> LeakyReLU -> Conv(k=1,s=1) -> BNorm -> LeakyReLU
"""
class Skip(nn.Module):
  def __init__(self, in_ch, out_ch):
    super(Skip, self).__init__()
    self.skip = nn.Sequential(
      nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
      nn.BatchNorm2d(out_ch),
      nn.LeakyReLU(negative_slope=0.1),
      nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding_mode="reflect"),
      nn.BatchNorm2d(out_ch),
      nn.LeakyReLU(negative_slope=0.1)
    )

  def forward(self, x):
    return self.skip(x)

"""
Residual downsampling block, H,W -> H/2, W/2.

    main:     Conv(k=4,s=2) -> BNorm -> LeakyReLU -> Conv(k=1,s=1) -> BNorm -> LeakyReLU
    residual: Conv(k=2,s=2)
    out = main(x) + residual(x)
Both paths stride 2, shapes match and enable for adding.
"""

class EncodeDownsample(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(EncodeDownsample, self).__init__()
        layers = [
            nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(out_size, out_size, kernel_size=1, stride=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_size),
            nn.LeakyReLU(negative_slope=0.1)
        ]
        
        if dropout:
            layers.append(nn.Dropout(dropout))

        residual = [
            nn.Conv2d(in_size, out_size, kernel_size=2, stride=2, bias=False, padding_mode="reflect"),
        ]

        self.main = nn.Sequential(*layers)
        self.residual = nn.Sequential(*residual)

    def forward(self, x):
        return self.main(x) + self.residual(x)

"""
Residual upsampling block + skip fusion, H,W -> 2H, 2W.

    x = Upsample(x, scale=2)        # bilinear; avoids transposed-conv checkerboard
    main:     Conv(k=3,s=1) -> BNorm -> ReLU -> Conv(k=1,s=1) -> BNorm -> ReLU
    residual: Conv(k=1,s=1)
    out = Channel_cat( main(x) + residual(x),  skip_connection)
"""

class DecodeUpsample(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(DecodeUpsample, self).__init__()

        layers = [
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=1, stride=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        residual = [
            nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, bias=False, padding_mode="reflect"),
        ]

        self.main = nn.Sequential(*layers)
        self.residual = nn.Sequential(*residual)

    def forward(self, x, skip_input):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.main(x) + self.residual(x)
        x = torch.cat((x, skip_input), dim=1)
        return x

"""
Residual U-Net, 512x512 -> 512x512.

Encoder d1..d6 resolution:  256 -> 128 -> 64 -> 32 -> 16 -> 8   (input 512)
Decoder Channel_cats skip(d_k) at each step, so up2 onward take doubled channels.

Output head:
    Conv(k=3,s=1):    128 -> 4*out_channels
    PixelShuffle(2):  (4*out, H, W) -> (out, 2H, 2W)   # 256 -> 512
    Sigmoid:          range -> [0, 1]
"""

class ResidualUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(ResidualUnet, self).__init__()

        #encoder
        self.down1 = EncodeDownsample(in_channels, 64)
        self.skip1 = Skip(64, 64)
        self.down2 = EncodeDownsample(64, 128)
        self.skip2 = Skip(128, 128)
        self.down3 = EncodeDownsample(128, 256)
        self.skip3 = Skip(256, 256)
        self.down4 = EncodeDownsample(256, 512)
        self.skip4 = Skip(512, 512)

        #bottleneck
        self.down5 = EncodeDownsample(512, 512)
        self.skip5 = Skip(512, 512)
        self.down6 = EncodeDownsample(512, 512)

        #decoder
        self.up1 = DecodeUpsample(512, 512)
        self.up2 = DecodeUpsample(1024, 512)
        self.up3 = DecodeUpsample(1024, 256)
        self.up4 = DecodeUpsample(512, 128)
        self.up5 = DecodeUpsample(256, 64)

        self.final = nn.Sequential(
            nn.Conv2d(128, out_channels * 4, 3, 1, 1), # [128 -> 12]
            nn.PixelShuffle(2), # latent depth -> RGB
            nn.Sigmoid() # [0,1]
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        
        # concat skip
        u1 = self.up1(d6, self.skip5(d5)) 
        u2 = self.up2(u1, self.skip4(d4))
        u3 = self.up3(u2, self.skip3(d3))
        u4 = self.up4(u3, self.skip2(d2))
        u5 = self.up5(u4, self.skip1(d1))
        return self.final(u5)
