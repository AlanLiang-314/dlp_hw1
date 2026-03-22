import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Contracting Path (Encoder)
        self.down1 = DoubleConv(n_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Expansive Path (Decoder)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up4 = DoubleConv(1024, 512) # 512 (from upconv) + 512 (from skip connection)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = DoubleConv(128, 64)

        # Final Layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def crop_tensor(self, source, target):
        """
        Crops the source tensor to the size of the target tensor.
        Required because of the loss of pixels in unpadded convolutions.
        """
        _, _, h, w = target.shape
        _, _, source_h, source_w = source.shape
        
        delta_h = (source_h - h) // 2
        delta_w = (source_w - w) // 2
        
        return source[:, :, delta_h:delta_h+h, delta_w:delta_w+w]

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        p1 = self.pool(x1)
        
        x2 = self.down2(p1)
        p2 = self.pool(x2)
        
        x3 = self.down3(p2)
        p3 = self.pool(x3)
        
        x4 = self.down4(p3)
        p4 = self.pool(x4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        d4 = self.upconv4(b)
        x4_cropped = self.crop_tensor(x4, d4)
        d4 = self.up4(torch.cat([x4_cropped, d4], dim=1))
        
        d3 = self.upconv3(d4)
        x3_cropped = self.crop_tensor(x3, d3)
        d3 = self.up3(torch.cat([x3_cropped, d3], dim=1))
        
        d2 = self.upconv2(d3)
        x2_cropped = self.crop_tensor(x2, d2)
        d2 = self.up2(torch.cat([x2_cropped, d2], dim=1))
        
        d1 = self.upconv1(d2)
        x1_cropped = self.crop_tensor(x1, d1)
        d1 = self.up1(torch.cat([x1_cropped, d1], dim=1))

        return self.outc(d1)

if __name__ == "__main__":
    model = UNet(n_channels=1, n_classes=2)
    x = torch.randn(1, 1, 572, 572)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}") # [1, 2, 388, 388]