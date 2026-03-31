import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = DoubleConv(512, 1024)
        
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up1 = DoubleConv(1024, 512)
        
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = DoubleConv(512, 256)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3 = DoubleConv(256, 128)
        
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up4 = DoubleConv(128, 64)
        
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def crop(self, enc_features, x):
        """
        Center-crops the features from the contracting path to match 
        the spatial dimensions of the upsampled features.
        """
        _, _, H, W = x.shape
        _, _, enc_H, enc_W = enc_features.shape
        
        # Calculate start indices for center crop
        diff_Y = (enc_H - H) // 2
        diff_X = (enc_W - W) // 2
        
        return enc_features[:, :, diff_Y:diff_Y+H, diff_X:diff_X+W]

    def forward(self, x):
        # --- Contracting Path ---
        c1 = self.down1(x)
        p1 = self.pool1(c1)
        
        c2 = self.down2(p1)
        p2 = self.pool2(c2)
        
        c3 = self.down3(p2)
        p3 = self.pool3(c3)
        
        c4 = self.down4(p3)
        p4 = self.pool4(c4)
        
        # --- Bottleneck ---
        b = self.bottleneck(p4)
        
        # --- Expansive Path ---
        u1 = self.upconv1(b)
        cropped_c4 = self.crop(c4, u1)
        u1 = torch.cat([cropped_c4, u1], dim=1)
        u1 = self.up1(u1)
        
        u2 = self.upconv2(u1)
        cropped_c3 = self.crop(c3, u2)
        u2 = torch.cat([cropped_c3, u2], dim=1)
        u2 = self.up2(u2)
        
        u3 = self.upconv3(u2)
        cropped_c2 = self.crop(c2, u3)
        u3 = torch.cat([cropped_c2, u3], dim=1)
        u3 = self.up3(u3)
        
        u4 = self.upconv4(u3)
        cropped_c1 = self.crop(c1, u4)
        u4 = torch.cat([cropped_c1, u4], dim=1)
        u4 = self.up4(u4)
        
        # --- Output ---
        out = self.outconv(u4)
        # out = F.interpolate(out, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return out
