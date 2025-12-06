import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationAE(nn.Module):
    """Basic Autoencoder for segmentation"""
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder - 4 upsampling to get back to 512x512
        self.dec3 = self._upconv_block(1024, 512)  # 32 -> 64
        self.dec2 = self._upconv_block(512, 256)   # 64 -> 128
        self.dec1 = self._upconv_block(256, 128)   # 128 -> 256
        self.dec0 = self._upconv_block(128, 64)    # 256 -> 512
        
        # Output
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2, 2)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encode
        e1 = self.enc1(x)              # 512x512
        e2 = self.enc2(self.pool(e1))  # 256x256
        e3 = self.enc3(self.pool(e2))  # 128x128
        e4 = self.enc4(self.pool(e3))  # 64x64
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # 32x32
        
        # Decode
        d3 = self.dec3(b)   # 64x64
        d2 = self.dec2(d3)  # 128x128
        d1 = self.dec1(d2)  # 256x256
        d0 = self.dec0(d1)  # 512x512
        
        return self.out(d0)


class SegmentationVAE(nn.Module):
    """Variational Autoencoder for segmentation with latent space regularization"""
    def __init__(self, in_channels=3, num_classes=20, latent_dim=256):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Latent space (after 4 pooling: H/16, W/16 = 32x32 for 512x512 input)
        self.fc_mu = nn.Conv2d(512, latent_dim, 1)
        self.fc_logvar = nn.Conv2d(512, latent_dim, 1)
        
        # Decoder - 3 upsampling layers to go from 32x32 back to 512x512
        self.dec_input = nn.Conv2d(latent_dim, 512, 1)
        self.dec3 = self._upconv_block(512, 256)   # 32 -> 64
        self.dec2 = self._upconv_block(256, 128)   # 64 -> 128
        self.dec1 = self._upconv_block(128, 64)    # 128 -> 256
        self.dec0 = self._upconv_block(64, 32)     # 256 -> 512
        
        self.out = nn.Conv2d(32, num_classes, kernel_size=1)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encode
        e1 = self.enc1(x)                    # 512x512
        e2 = self.enc2(self.pool(e1))        # 256x256
        e3 = self.enc3(self.pool(e2))        # 128x128
        e4 = self.enc4(self.pool(e3))        # 64x64
        e5 = self.pool(e4)                   # 32x32
        
        # Latent space
        mu = self.fc_mu(e5)
        logvar = self.fc_logvar(e5)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        d = self.dec_input(z)                # 32x32
        d3 = self.dec3(d)                    # 64x64
        d2 = self.dec2(d3)                   # 128x128
        d1 = self.dec1(d2)                   # 256x256
        d0 = self.dec0(d1)                   # 512x512
        
        out = self.out(d0)
        
        return out, mu, logvar