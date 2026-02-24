import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class DG2Generator(nn.Module):
    def __init__(self, csv_dim, mesh_dim, img_channels=1, output_channels=3):
        super(DG2Generator, self).__init__()
        
        # 1. Surface Image Encoder (2 input channels: chopped + full)
        def conv_block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.e1 = conv_block(img_channels * 2, 64, normalize=False)
        self.e2 = conv_block(64, 128)
        self.e3 = conv_block(128, 256)
        self.e4 = conv_block(256, 512)
        self.e5 = conv_block(512, 512)
        
        # 2. CSV and Mesh Encoders
        self.csv_enc = FeatureEncoder(csv_dim, 256)
        self.mesh_enc = nn.Sequential(
            nn.Linear(mesh_dim * 3, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256)
        )
        
        # Bottleneck Fusion
        self.fusion = nn.Sequential(
            nn.Linear(32768 + 256 + 256, 512),
            nn.ReLU(inplace=True)
        )

        # 3. Decoder
        def up_block(in_c, out_c, dropout=0.0):
            layers = [
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.InstanceNorm2d(out_c),
                nn.ReLU(inplace=True)
            ]
            if dropout:
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)

        self.d1 = up_block(512, 512, dropout=0.5)
        self.d2 = up_block(1024, 256)
        self.d3 = up_block(512, 128)
        self.d4 = up_block(256, 64)
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, output_channels, 4, 2, 1),
            nn.Tanh() # Output in [-1, 1] range
        )

    def forward(self, img_chopped, img_full, mesh, csv):
        img = torch.cat([img_chopped, img_full], dim=1)
        
        x1 = self.e1(img)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        x5 = self.e5(x4)
        
        c_feat = self.csv_enc(csv)
        m_feat = self.mesh_enc(mesh.view(mesh.size(0), -1))
        
        b, c, h, w = x5.size()
        feat_combined = torch.cat([x5.reshape(b, -1), c_feat, m_feat], dim=1)
        fused = self.fusion(feat_combined).view(b, 512, 1, 1)
        fused = F.interpolate(fused, size=(h, w))
        
        u1 = self.d1(fused)
        u2 = self.d2(torch.cat([u1, x4], dim=1))
        u3 = self.d3(torch.cat([u2, x3], dim=1))
        u4 = self.d4(torch.cat([u3, x2], dim=1))
        
        out = self.final(torch.cat([u4, x1], dim=1))
        return out
