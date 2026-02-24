import torch
from torch.utils.data import DataLoader
from utils.dataset import CompositeDataset
from models.fusion_generator import DG2Generator
from models.discriminator import Discriminator
import config
from utils.losses import *

import torchvision.utils as vutils
import os

os.makedirs("outputs", exist_ok=True)

ds = CompositeDataset(config.DATA_ROOT, img_size=config.IMAGE_SIZE, num_v=config.MESH_FEATURES)
dl = DataLoader(ds,batch_size=config.BATCH_SIZE,shuffle=True)

G = DG2Generator(config.CSV_FEATURES, config.MESH_FEATURES, output_channels=3).to(config.DEVICE)
D = Discriminator(condition_channels=1, target_channels=3).to(config.DEVICE)

optG = torch.optim.Adam(G.parameters(),lr=config.LR,betas=(0.5,0.999))
optD = torch.optim.Adam(D.parameters(),lr=config.LR,betas=(0.5,0.999))

for epoch in range(config.EPOCHS):
    for i, (img_chopped, img_full, mesh_data, csv_features, target_img) in enumerate(dl):
        img_chopped = img_chopped.to(config.DEVICE)
        img_full = img_full.to(config.DEVICE)
        mesh_data = mesh_data.to(config.DEVICE)
        csv_features = csv_features.to(config.DEVICE)
        target_img = target_img.to(config.DEVICE)

        fake_img = G(img_chopped, img_full, mesh_data, csv_features)

        # Train D
        real_pred = D(img_chopped, img_full, target_img)
        fake_pred = D(img_chopped, img_full, fake_img.detach())
        lossD = discriminator_loss(real_pred, fake_pred)

        optD.zero_grad()
        lossD.backward()
        optD.step()

        # Train G
        fake_pred = D(img_chopped, img_full, fake_img)
        lossG = generator_loss(fake_pred, fake_img, target_img)

        optG.zero_grad()
        lossG.backward()
        optG.step()

    print(f"Epoch {epoch} | G {lossG.item():.3f} | D {lossD.item():.3f}")

    if epoch % 10 == 0:
        # Save checkpoints
        torch.save(G.state_dict(), "generator.pth")
        torch.save(D.state_dict(), "discriminator.pth")
        
        # Save sample images
        with torch.no_grad():
            sample_fake = G(img_chopped, img_full, mesh_data, csv_features)
            # Create a comparison grid: Full Surface (1->3ch) | Target Internal (3ch) | Predicted Internal (3ch)
            s_full_rgb = img_full[:1].repeat(1, 3, 1, 1)
            grid = torch.cat([s_full_rgb, target_img[:1], sample_fake[:1]], dim=0)
            vutils.save_image(grid, f"outputs/epoch_{epoch}.png", normalize=True)
