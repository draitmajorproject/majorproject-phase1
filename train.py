import torch
from torch.utils.data import DataLoader
from utils.dataset import CompositeDataset
from models.fusion_generator import DG2Generator
from models.discriminator import Discriminator
import config
from utils.losses import *

ds = CompositeDataset("data")
dl = DataLoader(ds,batch_size=config.BATCH_SIZE,shuffle=True)

G = DG2Generator(config.CSV_FEATURES,config.MESH_FEATURES).to(config.DEVICE)
D = Discriminator().to(config.DEVICE)

optG = torch.optim.Adam(G.parameters(),lr=config.LR,betas=(0.5,0.999))
optD = torch.optim.Adam(D.parameters(),lr=chonfig.LR,betas=(0.5,0.999))

for epoch in range(config.EPOCHS):
    for a,c,csv,mesh,label in dl:
        a,c,csv,mesh,label = a.to(config.DEVICE),c.to(config.DEVICE),csv.to(config.DEVICE),mesh.to(config.DEVICE),label.to(config.DEVICE)

        fake = G(a,c,csv,mesh)

        # Train D
        real_pred = D(label)
        fake_pred = D(fake.detach())
        lossD = discriminator_loss(real_pred,fake_pred)

        optD.zero_grad()
        lossD.backward()
        optD.step()

        # Train G
        fake_pred = D(fake)
        lossG = generator_loss(fake_pred,fake,label)

        optG.zero_grad()
        lossG.backward()
        optG.step()

    print(f"Epoch {epoch} | G {lossG.item():.3f} | D {lossD.item():.3f}")
