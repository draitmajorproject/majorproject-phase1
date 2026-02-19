import torch.nn as nn

bce = nn.BCELoss()
l1 = nn.L1Loss()

def generator_loss(fake_pred, fake_img, real_img):
    adv = bce(fake_pred, torch.ones_like(fake_pred))
    recon = l1(fake_img, real_img)
    return adv + 100*recon

def discriminator_loss(real_pred,fake_pred):
    real = bce(real_pred, torch.ones_like(real_pred))
    fake = bce(fake_pred, torch.zeros_like(fake_pred))
    return (real+fake)/2
