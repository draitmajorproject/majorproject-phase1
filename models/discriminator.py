import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, condition_channels=1, target_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Inputs: img_chopped (1) + img_full (1) + heatmap (3) = 5 channels
        self.model = nn.Sequential(
            *discriminator_block(condition_channels * 2 + target_channels, 64, normalization=False), 
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_chopped, img_full, target_img):
        # Concatenate: [1ch, 1ch, 3ch] -> 5ch
        img_input = torch.cat((img_chopped, img_full, target_img), 1)
        return self.model(img_input)
