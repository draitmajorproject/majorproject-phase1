import os, torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class CompositeDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.samples = os.listdir(root)
        self.tf = T.Compose([
            T.Resize((256,256)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.samples[idx])

        a = self.tf(Image.open(os.path.join(path,"a_scan.png")).convert("L"))
        c = self.tf(Image.open(os.path.join(path,"c_scan.png")).convert("L"))
        label = self.tf(Image.open(os.path.join(path,"label.png")).convert("L"))

        csv = torch.tensor(pd.read_csv(os.path.join(path,"features.csv")).values.flatten()).float()
        mesh = torch.tensor(np.load(os.path.join(path,"mesh.npy"))).float()

        return a, c, csv, mesh, label
