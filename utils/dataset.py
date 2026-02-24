import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import open3d as o3d
import config
import matplotlib.pyplot as plt

class CompositeDataset(Dataset):
    def __init__(self, root, img_size=256, num_v=1024):
        self.root = root
        self.img_size = img_size
        self.num_v = num_v
        
        # Identify subjects (folders starting with V-)
        self.subjects = [d for d in os.listdir(root) if d.startswith("V-") and os.path.isdir(os.path.join(root, d))]
        
        self.img_tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])
        
        # For target heatmap, we want RGB but same size
        self.target_tf = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.subjects)

    def process_mesh(self, ply_path):
        try:
            pcd = o3d.io.read_point_cloud(ply_path)
            points = np.asarray(pcd.points)
            if len(points) == 0:
                return torch.zeros((self.num_v, 3))
            
            if len(points) >= self.num_v:
                idx = np.random.choice(len(points), self.num_v, replace=False)
                points = points[idx]
            else:
                padding = np.zeros((self.num_v - len(points), 3))
                points = np.vstack((points, padding))
            
            return torch.from_numpy(points).float()
        except:
            return torch.zeros((self.num_v, 3))

    def csv_to_heatmap(self, csv_path):
        try:
            # Read CSV, skip header if it's "mm" based
            df = pd.read_csv(csv_path, index_col=0)
            data = df.values.astype(float)
            
            # Normalize to [0, 1]
            data_min, data_max = np.nanmin(data), np.nanmax(data)
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min)
            else:
                data = np.zeros_like(data)
            
            # Apply colormap (Jet/Viridis/etc.)
            colored_data = plt.get_cmap('jet')(data)[:, :, :3] # Take RGB, ignore Alpha
            colored_data = (colored_data * 255).astype(np.uint8)
            
            return self.target_tf(colored_data)
        except Exception as e:
            # Fallback to zeros if CSV parsing fails
            return torch.zeros((3, self.img_size, self.img_size))

    def __getitem__(self, idx):
        subj = self.subjects[idx]
        subj_path = os.path.join(self.root, subj)
        
        # 1. External Surface Images
        pngs = [f for f in os.listdir(subj_path) if f.endswith(".png")]
        surface_chopped_path = None
        surface_full_path = None
        
        if len(pngs) >= 2:
            chopped_candidates = [f for f in pngs if f.startswith("V-") and subj in f]
            if not chopped_candidates:
                 chopped_candidates = [f for f in pngs if f.startswith("V-")]
            
            if chopped_candidates:
                surface_chopped_name = chopped_candidates[0]
                surface_chopped_path = os.path.join(subj_path, surface_chopped_name)
                other_pngs = [f for f in pngs if f != surface_chopped_name]
                surface_full_path = os.path.join(subj_path, other_pngs[0]) if other_pngs else None
            else:
                surface_chopped_path = os.path.join(subj_path, pngs[0])
                surface_full_path = os.path.join(subj_path, pngs[1])
        elif len(pngs) == 1:
            surface_chopped_path = os.path.join(subj_path, pngs[0])
            surface_full_path = surface_chopped_path
        
        img_chopped = self.img_tf(Image.open(surface_chopped_path).convert("L")) if surface_chopped_path else torch.zeros((1, self.img_size, self.img_size))
        img_full = self.img_tf(Image.open(surface_full_path).convert("L")) if surface_full_path else torch.zeros((1, self.img_size, self.img_size))

        # 2. Mesh data (.ply)
        ply_files = [f for f in os.listdir(subj_path) if f.endswith(".ply")]
        mesh_data = self.process_mesh(os.path.join(subj_path, ply_files[0])) if ply_files else torch.zeros((self.num_v, 3))

        # 3. Target: Color Heatmap generated from CSV
        target_img = torch.zeros((3, self.img_size, self.img_size))
        excel_dir = os.path.join(self.root, "EXCEL DATA 140kHz")
        if os.path.exists(excel_dir):
            # Match CSV to subject ID
            clean_subj = subj.replace("-","")
            csv_files = [f for f in os.listdir(excel_dir) if clean_subj in f.replace("-","") and f.endswith(".csv")]
            if csv_files:
                target_img = self.csv_to_heatmap(os.path.join(excel_dir, csv_files[0]))

        # 4. CSV Features (optional/auxiliary)
        csv_features = torch.zeros(config.CSV_FEATURES)
        # We can reuse the target CSV or just take the first row as features
        # For now, let's keep it consistent
        return img_chopped, img_full, mesh_data, csv_features, target_img
