import torch
import config
from utils.dataset import CompositeDataset
from models.fusion_generator import DG2Generator
from models.discriminator import Discriminator

def test():
    print("Initializing Dataset...")
    ds = CompositeDataset(config.DATA_ROOT, img_size=config.IMAGE_SIZE, num_v=config.MESH_FEATURES)
    print(f"Dataset length: {len(ds)}")
    
    if len(ds) == 0:
        print("No samples found. Check DATA_ROOT.")
        return

    img_chopped, img_full, mesh_data, csv_features, target_img = ds[0]
    print(f"Chopped Image Shape: {img_chopped.shape}")
    print(f"Full Image Shape: {img_full.shape}")
    print(f"Mesh Data Shape: {mesh_data.shape}")
    print(f"CSV Features Shape: {csv_features.shape}")
    print(f"Target Heatmap (Color) Shape: {target_img.shape}")

    print("\nInitializing Models...")
    # Update G for 3-channel output
    G = DG2Generator(config.CSV_FEATURES, config.MESH_FEATURES, output_channels=3)
    # Update D for 5-channel input (2 condition + 3 target)
    D = Discriminator(condition_channels=1, target_channels=3)

    # Add batch dimension
    s_chopped = img_chopped.unsqueeze(0)
    s_full = img_full.unsqueeze(0)
    m_data = mesh_data.unsqueeze(0)
    c_feat = csv_features.unsqueeze(0)
    t_img = target_img.unsqueeze(0)

    print("Running Generator...")
    fake = G(s_chopped, s_full, m_data, c_feat)
    print(f"Fake Image Shape (RGB): {fake.shape}")

    print("Running Discriminator...")
    pred = D(s_chopped, s_full, fake)
    print(f"Discriminator Prediction Shape: {pred.shape}")

    print("\nVerification Successful!")

if __name__ == "__main__":
    test()
