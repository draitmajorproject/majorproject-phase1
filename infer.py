import torch
import config
from models.fusion_generator import DG2Generator
from utils.dataset import CompositeDataset
import torchvision.utils as vutils
import os

def run_inference(sample_idx=0):
    # 1. Initialize Model for 3-channel output
    model = DG2Generator(config.CSV_FEATURES, config.MESH_FEATURES, output_channels=3).to(config.DEVICE)
    
    if not os.path.exists("generator.pth"):
        print("Error: generator.pth not found. Please train the model first.")
        return
        
    model.load_state_dict(torch.load("generator.pth", map_location=config.DEVICE))
    model.eval()

    # 2. Load Dataset
    ds = CompositeDataset(config.DATA_ROOT, img_size=config.IMAGE_SIZE, num_v=config.MESH_FEATURES)
    if len(ds) == 0:
        print("No samples found in dataset.")
        return

    # 3. Get Sample Data
    img_chopped, img_full, mesh_data, csv_features, target_img = ds[sample_idx]
    
    # Add batch dimension and move to device
    s_chopped = img_chopped.unsqueeze(0).to(config.DEVICE)
    s_full = img_full.unsqueeze(0).to(config.DEVICE)
    m_data = mesh_data.unsqueeze(0).to(config.DEVICE)
    c_feat = csv_features.unsqueeze(0).to(config.DEVICE)

    # 4. Generate Heatmap
    with torch.no_grad():
        predicted_heatmap = model(s_chopped, s_full, m_data, c_feat)

    # 5. Save Results
    os.makedirs("results", exist_ok=True)
    
    # Comparison: Full Surface Scan (1->3ch) | Predicted Internal Damage (3ch) | Actual Internal Damage (3ch)
    s_full_rgb = s_full.cpu().repeat(1, 3, 1, 1)
    comparison = torch.cat([s_full_rgb, predicted_heatmap.cpu(), target_img.unsqueeze(0)], dim=0)
    vutils.save_image(comparison, f"results/prediction_sample_{sample_idx}.png", normalize=True)
    
    print(f"Inference complete. Heatmap saved to results/prediction_sample_{sample_idx}.png")

if __name__ == "__main__":
    # Run inference on the first sample
    run_inference(0)
