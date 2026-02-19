import torch
from models.fusion_generator import DG2Generator
import config

model = DG2Generator(config.CSV_FEATURES,config.MESH_FEATURES)
model.load_state_dict(torch.load("generator.pth"))
model.eval()

# load sample tensors same as dataset loader
heatmap = model(a,c,csv,mesh)
