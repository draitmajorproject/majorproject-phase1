import torch.nn as nn

class FeatureEncoder(nn.Module):
    def __init__(self,csv_dim,mesh_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(csv_dim+mesh_dim,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,1024)
        )

    def forward(self,csv,mesh):
        x = nn.functional.concat([csv,mesh],1)
        return self.net(x)
