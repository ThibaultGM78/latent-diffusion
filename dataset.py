import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class SimpleHandsDataset(Dataset):
    def __init__(self, root_dir, size=256):
        self.root_dir = root_dir
        self.size = size
        # On liste toutes les images du dossier
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB").resize((self.size, self.size))
        
        # Normalisation entre -1 et 1 (standard pour la diffusion)
        img = (np.array(img) / 127.5 - 1.0).astype(np.float32)
        return torch.from_numpy(img).permute(2, 0, 1) # Format [C, H, W]