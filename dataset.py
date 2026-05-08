import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class SimpleHandsDataset(Dataset):
    def __init__(self, root_dir, size=256, txt_prompts=False):
        self.root_dir = root_dir
        self.size = size
        self.txt_prompts = txt_prompts
        # On liste toutes les images du dossier
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 1. Traitement de l'image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert("RGB").resize((self.size, self.size))
        
        # Normalisation entre -1 et 1
        img = (np.array(img) / 127.5 - 1.0).astype(np.float32)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1) # Format [C, H, W]

        # 2. Traitement du texte (Conditionnement)
        if self.txt_prompts:
            # Récupère le nom de fichier sans l'extension (ex: "main_01.png" -> "main_01")
            base_name = os.path.splitext(img_name)[0]
            txt_path = os.path.join(self.root_dir, base_name + ".txt")
            
            # Vérifie si le fichier texte existe
            if os.path.exists(txt_path):
                with open(txt_path, "r", encoding="utf-8") as f:
                    prompt = f.read().strip()
            else:
                # Fallback générique si le fichier texte manque
                prompt = "a close up photo of a hand" 
                
            return img_tensor, prompt

        # Si txt_prompts est False, on retourne juste l'image (rétrocompatibilité)
        return img_tensor