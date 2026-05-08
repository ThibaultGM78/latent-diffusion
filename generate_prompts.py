import pandas as pd
import os
from tqdm import tqdm

# --- CONFIGURATION ---
# Remplace par le chemin exact vers ton fichier CSV
csv_path = "./data/hands_metadata.csv" 

# Le dossier où se trouvent tes images (et où les .txt seront sauvegardés)
output_dir = "./data/hands" 

def generate_text_prompts():
    print("Lecture du fichier CSV...")
    # Chargement du CSV
    df = pd.read_csv(csv_path)
    
    print(f"Génération des fichiers .txt pour {len(df)} images...")
    
    # Parcours de chaque ligne du CSV
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # Extraction des valeurs (conversion en chaîne de caractères et nettoyage des espaces)
        age = str(row['age']).strip()
        gender = str(row['gender']).strip()
        skin_color = str(row['skinColor']).strip()
        accessories_val = str(row['accessories']).strip()
        image_name = str(row['imageName']).strip()
        
        # Logique conditionnelle pour les accessoires
        if accessories_val == '1':
            acc_prompt = "with accessories"
        else:
            acc_prompt = "without accessories"
            
        # Concaténation pour créer le prompt final
        # Ex: "25 male pale with accessories" ou "adult female dark without accessories"
        prompt = f"{age} {gender} {skin_color} {acc_prompt}"
        
        # Nettoyage des espaces multiples (au cas où une colonne serait vide)
        prompt = " ".join(prompt.split())
        
        # Création du nom de fichier texte correspondant à l'image
        # Si imageName est "hand_001.jpg", base_name devient "hand_001"
        base_name = os.path.splitext(image_name)[0]
        txt_filename = f"{base_name}.txt"
        txt_filepath = os.path.join(output_dir, txt_filename)
        
        # Écriture du prompt dans le fichier .txt
        with open(txt_filepath, "w", encoding="utf-8") as f:
            f.write(prompt)

    print("Terminé ! Tous les fichiers .txt ont été générés avec succès.")

if __name__ == "__main__":
    # Crée le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    generate_text_prompts()