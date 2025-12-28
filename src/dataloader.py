import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset

# Ajout du chemin racine pour permettre l'exécution directe du script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.utils.graph import Graph

class NTUDataset(Dataset):
    """
    Dataset PyTorch pour le chargement et le prétraitement des données squelettiques NTU RGB+D.
    
    Ce dataset est spécialisé pour la classification binaire :
    - Classe 0 : Marche normale (Sain)
    - Classe 1 : Marche pathologique/Titubation (Pathologique)
    
    Format des données :
    - Entrée brute : (N, T, 150) où 150 = 2 personnes * 25 joints * 3 coordonnées.
    - Sortie modèle : (N, C, T, V) soit (N, 3, 300, 25).
    """

    # Constantes du Dataset NTU
    NUM_JOINTS = 25
    NUM_PERSONS = 2
    NUM_CHANNELS = 3  # X, Y, Z
    
    # Indices des classes dans le dataset original
    CLASS_WALKING = 0    # A1
    CLASS_STAGGERING = 7 # A8

    def __init__(self, file_path, graph, mode='train'):
        """
        Initialise le dataset.

        Args:
            file_path (str): Chemin vers le fichier .npz compressé.
            graph (Graph): Objet graphe définissant la topologie du squelette.
            mode (str): 'train' pour l'entraînement, 'test' pour l'évaluation.
        """
        print(f"Initialisation du dataset ({mode}) depuis : {file_path}")
        
        # 1. Chargement du fichier
        if not os.path.exists(file_path):
            print(f"Erreur critique : Le fichier {file_path} est introuvable.")
            sys.exit(1)
            
        try:
            raw_data = np.load(file_path)
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier .npz : {e}")
            sys.exit(1)
        
        # Sélection des clés selon le mode
        if mode == 'train':
            y_one_hot = raw_data['y_train']
            x_flattened = raw_data['x_train']
        else:
            y_one_hot = raw_data['y_test']
            x_flattened = raw_data['x_test']
            
        # Conversion des labels One-Hot en entiers
        self.y_all = np.argmax(y_one_hot, axis=1)
        
        # 2. Reshape et Permutation des Dimensions
        # Dimension brute : (N, T, Features) -> (N, 300, 150)
        N, T, _ = x_flattened.shape
        
        # Restructuration spatiale : (N, T, M, V, C)
        # On suppose l'ordre standard : Personne -> Joint -> Coordonnée
        x_reshaped = x_flattened.reshape(N, T, self.NUM_PERSONS, self.NUM_JOINTS, self.NUM_CHANNELS)
        
        # Sélection de la première personne uniquement (Indice 0 sur la dim M)
        # Nouvelle shape : (N, T, V, C)
        x_person_1 = x_reshaped[:, :, 0, :, :]
        
        # Permutation pour le format attendu par ST-GCN (PyTorch)
        # (N, T, V, C) -> (N, C, T, V)
        self.x_all = x_person_1.transpose(0, 3, 1, 2)
        
        # 3. Filtrage des Classes (Binary Classification)
        # On ne garde que les indices correspondant à "Walking" (0) et "Staggering" (7)
        idx_sain = np.where(self.y_all == self.CLASS_WALKING)[0]
        idx_patho = np.where(self.y_all == self.CLASS_STAGGERING)[0]
        
        self.indices = np.concatenate((idx_sain, idx_patho))
        
        # Création des labels binaires normalisés pour l'entraînement
        # 0 = Sain, 1 = Pathologique
        self.labels = np.concatenate((
            np.zeros(len(idx_sain)), 
            np.ones(len(idx_patho))
        ))
        
        print(f"Dataset chargé avec succès.")
        print(f"   - Total échantillons : {len(self.indices)}")
        print(f"   - Sains (Walking)    : {len(idx_sain)}")
        print(f"   - Patho (Staggering) : {len(idx_patho)}")
        print(f"   - Dimensions Tenseur : {self.x_all.shape[1:]} (C, T, V)")

    def __len__(self):
        """Retourne la taille totale du dataset filtré."""
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Récupère un échantillon par son index.

        Returns:
            x (torch.Tensor): Séquence squelettique (C, T, V).
            y (torch.Tensor): Label (0 ou 1).
        """
        # Mapping de l'index virtuel vers l'index réel dans le tableau global
        real_idx = self.indices[idx]
        
        # Conversion en tenseurs PyTorch (Float32 pour les données, Long pour les labels)
        x = torch.tensor(self.x_all[real_idx], dtype=torch.float)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return x, y

if __name__ == '__main__':
    # Bloc de test unitaire
    print("--- Test unitaire du DataLoader ---")
    
    # Tentative de localisation automatique du fichier de données
    possible_paths = [
        'data/raw/NTU60_CS.npz',
        os.path.join(os.path.dirname(__file__), '../../data/raw/NTU60_CS.npz')
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
            
    if data_path:
        # Initialisation du graphe et chargement
        g = Graph()
        dataset = NTUDataset(data_path, g)
        
        # Vérification du premier élément
        x, y = dataset[0]
        print(f"Test réussi :")
        print(f"   - Input Shape : {x.shape}")
        print(f"   - Label Value : {y}")
    else:
        print("Test annulé : Fichier de données introuvable dans les chemins par défaut.")