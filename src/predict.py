import torch
import torch.nn.functional as F
import sys
import os
import random

# Configuration du chemin système pour permettre l'import des modules du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.stgcn import STGCN
from src.dataloader import NTUDataset
from src.utils.graph import Graph

def diagnose_patient():
    """
    Simule le diagnostic d'un patient aléatoire issu du jeu de test.
    Charge le modèle, effectue l'inférence sur un échantillon et affiche le rapport.
    """
    
    # --- 1. Configuration ---
    DATA_PATH = 'data/raw/NTU60_CS.npz'
    MODEL_PATH = 'models/stgcn_gait_prototype.pth'
    
    # Sélection du périphérique (MPS pour Apple Silicon, sinon CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # --- 2. Préparation des Données ---
    print("--- MODULE DE DIAGNOSTIC CLINIQUE (IA) ---")
    print("Initialisation et chargement des données de test...")
    
    graph = Graph()
    
    # Chargement du dataset en mode 'test' (données inconnues du modèle)
    try:
        dataset = NTUDataset(DATA_PATH, graph, mode='test')
    except FileNotFoundError:
        print(f"Erreur critique : Le fichier de données {DATA_PATH} est introuvable.")
        return

    # Sélection aléatoire d'un patient pour la simulation
    patient_idx = random.randint(0, len(dataset) - 1)
    features, true_label_idx = dataset[patient_idx]
    
    # Formatage du tenseur : ajout de la dimension batch (1, Channels, Time, Vertices)
    input_tensor = features.unsqueeze(0).to(device)
    
    # Dictionnaire de correspondance des classes
    classes = {0: "SAIN (Marche Normale)", 1: "PATHOLOGIQUE (Titubation)"}
    true_diagnosis = classes[true_label_idx.item()]

    # --- 3. Chargement du Modèle ---
    print("Chargement de l'architecture ST-GCN et des poids...")
    
    model = STGCN(
        num_class=2, 
        in_channels=3, 
        edge_importance_weighting=True, 
        graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
    )
    
    # Chargement sécurisé des poids du modèle
    if not os.path.exists(MODEL_PATH):
        print(f"Erreur : Le modèle {MODEL_PATH} est introuvable. Veuillez lancer l'entraînement (train.py) avant.")
        return

    try:
        # 'weights_only=True' est recommandé pour la sécurité sur les versions récentes de PyTorch
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    except TypeError:
        # Fallback pour compatibilité avec les anciennes versions de PyTorch
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
    model.to(device)
    model.eval() # Passage en mode évaluation (désactive Dropout et BatchNorm)

    # --- 4. Inférence (Diagnostic) ---
    print("Analyse des biomarqueurs cinématiques en cours...")
    
    with torch.no_grad(): # Désactivation du calcul de gradient pour l'inférence
        output = model(input_tensor)
        
        # Conversion des logits en probabilités via Softmax
        probabilities = F.softmax(output, dim=1)
        
        # Extraction de la classe prédite et du score de confiance
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item() * 100

    predicted_diagnosis = classes[predicted_idx]

    # --- 5. Rapport de Résultats ---
    print("\n" + "="*50)
    print(f"DOSSIER PATIENT #{patient_idx}")
    print("="*50)
    print(f"Condition Clinique Réelle : {true_diagnosis}")
    print("-" * 50)
    print(f"Diagnostic du Modèle      : {predicted_diagnosis}")
    print(f"Indice de Confiance       : {confidence:.2f}%")
    print("-" * 50)

    if predicted_idx == true_label_idx.item():
        print("CONCLUSION : DIAGNOSTIC CONFORME")
    else:
        print("CONCLUSION : DIVERGENCE (Erreur de diagnostic)")

if __name__ == '__main__':
    diagnose_patient()