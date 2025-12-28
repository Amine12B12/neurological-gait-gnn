import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# Configuration du chemin système pour permettre l'import des modules du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.models.stgcn import STGCN
from src.dataloader import NTUDataset
from src.utils.graph import Graph

def train():
    """
    Exécute la pipeline d'entraînement complète du modèle ST-GCN.
    
    Étapes :
    1. Configuration de l'environnement (Device, Hyperparamètres).
    2. Chargement et préparation des données.
    3. Initialisation du modèle et de l'optimiseur.
    4. Boucle d'apprentissage (Forward/Backward pass).
    5. Sauvegarde des poids du modèle.
    """
    
    # --- 1. Configuration et Hyperparamètres ---
    BATCH_SIZE = 16        # Taille du lot (réduite pour compatibilité mémoire MPS/CPU)
    LEARNING_RATE = 0.01   # Taux d'apprentissage initial
    EPOCHS = 5             # Nombre d'itérations complètes sur le dataset
    DATA_PATH = 'data/raw/NTU60_CS.npz'
    
    # Sélection du périphérique de calcul (MPS pour Apple Silicon, sinon CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Configuration : Accélération matérielle Apple Metal (MPS) activée.")
    else:
        device = torch.device("cpu")
        print("Configuration : Entraînement sur CPU (Vitesse réduite).")

    # --- 2. Chargement des Données ---
    print("\n[Phase 1/4] Chargement du graphe et des données...")
    graph = Graph()
    
    # Instanciation du Dataset complet
    full_dataset = NTUDataset(DATA_PATH, graph)
    
    # Séparation Entraînement (80%) / Validation (20%)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    # Création du DataLoader pour l'itération par batchs mélangés
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset prêt : {len(train_dataset)} séquences allouées à l'entraînement.")

    # --- 3. Initialisation du Modèle ---
    print("[Phase 2/4] Initialisation de l'architecture ST-GCN...")
    model = STGCN(
        num_class=2, 
        in_channels=3, 
        edge_importance_weighting=True, 
        graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'}
    )
    model.to(device)

    # Définition de la fonction de coût (Cross Entropy) et de l'optimiseur (Adam)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. Boucle d'Entraînement ---
    print(f"\n[Phase 3/4] Démarrage de l'entraînement sur {EPOCHS} époques...")
    model.train() # Activation du mode entraînement (Dropout, BatchNorm, etc.)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\n--- Époque {epoch + 1}/{EPOCHS} ---")
        
        for i, (inputs, labels) in enumerate(train_loader):
            # Transfert des tenseurs vers le périphérique (GPU/MPS ou CPU)
            inputs, labels = inputs.to(device), labels.to(device)

            # Réinitialisation des gradients pour la nouvelle itération
            optimizer.zero_grad()

            # A. Propagation avant (Forward pass)
            outputs = model(inputs)
            
            # B. Calcul de la perte (Loss)
            loss = criterion(outputs, labels)
            
            # C. Rétropropagation (Backward pass) et mise à jour des poids
            loss.backward()
            optimizer.step()

            # Suivi des métriques de performance
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Affichage du journal tous les 10 batchs
            if i % 10 == 0:
                current_acc = 100 * correct / total
                print(f"   Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f} | Accuracy: {current_acc:.2f}%")

        # Bilan de fin d'époque
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Fin Époque {epoch + 1} | Perte Moyenne: {epoch_loss:.4f} | Précision Globale: {epoch_acc:.2f}%")

    # --- 5. Sauvegarde ---
    print("\n[Phase 4/4] Sauvegarde du modèle entraîné...")
    save_dir = 'models'
    
    # Création du dossier si nécessaire
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, 'stgcn_gait_prototype.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Modèle sauvegardé avec succès dans : {save_path}")

if __name__ == '__main__':
    train()