import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import random

# Configuration du chemin d'accès pour permettre les imports depuis le dossier parent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.dataloader import NTUDataset
from src.utils.graph import Graph

def visualize_skeleton(dataset, index=None):
    """
    Génère et affiche une animation 3D d'une séquence squelettique.

    Args:
        dataset (NTUDataset): Le jeu de données contenant les séquences.
        index (int, optional): L'index de l'échantillon à visualiser. 
                               Si None, un index aléatoire est choisi.
    """
    # Sélection aléatoire d'un échantillon si aucun index n'est fourni
    if index is None:
        index = random.randint(0, len(dataset) - 1)
    
    # Extraction des données et du label
    # data shape : (3, 300, 25) -> (Channels, Time, Vertices)
    data, label = dataset[index]
    
    # Conversion du tenseur PyTorch en tableau NumPy pour l'affichage
    skeleton = data.numpy()
    
    # Interprétation du label pour l'affichage
    label_text = "SAIN (Marche)" if label == 0 else "PATHOLOGIQUE (Titubation)"
    print(f"Génération de l'animation pour le patient #{index}")
    print(f"Diagnostic réel : {label_text}")

    # Initialisation de la figure Matplotlib en 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Définition de la structure squelettique (paires d'indices à relier)
    # Basé sur la topologie du dataset NTU RGB+D (25 articulations)
    bones = [
        (0, 1), (1, 20), (20, 2), (2, 3),    # Tronc
        (20, 4), (4, 5), (5, 6), (6, 7),     # Bras Droit
        (7, 22), (22, 21),                   # Main Droite
        (20, 8), (8, 9), (9, 10), (10, 11),  # Bras Gauche
        (11, 24), (24, 23),                  # Main Gauche
        (0, 12), (12, 13), (13, 14), (14, 15), # Jambe Droite
        (0, 16), (16, 17), (17, 18), (18, 19), # Jambe Gauche
        (2, 20)                              # Cou
    ]

    # Calcul des limites globales des axes pour stabiliser la caméra durant l'animation
    x_min, x_max = skeleton[0].min(), skeleton[0].max()
    y_min, y_max = skeleton[1].min(), skeleton[1].max()
    z_min, z_max = skeleton[2].min(), skeleton[2].max()

    def update(frame):
        """Fonction de mise à jour appelée à chaque frame de l'animation."""
        ax.clear()
        
        # Titre dynamique indiquant le numéro de frame
        ax.set_title(f"Patient #{index} | {label_text} | Frame {frame}", fontsize=15)
        
        # Fixation des limites de la caméra
        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_ylim(y_min - 0.5, y_max + 0.5)
        ax.set_zlim(z_min - 0.5, z_max + 0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Récupération des coordonnées pour la frame courante
        xs = skeleton[0, frame, :]
        ys = skeleton[1, frame, :]
        zs = skeleton[2, frame, :]

        # 1. Affichage des articulations (Noeuds)
        ax.scatter(xs, ys, zs, c='r', marker='o')

        # 2. Affichage des os (Arêtes)
        for connection in bones:
            if connection[0] < 25 and connection[1] < 25:
                ax.plot(
                    [xs[connection[0]], xs[connection[1]]],
                    [ys[connection[0]], ys[connection[1]]],
                    [zs[connection[0]], zs[connection[1]]],
                    c='b'
                )

    # Création de l'animation (intervalle de 50ms ~= 20 fps)
    ani = animation.FuncAnimation(fig, update, frames=300, interval=50)
    
    print("Lancement de la fenêtre de visualisation...")
    plt.show()

if __name__ == "__main__":
    # Chargement du graphe et du dataset en mode test
    graph = Graph()
    dataset_path = 'data/raw/NTU60_CS.npz'
    
    # Vérification sommaire de l'existence du fichier avant chargement
    if os.path.exists(dataset_path):
        dataset = NTUDataset(dataset_path, graph, mode='test')
        visualize_skeleton(dataset)
    else:
        print(f"Erreur : Le fichier de données {dataset_path} est introuvable.")