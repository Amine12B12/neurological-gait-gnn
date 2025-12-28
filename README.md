# Neurological Gait Analysis with ST-GCN

Un système d'intelligence artificielle basé sur les **Graphes Spatio-Temporels (ST-GCN)** pour le diagnostic automatisé des pathologies de la marche (AVC, Parkinson, etc.) à partir de données squelettiques.

![Demo Squelette](lien_vers_ton_gif_ou_image.gif) *<-- Mets une capture d'écran ici*

## 🎯 Objectif
Ce projet répond à une problématique clinique : comment quantifier objectivement les troubles de la marche sans équipement lourd ?
En modélisant le corps humain sous forme de graphe et en analysant les connexions fonctionnelles (ex: symétrie gauche/droite), ce modèle atteint une précision de **98%** sur la distinction Marche Saine vs Pathologique.

## 🛠️ Architecture Technique
* **Modèle :** ST-GCN (Spatial Temporal Graph Convolutional Network).
* **Données :** NTU RGB+D (Squelettes 3D).
* **Input :** Tenseur (C, T, V) = (3 Coordonnées, 300 Frames, 25 Articulations).
* **Framework :** PyTorch & PyTorch Geometric.

## 🚀 Installation

```bash
# Cloner le projet
git clone [https://github.com/ton-pseudo/neurological-gait-gnn.git](https://github.com/ton-pseudo/neurological-gait-gnn.git)
cd neurological-gait-gnn

# Créer l'environnement
python -m venv venv
source venv/bin/activate  # Sur Mac/Linux

# Installer les dépendances
pip install -r requirements.txt