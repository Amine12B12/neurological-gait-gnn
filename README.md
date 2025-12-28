# 🧠 Analyse Neurologique de la Marche par Graphes Spatio-Temporels (ST-GCN)

> **Diagnostic automatisé des pathologies motrices (AVC, Parkinson) via l'analyse cinématique du squelette.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Présentation
Ce projet implémente une intelligence artificielle basée sur les **réseaux de neurones graphiques (ST-GCN)** pour distinguer une marche saine d'une marche pathologique. 

Contrairement aux approches classiques (vidéo RGB), ce modèle travaille sur les **coordonnées squelettiques**, garantissant :
1. **L'anonymat des patients** (Pas de visage, respect RGPD).
2. **La robustesse** (L'IA analyse le mouvement pur, pas les vêtements).
3. **La précision** (Modélisation des interactions fonctionnelles entre les membres).

## [Aperçu Squelette] (https://private-user-images.githubusercontent.com/215486463/530570013-bdc10532-90c7-4017-a333-07475880c10b.gif?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NjY5NDg1NzUsIm5iZiI6MTc2Njk0ODI3NSwicGF0aCI6Ii8yMTU0ODY0NjMvNTMwNTcwMDEzLWJkYzEwNTMyLTkwYzctNDAxNy1hMzMzLTA3NDc1ODgwYzEwYi5naWY_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUxMjI4JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MTIyOFQxODU3NTVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT04OGVmYjlmZDM3ZTA5OTFiMzAwZWIwMWI1NDg1MjFlNjAwMmYzMDE3MTI5NTE0OTA4NjU3OWViYzIxZWNiZjkyJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.p1W6Y3ieHT6xP-TNHE0AlIWdg4A971ZBL9EEG8erH0s)

## 🎯 Objectifs Scientifiques
* **Modéliser** le corps humain sous forme de graphe $(V, E)$.
* **Détecter** les asymétries fonctionnelles (ex: désynchronisation Gauche/Droite caractéristique de l'hémiplégie).
* **Diagnostiquer** en temps réel avec une puissance de calcul modérée.

---

## 🛠️ Architecture Technique

### Le Modèle : ST-GCN
Nous utilisons un *Spatial Temporal Graph Convolutional Network* qui combine :
* **Convolutions Spatiales (GCN) :** Pour comprendre la posture à un instant $t$.
* **Convolutions Temporelles (TCN) :** Pour comprendre la dynamique sur 300 frames.

### Les Données
* **Source :** NTU RGB+D 60 Dataset.
* **Format d'entrée :** Tenseur de dimension $(N, C, T, V)$.
  * $N$ : Batch size
  * $C$ : 3 (Coordonnées x, y, z)
  * $T$ : 300 (Frames temporelles)
  * $V$ : 25 (Articulations/Noeuds du graphe)

---

## 📊 Résultats
Sur un jeu de test parfaitement équilibré (jamais vu par le modèle durant l'entraînement) :

| Métrique | Valeur |
| :--- | :--- |
| **Précision (Accuracy)** | **98.78%** |
| **Perte (Loss)** | 0.038 |
| **Vitesse d'inférence** | < 50ms / patient |

---

## 🚀 Installation

### 1. Cloner le projet
```bash
git clone [https://github.com/Amine12B12/neurological-gait-gnn.git](https://github.com/votre-pseudo/neurological-gait-gnn.git)
cd neurological-gait-gnn

```

### 2. Créer l'environnement virtuel

```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt

```

### 4. Configuration des Données

Le dataset étant volumineux, il n'est pas inclus dans le dépôt Git.

1. Téléchargez le fichier `NTU60_CS.npz`.
2. Placez-le dans le dossier : `data/raw/NTU60_CS.npz`.

---

## 🧠 Utilisation

### 1. Entraîner le Modèle (`train.py`)

Lance l'apprentissage sur les données brutes. Le script gère le chargement, la création du graphe, et la rétropropagation.

```bash
python src/train.py

```

> *Le modèle entraîné sera sauvegardé dans `models/stgcn_gait_prototype.pth`.*

### 2. Diagnostic Unitaire (`predict.py`)

Simule l'arrivée d'un nouveau patient. Le script pioche un échantillon inconnu dans le test-set et établit un diagnostic avec un score de confiance.

```bash
python src/predict.py

```

**Exemple de sortie :**

```text
📄 DOSSIER PATIENT #224
Véritable condition : PATHOLOGIQUE (Titubation)
🤖 AVIS DE L'IA     : PATHOLOGIQUE
📊 Confiance        : 99.12%

```

### 3. Visualisation 3D (`visualize.py`)

Reconstruit le squelette en 3D et l'anime pour valider visuellement la pathologie détectée par l'IA.

```bash
python src/visualize.py

```

---

## 📂 Structure du Projet

```text
neurological-gait-gnn/
├── data/                  # Dossier des données (ignoré par Git)
├── models/                # Poids du modèle entraîné (.pth)
├── src/
│   ├── models/            # Architecture ST-GCN (Couches Neurales)
│   ├── utils/             # Définition du Graphe (Noeuds & Arêtes)
│   ├── dataloader.py      # Pipeline de chargement et preprocessing
│   ├── train.py           # Boucle d'entraînement
│   ├── predict.py         # Script d'inférence clinique
│   └── visualize.py       # Moteur de rendu 3D (Matplotlib)
├── requirements.txt       # Liste des librairies
└── README.md              # Documentation

```

## ⚖️ Crédits

* **Auteur :** Amine Benyoucef
* **Dataset :** NTU RGB+D 60 (Rose Lab).
* **Papier de référence :** *Yan et al., "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition", AAAI 2018.*

```

```
