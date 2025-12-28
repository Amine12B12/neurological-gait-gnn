import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Configuration du chemin système pour permettre l'import du module utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.graph import Graph

class GraphConv(nn.Module):
    """
    Couche de Convolution Graphique Spatiale.
    
    Cette couche applique une convolution sur les nœuds du graphe en utilisant
    la matrice d'adjacence pour propager l'information entre les voisins.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        
        # Convolution 1x1 pour transformer les caractéristiques avant la propagation
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels * kernel_size, 
            kernel_size=(1, 1)
        )

    def forward(self, x, A):
        """
        Args:
            x (torch.Tensor): Données d'entrée (N, C, T, V).
            A (torch.Tensor): Matrice d'adjacence (K, V, V).
        
        Returns:
            torch.Tensor: Données convolées (N, C_out, T, V).
        """
        # 1. Transformation des features (N, C, T, V) -> (N, C*K, T, V)
        x = self.conv(x)
        
        # 2. Reshape pour séparer la dimension du noyau K
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        
        # 3. Application de la matrice d'adjacence via Einstein Summation
        # 'nkctv' : Input (N, Kernel, Channels, Time, Vertices)
        # 'kvw'   : Adjacency (Kernel, Vertices, Neighbor_Vertices)
        # -> 'nctw' : Output (N, Channels, Time, Vertices)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        
        return x.contiguous()

class STGCNBlock(nn.Module):
    """
    Bloc fondamental du ST-GCN.
    Combinaison séquentielle de :
    1. Convolution Spatiale (GCN)
    2. Convolution Temporelle (TCN)
    3. Connexion Résiduelle (Skip Connection)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0):
        super().__init__()
        
        # --- Branche Spatiale (GCN) ---
        self.gcn = GraphConv(in_channels, out_channels, kernel_size[1])
        
        # --- Branche Temporelle (TCN) ---
        # Utilise une convolution Nx1 sur l'axe du temps
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, 
                out_channels, 
                (kernel_size[0], 1), 
                (stride, 1), 
                padding=((kernel_size[0] - 1) // 2, 0)
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        
        # --- Connexion Résiduelle ---
        # Si les dimensions changent (stride > 1 ou changement de canaux),
        # on projette l'entrée pour qu'elle corresponde à la sortie.
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        
        # Application GCN (Espace)
        x = self.gcn(x, A)
        
        # Application TCN (Temps)
        x = self.tcn(x)
        
        # Ajout du résidu et activation
        return self.relu(x + res)

class STGCN(nn.Module):
    """
    Architecture Spatial Temporal Graph Convolutional Network (ST-GCN).
    Adaptée pour la classification de séquences squelettiques (Gait Analysis).
    """
    def __init__(self, num_class=2, in_channels=3, edge_importance_weighting=True, **kwargs):
        super().__init__()

        # Chargement de la structure du graphe (Topologie NTU RGB+D)
        self.graph = Graph()
        
        # Enregistrement de la matrice d'adjacence A dans le buffer du modèle
        # (Elle ne sera pas mise à jour par la descente de gradient, mais fait partie de l'état)
        # Shape attendue : (K, V, V) où K est la stratégie spatiale (ex: 3 pour uniform/distance/spatial)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # Normalisation des données d'entrée
        # On normalise sur l'axe temporel pour chaque joint/channel
        self.data_bn = nn.BatchNorm1d(in_channels * 25)
        
        # Architecture simplifiée (Prototype léger)
        # 3 Blocs ST-GCN empilés
        self.st_gcn_networks = nn.ModuleList((
            STGCNBlock(in_channels, 64, (9, 1), 1, dropout=0.1),
            STGCNBlock(64, 64, (9, 1), 1, dropout=0.1),
            STGCNBlock(64, 128, (9, 1), 2, dropout=0.1), # Stride 2 réduit la dimension temporelle
        ))

        # Couche de classification finale (Fully Convolutional)
        self.fcn = nn.Conv2d(128, num_class, kernel_size=1)

    def forward(self, x):
        # Input x : (Batch, Channel, Time, Node) -> (N, C, T, V)
        N, C, T, V = x.size()
        
        # --- 1. Normalisation Initiale ---
        # Permutation pour appliquer BatchNorm1d sur (C * V)
        x = x.permute(0, 3, 1, 2).contiguous() # (N, V, C, T)
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        # Retour au format (N, C, T, V)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        # --- 2. Extraction de Features (Backbone) ---
        for gcn in self.st_gcn_networks:
            # On passe la matrice d'adjacence à chaque couche
            x = gcn(x, self.A)

        # --- 3. Classification Globale ---
        # Global Average Pooling sur l'espace et le temps
        # Réduit (N, 128, T_out, V) -> (N, 128, 1, 1)
        x = F.avg_pool2d(x, x.size()[2:])
        
        # Prédiction (N, Num_Class, 1, 1)
        x = self.fcn(x)
        
        # Aplatissement final (N, Num_Class)
        x = x.view(x.size(0), -1)

        return x

if __name__ == "__main__":
    # Test Unitaire du Modèle
    # Vérifie que les dimensions sont cohérentes pour un passage avant/arrière
    
    model = STGCN(num_class=2)
    
    # Simulation d'un batch de données
    # Batch=2, Channels=3 (XYZ), Time=300 frames, Nodes=25 articulations
    dummy_input = torch.randn(2, 3, 300, 25) 
    
    print("--- Test du modèle ST-GCN ---")
    try:
        output = model(dummy_input)
        print(f"Modèle initialisé avec succès.")
        print(f"Input shape  : {dummy_input.shape}")
        print(f"Output shape : {output.shape} (Attendu: [2, 2])")
        
        # Vérification simple des paramètres
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Paramètres   : {num_params:,} (Léger/Prototype)")
        
    except Exception as e:
        print(f"Erreur lors du test : {e}")