import numpy as np

class Graph:
    """
    Définit la topologie du squelette pour le GNN.
    Inclut les arêtes physiques (os) ET les arêtes fonctionnelles (synergies).
    """
    def __init__(self, strategy='spatial'):
        self.get_edge()
        self.hop_size = 1
        self.strategy = strategy

    def get_edge(self):
        # Configuration standard NTU RGB+D (25 joints)
        # Chaque chiffre correspond à une articulation (ex: 1=Base de la colonne, 21=Main droite...)
        self.num_node = 25
        self_link = [(i, i) for i in range(self.num_node)]
        
        # A. Connexions Physiques (Le squelette réel)
        neighbor_link = [
            (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
            (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
            (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
            (19, 18), (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)
        ]
        
        # B. Connexions Fonctionnelles (TA VALEUR AJOUTÉE pour la thèse)
        # On relie les membres gauche/droite pour capturer la coordination
        # C'est crucial pour détecter les asymétries (AVC, boiterie)
        functional_link = [
            (22, 24), # Main Gauche <-> Main Droite (Coordination bras)
            (16, 20), # Pied Gauche <-> Pied Droit (Coordination marche)
            (15, 19), # Cheville G <-> Cheville D
            (14, 18), # Genou G <-> Genou D
            (23, 25)  # Coude G <-> Coude D
        ]
        
        # On combine tout : Liens soi-même + Voisins physiques + Liens fonctionnels
        self.edge = self_link + neighbor_link + functional_link
        
        # Création de la matrice d'adjacence (A)
        self.A = self.get_adjacency_matrix(self.edge, self.num_node)

    def get_adjacency_matrix(self, edge, num_node):
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            # Note: Le dataset commence à l'index 1, python à 0 -> donc i-1
            A[i-1, j-1] = 1
            A[j-1, i-1] = 1 # Graphe non dirigé
        return A

# Petit test pour vérifier que ça marche
if __name__ == '__main__':
    g = Graph()
    print("Graphe construit avec succès !")
    print(f"Matrice d'adjacence shape: {g.A.shape}")
    print("Nombre d'arêtes fonctionnelles ajoutées : 5 (Coordination G/D)")