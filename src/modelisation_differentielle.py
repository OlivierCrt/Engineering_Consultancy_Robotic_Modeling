from const_v import *



import numpy as np

def calculer_jacobien(T_matrices, types_articulations):
    """
    Calcule le Jacobien d'un manipulateur à partir d'une liste de matrices de transformation 4x4.
    
    Paramètres:
    T_matrices : list of np.ndarray
        Liste des matrices de transformation homogène 4x4 de chaque articulation jusqu'à l'effecteur final.
    types_articulations : list of int
        Liste des types d'articulations (0 pour rotoïde, 1 pour prismatique).
        
    Retourne:
    np.ndarray
        Jacobien 6xN où N est le nombre d'articulations.
    """
    n = len(T_matrices)  # Nombre d'articulations

    # Position de l'effecteur final
    p_e = T_matrices[-1][:3, 3]  # Position de l'effecteur final (les trois premières lignes de la dernière colonne)

    # Initialisation des parties de translation et de rotation du Jacobien
    J_P = []  # Partie de translation du Jacobien
    J_O = []  # Partie de rotation du Jacobien

    # Calcul des colonnes du Jacobien pour chaque articulation
    for i in range(n):
        T_i = T_matrices[i]        # Matrice de transformation pour l'articulation i
        p_i = T_i[:3, 3]           # Position de l'articulation i
        z_i_1 = T_i[:3, 2]         # Axe z de la rotation pour l'articulation i (troisième colonne)

        # Différencier entre articulation prismatique et rotoïde
        if types_articulations[i] == 0:  # Articulation rotoïde
            J_P_i = np.cross(z_i_1, p_e - p_i)
            J_O_i = z_i_1
        else:  # Articulation prismatique
            J_P_i = z_i_1
            J_O_i = np.zeros(3)

        # Ajouter les colonnes de JP et JO pour cette articulation
        J_P.append(J_P_i)
        J_O.append(J_O_i)

    # Conversion en matrices numpy pour la sortie
    J_P = np.array(J_P).T  # Transformer en une matrice 3xN
    J_O = np.array(J_O).T  # Transformer en une matrice 3xN

    # Concatenation pour obtenir le Jacobien 6xN
    J = np.vstack((J_P, J_O))
    
    return J
