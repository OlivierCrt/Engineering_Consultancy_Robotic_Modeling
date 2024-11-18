import numpy as np
import matplotlib.pyplot as plt
from const_v import *


def matrice_Tim1_Ti(qi, ai_m1, alphai_m1, ri):
    """
    Valeur unique qi
    valeur unique ai-1
    valeur unique alphai-1
    valeur unique ri
    return matrice Ti-1,i avec les éléments arrondis
    """
    matrix_res = np.zeros((4, 4))

    matrix_res[0, 0] = np.cos(qi)
    matrix_res[0, 1] = -np.sin(qi)
    matrix_res[0, 2] = 0
    matrix_res[0, 3] = ai_m1

    matrix_res[1, 0] = np.sin(qi) * np.cos(alphai_m1)
    matrix_res[1, 1] = np.cos(qi) * np.cos(alphai_m1)
    matrix_res[1, 2] = -np.sin(alphai_m1)
    matrix_res[1, 3] = -ri * np.sin(alphai_m1)

    matrix_res[2, 0] = np.sin(qi) * np.sin(alphai_m1)
    matrix_res[2, 1] = np.cos(qi) * np.sin(alphai_m1)
    matrix_res[2, 2] = np.cos(alphai_m1)
    matrix_res[2, 3] = ri * np.cos(alphai_m1)

    matrix_res[3, 0] = 0
    matrix_res[3, 1] = 0
    matrix_res[3, 2] = 0
    matrix_res[3, 3] = 1

    return matrix_res


# mgd
def matrice_Tn(dh):
    """
    qi: liste des qi
    alphai-1 liste des alpha
    ri liste des r
    ai-1 liste
    round tuple : (decimals,threshold)
    return matrice T0,n correspondante avec les éléments arrondis.
    """
    nbliaison = len(dh["sigma_i"])
    mat_list = []
    for i in range(nbliaison):
        mat_temp = matrice_Tim1_Ti(dh["sigma_i"][i], dh["a_i_m1"][i], dh["alpha_i_m1"][i], dh["r_i"][i])
        mat_list.append(mat_temp)

    result_matrix = np.eye(4)
    for mat in mat_list:
        result_matrix = np.dot(result_matrix, mat)

    return result_matrix




def calculer_jacobien(dh):
    """
    Calcule le Jacobien d'un manipulateur à partir des paramètres DH et des matrices de transformation.
    
    Paramètres:
    dh : dict
        Dictionnaire contenant les paramètres DH du manipulateur et les types d'articulations :
        - "sigma_i": liste des types d'articulations (0 pour rotoïde, 1 pour prismatique)
        - "a_i_m1", "alpha_i_m1", "r_i", "sigma_i" : paramètres DH
    
    Retourne:
    np.ndarray
        Jacobien 6xN où N est le nombre d'articulations.
    """
    
    nbliaison = len(dh["sigma_i"])  # Nombre d'articulations
    T_matrices = []  # Stocker les matrices de transformation successives
    result_matrix = np.eye(4)  # Initialiser avec la matrice identité 4x4
    
    # Calcul des matrices de transformation T0_1, T1_2, ..., Tn-1_n
    for i in range(nbliaison):
        # Calculer la transformation de chaque articulation
        mat_temp = matrice_Tim1_Ti(dh["sigma_i"][i], dh["a_i_m1"][i], dh["alpha_i_m1"][i], dh["r_i"][i])
        result_matrix = np.dot(result_matrix, mat_temp)
        T_matrices.append(result_matrix)  # Accumuler la transformation totale

    # Position de l'effecteur final (dernier point de transformation)
    p_e = result_matrix[:3, 3]  # Les trois premières lignes de la dernière colonne de Tn

    # Initialisation du Jacobien
    J_P = []  # Partie de translation du Jacobien
    J_O = []  # Partie de rotation du Jacobien

    # Calcul des colonnes du Jacobien pour chaque articulation
    for i in range(nbliaison):
        # Matrice de transformation de i-1 à i
        T_i = T_matrices[i]
        p_i = T_i[:3, 3]  # Position de l'articulation i
        z_i_1 = T_i[:3, 2]  # Z_i-1 : troisième colonne pour la rotation de l'articulation i-1

        # Différencier entre articulation prismatique et rotoïde
        if dh["sigma_i"][i] == 0:  # Articulation rotoïde
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













# Fonction qui donne les coordonnées obtenus d'une matrice T(0,n)
def xy_Ot(result_matrix):
    return (result_matrix[:3, -1])


# MGD avec q liste d'angles, L liste de longueurs
def mgd(q, Liaisons):
    q_rad = np.radians(q)

    L1 = Liaisons["Liaison 1"]
    L2 = Liaisons["Liaison 2"]
    L3 = Liaisons["Liaison 3"]

    # Angles
    teta1 = q_rad[0]
    teta2 = q_rad[1]
    teta3 = q_rad[2]

    x = L1[0] * np.cos(teta1) + L2[2] * np.cos(teta1 + np.pi / 2) + L2[0] * np.cos(teta1) * np.cos(teta2) + L3[
        2] * np.cos(teta1 - np.pi / 2) + L3[0] * np.cos(teta1) * np.cos(teta3 + teta2)
    y = L1[0] * np.sin(teta1) + L2[2] * np.sin(teta1 + np.pi / 2) + L2[0] * np.sin(teta1) * np.cos(teta2) + L3[
        2] * np.sin(teta1 - np.pi / 2) + L3[0] * np.sin(teta1) * np.cos(teta3 + teta2)
    z = L1[1] + L2[0] * np.sin(teta2) + L3[0] * np.sin(teta3 + teta2)

    Xd = np.array([x, y, z])
    return Xd


jacobien=calculer_jacobien(dh)
print(jacobien)




