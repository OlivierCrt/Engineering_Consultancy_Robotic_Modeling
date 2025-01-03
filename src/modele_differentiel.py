import numpy as np
import time
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
    import numpy as np

    n = len(T_matrices)
    T_0i = np.eye(4)  # Transformation cumulée initiale
    p_e = T_matrices[-1][:3, 3]  # Position de l'effecteur final dans R0

    J_P = np.zeros((3, n))
    J_O = np.zeros((3, n))

    for i in range(n):
        T_0i = T_0i @ T_matrices[i]  # Calcul cumulatif de la transformation
        z_i_1 = T_0i[:3, 2]  # Axe Z dans R0
        p_i = T_0i[:3, 3]  # Position dans R0

        if types_articulations[i] == 0:  # Articulation rotoïde
            J_P[:, i] = np.cross(z_i_1, p_e - p_i)
            J_O[:, i] = z_i_1
        else:  # Articulation prismatique
            J_P[:, i] = z_i_1
            J_O[:, i] = np.zeros(3)

    J = np.vstack((J_P, J_O))
    return J

def MDD(v,J) :
    """
    Return vitesses OT
    parametre Vitesses articulaires, J jacobienne

    """
    return np.dot(J,v)


def MDI(x,J):
    """return q vitesse
    param : x vitesse de l OT souhaitée , J jacobienne"""
    return np.dot(np.linalg.pinv(J),x)


