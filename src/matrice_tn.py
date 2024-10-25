import numpy as np


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

#mgd
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

def xy_Ot(result_matrix):
    return (result_matrix[:3,-1])








"""def xy_Ot(qi, L):
    qi list
        longueur list

    x = L[0] * np.cos(qi[0]) + L[1] * np.cos(qi[0] + qi[1]) + L[2] * np.cos(qi[0] + qi[1] + qi[2])
    y = L[0] * np.sin(qi[0]) + L[1] * np.sin(qi[0] + qi[1]) + L[2] * np.sin(qi[0] + qi[1] + qi[2])
    return x, y"""

def H(xyOt,Xd,rayon_max_p):
    """xyOt coo calculé avec mgd
        Xd coo demandé"""
    
    if np.sqrt(Xd[0]**2+Xd[1]**2+Xd[2]**2) <= rayon_max_p and Xd[1] >=0:

        return np.linalg.norm((Xd-xyOt))
    print("Parametre probleme on ne peut pas atteindre Xd\n")

    return None
    

# Fonction de coût qui utilise le MGD et retourne l'erreur par rapport à Xe
def fonction_cout(qi, Xe, dh):
    # Mettre à jour les paramètres DH avec les angles qi
    dh["sigma_i"] = qi  # Ajuste les valeurs d'angles avec les `qi` donnés en entrée
    X_calculé = matrice_Tn(dh)[:3, -1]  # Obtient les coordonnées finales avec le MGD et les `qi`
    return np.linalg.norm(Xe - X_calculé)  # Retourne l'écart par rapport à Xe
