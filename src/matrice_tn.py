import numpy as np
from scipy.optimize import minimize


def matrice_Tim1_Ti(qi, ai_m1, alphai_m1, ri, round_m=()):
    """
    valeur unique qi
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
    if round_m :
        # Arrondissement
        matrix_res = np.round(matrix_res, round_m[0])

        #Si tres petit = 0
        matrix_res[np.abs(matrix_res) < round_m[1]] = 0

    return matrix_res

#mgd
def matrice_Tn(dh, round_m=()):
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
        mat_temp = matrice_Tim1_Ti(dh["sigma_i"][i], dh["a_i_m1"][i], dh["alpha_i_m1"][i], dh["r_i"][i], round_m=round_m)
        mat_list.append(mat_temp)

    result_matrix = np.eye(4)
    for mat in mat_list:
        result_matrix = np.dot(result_matrix, mat)

    # Arrondissement
    if round_m :
        result_matrix = np.round(result_matrix, round_m[0])

        #Si tres petit = 0
        result_matrix[np.abs(result_matrix) < round_m[1]] = 0

    return result_matrix

#MGD avec q liste d'angles, L liste de longueurs
def mgd(q, Lxz, p1):
    rayon_max = 1600
    lz1 = Lxz[0]
    lz2 = Lxz[1]
    lz3 = Lxz[2]
    q=np.radians(q) #Passage a Radians pour utiliser les formules trigo

    teta = (np.pi/2)-np.arctan(3 / 11)
    teta1 = q[0]
    teta2 = q[1]
    teta3 = q[2]

    x = lz1 * np.cos(teta) + lz2 * np.cos(teta + teta2) + lz3 * np.cos(teta + teta2 + teta3)
    x = round(x, 2)
    z = lz1 * np.sin(teta) + lz2 * np.sin(teta + teta2) + lz3 * np.sin(teta + teta2 + teta3)
    z = round(z, 2)

    h=lz1*np.cos(teta)+lz2+lz3
    alpha=np.arcsin(p1/h)
    y = (x / np.cos(teta1)) * np.sin(teta1+alpha)
    y = round(y, 2)

    Xd = np.array([x, y, z])
    ray=round(np.sqrt(x ** 2 + y ** 2 + z ** 2),2)

    if ray <= rayon_max and z >= 0+5:
        return print("Valeurs de q correctes, coordonnés finales (x,y,z): \n",Xd)
    else:
        return print("Valeurs de q incorrectes, dépasement du rayon de 1600 mm (rayon actuel= ",ray, "mm) ou valeur de z négative, coordonnés finales (x,y,z): \n",Xd)



#Fonction qui donne les coordonnées obtenus d'une matrice T(0,n)
def xy_Ot(result_matrix):
    return (result_matrix[:3,-1])

#Matrice pour definir le critére d'erreur
def H(Xd, q):
    X_actuel = mgd(q)
    erreur = Xd - X_actuel
    C = 0.5 * np.linalg.norm(erreur)**2
    return C, erreur