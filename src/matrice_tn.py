import numpy as np
import matplotlib.pyplot as plt


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

#Fonction qui donne les coordonnées obtenus d'une matrice T(0,n)
def xy_Ot(result_matrix):
    return (result_matrix[:3,-1])

#MGD avec q liste d'angles, L liste de longueurs
def mgd(q, Lxz, p,inclin_horiz):
    lz1 = Lxz[0]
    lz2 = Lxz[1]
    lz3 = Lxz[2]
    q=np.radians(q) #Passage a Radians pour utiliser les formules trigo

    teta = inclin_horiz
    teta1 = q[0]
    teta2 = q[1]
    teta3 = q[2]

    x = (lz1 * np.cos(teta) + lz2 * np.cos(teta + teta2) + lz3 * np.cos(teta + teta2 + teta3))*np.cos(teta1)
    z = (lz1 * np.sin(teta) + lz2 * np.sin(teta + teta2) + lz3 * np.sin(teta + teta2 + teta3))*np.cos(teta1)

    h=lz1*np.cos(teta)+lz2+lz3
    alpha=np.arcsin(p/h)
    y = (x / np.cos(teta1)) * np.sin(teta1+alpha)

    Xd = np.array([x, y, z])
    return Xd

#Matrice pour definir le critére d'erreur
def H(Xd, q, Lxz,p,inclin_horiz):
    X_actuel = mgd(q,Lxz,p,inclin_horiz)
    erreur = Xd - X_actuel
    C = 0.5 * np.linalg.norm(erreur)**2
    return C, erreur


def jacobienne(q,Lxz,p,inclin_horiz):
    l1 = Lxz[0]
    l2 = Lxz[1]
    l3 = Lxz[2]
    q = np.radians(q)  # Passage a Radians pour utiliser les formules trigo

    teta = inclin_horiz
    teta1 = q[0]
    teta2 = q[1]
    teta3 = q[2]

    h = l1 * np.cos(teta) + l2 + l3
    alpha = np.arcsin(p / h)
    J = np.zeros((3, 3))

    J[0, 0] = 0  # ∂x/∂q1
    J[1, 0] = ((np.sin(teta1)*(l1*np.cos(teta)+l2*np.cos(teta+teta2)+l3*np.cos(teta+teta2+teta3)))/(np.cos(teta1))**2)*np.sin(teta1+alpha)  # ∂y/∂q1
    J[2, 0] = 0  # ∂z/∂q1

    J[0, 1] = -l2 * np.sin(teta + teta2) - l3 * np.sin(teta + teta2 + teta3)  # ∂x/∂q2
    J[1, 1] = ((-l2*np.sin(teta+teta2)-l3*np.sin(teta+teta2+teta3))/np.cos(teta1))*np.cos(teta1+alpha)  # ∂y/∂q2
    J[2, 1] = l2 * np.cos(teta + teta2) + l3 * np.cos(teta + teta2 + teta3)  # ∂z/∂q2

    J[0, 2] = -l3 * np.sin(teta + teta2 + teta3)  # ∂x/∂q3
    J[1, 2] = ((-l3*np.sin(teta+teta2+teta3))/np.cos(teta1))*np.sin(teta1+alpha)    # ∂y/∂q3
    J[2, 2] = l3 * np.cos(teta + teta2 + teta3)  # ∂z/∂q3

    return J

def calcul_direction(q, erreur,Lxz, p1,inclin_horiz):
    J = jacobienne(q,Lxz,p1,inclin_horiz)
    directionG = np.dot(J.T, erreur)
    return directionG

#MGI: On donne une configuration initiale au robot et on demande de ce mettre dans une autre
#On rentre coordonnées et on récupere des angles
def mgi(Xd, q_initial, Lxz, p,inclin_horiz,Nb_iter,pas=1, tolerence=1e-7):
    historique_erreur = []
    q = np.radians(q_initial)

    for i in range(Nb_iter):
        #Calcul initial de Xd et l'erreur
        C, erreur = H(Xd, q, Lxz, p, inclin_horiz)
        norme_error = np.linalg.norm(erreur)
        historique_erreur.append(norme_error)  # Sauvegarde de chaque erreur

        if norme_error < tolerence:
            print(f"Convergence a {i} itérations.")
            break

        directionG=calcul_direction(q, erreur,Lxz, p,inclin_horiz)

        # Actualisation des angles
        q = q + pas * directionG

    # Convertir angles finales de radians a degrés (0°-360°)
    q_final = np.degrees(q)%360
    return q_final, historique_erreur

def evaluer_plusieurs_pas(Xd, q_initial, Lxz, p, inclin_horiz, valeurs_pas, Nb_iter):
    resultats = {}
    for pas in valeurs_pas:
        print(f"\nEvaluation avec un pas de {pas}")
        q_final, historique_erreur = mgi(Xd, q_initial, Lxz, p, inclin_horiz, Nb_iter, pas=pas, tolerence=1e-7)
        resultats[pas] = historique_erreur

    return resultats

def Courbe(resultats, titre, labelx, labely):
    plt.figure(figsize=(10, 6))
    for pas, historique_erreur in resultats.items():
        plt.plot(historique_erreur, label=f"Pas = {pas}")
    plt.title(titre)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.legend()
    plt.grid(True)
    plt.show()


def fonction_cout(qi, Xe, dh):
    # Mettre à jour les paramètres DH avec les angles qi
    dh["sigma_i"] = qi  # Ajuste les valeurs d'angles avec les `qi` donnés en entrée
    X_calculé = matrice_Tn(dh)[:3, -1]  # Obtient les coordonnées finales avec le MGD et les `qi`
    return np.linalg.norm(Xe - X_calculé)  # Retourne l'écart par rapport à Xe.