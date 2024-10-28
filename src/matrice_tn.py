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


# Matrice pour definir le critére d'erreur
def H(Xd, q, Liaisons):
    X_actuel = mgd(q, Liaisons)
    erreur = Xd - X_actuel
    C = 0.5 * np.linalg.norm(erreur) ** 2
    return C, erreur


def jacobienne(q, Liaisons):
    l1 = Liaisons["Liaison 1"]
    l2 = Liaisons["Liaison 2"]
    l3 = Liaisons["Liaison 3"]
    q = np.radians(q)  # Passage a Radians pour utiliser les formules trigo

    teta1 = q[0]
    teta2 = q[1]
    teta3 = q[2]

    J = np.zeros((3, 3))

    J[0, 0] = l1[0] * np.sin(teta1) - l2[2] * np.sin(teta1 + (np.pi / 2)) + l2[0] * np.sin(teta1) * np.cos(teta2) - l3[
        2] * np.sin(teta1 - (np.pi / 2)) - l3[0] * np.sin(teta1) * np.cos(teta3)  # ∂x/∂q1
    J[1, 0] = l1[0] * np.cos(teta1) - l2[2] * np.cos(teta1 + (np.pi / 2)) + l2[0] * np.cos(teta1) * np.cos(teta2) - l3[
        2] * np.cos(teta1 - (np.pi / 2)) - l3[0] * np.cos(teta1) * np.cos(teta3)  # ∂y/∂q1
    J[2, 0] = 0  # ∂z/∂q1

    J[0, 1] = -l2[0] * np.cos(teta1) * np.sin(teta2)  # ∂x/∂q2
    J[1, 1] = -l2[0] * np.sin(teta1) * np.sin(teta2)  # ∂y/∂q2
    J[2, 1] = l2[0] * np.cos(teta2)  # ∂z/∂q2

    J[0, 2] = -l3[0] * np.cos(teta1) * np.sin(teta3)  # ∂x/∂q3
    J[1, 2] = -l3[0] * np.sin(teta1) * np.sin(teta3)  # ∂y/∂q3
    J[2, 2] = l3[0] * np.cos(teta3)  # ∂z/∂q3

    return J


def calcul_direction(q, erreur, Liaisons):
    J = jacobienne(q, Liaisons)
    directionG = np.dot(J.T, erreur)
    return directionG


# MGI: On donne une configuration initiale au robot et on demande de ce mettre dans une autre
# On rentre coordonnées et on récupere des angles
def mgi(Xd, q_initial, Liaisons, Nb_iter, pas=1, tolerence=1e-7):
    historique_erreur = []
    q = np.radians(q_initial)

    for i in range(Nb_iter):
        # Calcul initial de Xd et l'erreur
        C, erreur = H(Xd, q, Liaisons)
        norme_error = np.linalg.norm(erreur)
        historique_erreur.append(norme_error)  # Sauvegarde de chaque erreur

        if norme_error < tolerence:
            print(f"Convergence a {i} itérations.")
            break

        directionG = calcul_direction(q, erreur, Liaisons)

        # Actualisation des angles
        q = q + pas * directionG

    # Convertir angles finales de radians a degrés (0°-360°)
    q_final = np.degrees(q) % 360
    return q_final, historique_erreur


def evaluer_plusieurs_pas(Xd, q_initial, Liaisons, valeurs_pas, Nb_iter):
    resultats = {}
    pas = 0
    for pas in valeurs_pas:
        print(f"\nEvaluation avec un pas de {pas}")
        q_final, historique_erreur = mgi(Xd, q_initial, Liaisons, Nb_iter, pas=pas, tolerence=1e-7)
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
