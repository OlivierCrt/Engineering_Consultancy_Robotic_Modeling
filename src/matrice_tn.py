import numpy as np
import matplotlib.pyplot as plt


def matrice_Tim1_Ti(qi, ai_m1, alphai_m1, ri):
    """
    Calcule la matrice de transformation DH entre deux liaisons successives.

    Arguments :
        qi : Angle de la liaison i (en radians).
        ai_m1 : Longueur entre les axes des liaisons (en mm).
        alphai_m1 : Angle entre les axes z_{i-1} et z_i (en radians).
        ri : Décalage suivant z_i (en mm).

    Retourne :
        Matrice 4x4 de transformation DH.
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


def generate_transformation_matrices(q, dh, round_p=None):
    """
    Génère une liste de matrices de transformation T(i, i+1) à partir des paramètres DH.

    Arguments :
        dh : Dictionnaire contenant les paramètres DH (a_i_m1, alpha_i_m1, r_i).
        q : Liste des angles articulaires (en radians).
        round_p : Tuple (nombre de décimales, seuil pour arrondir à 0).

    Retourne :
        Liste de matrices 4x4 de transformation T(i, i+1).
    """
    transformation_matrices = []

    for i in range(len(dh['a_i_m1'])):
        t_i_ip1 = matrice_Tim1_Ti(q[i], dh['a_i_m1'][i], dh['alpha_i_m1'][i], dh['r_i'][i])

        if round_p:
            t_i_ip1 = np.round(t_i_ip1, round_p[0])
            t_i_ip1[np.abs(t_i_ip1) < round_p[1]] = 0

        transformation_matrices.append(t_i_ip1)

    return transformation_matrices


# mgd
def matrice_Tn(dh, q):
    """
    Calcule la matrice T0,n en utilisant les paramètres DH et les angles q.

    Arguments :
        dh : Dictionnaire contenant les paramètres DH.
        q : Liste des angles articulaires (en radians).

    Retourne :
        Matrice T0,n (4x4).
    """
    nbliaison = len(dh['a_i_m1'])
    result_matrix = np.eye(4)

    for i in range(nbliaison):
        mat_temp = matrice_Tim1_Ti(q[i], dh['a_i_m1'][i], dh['alpha_i_m1'][i], dh['r_i'][i])
        result_matrix = np.dot(result_matrix, mat_temp)

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

    x = L1[0] * np.cos(teta1) + L2[2] * np.cos(teta1 + np.pi / 2) + L2[1] * np.cos(teta1) * np.cos(teta2) + \
        L3[
            2] * np.cos(teta1 - np.pi / 2) + L3[1] * np.cos(teta1) * np.cos(teta3 + teta2)
    y = L1[0] * np.sin(teta1) + L2[2] * np.sin(teta1 + np.pi / 2) + L2[1] * np.sin(teta1) * np.cos(teta2) + \
        L3[
            2] * np.sin(teta1 - np.pi / 2) + L3[1] * np.sin(teta1) * np.cos(teta3 + teta2)
    z = L1[1] + L2[1] * np.sin(teta2) + L3[1] * np.sin(teta3 + teta2)

    Xd = np.array([x, y, z])
    return Xd


# MGI: On donne une configuration initiale au robot et on demande de ce mettre dans une autre
# On rentre coordonnées et on récupere des angles
def mgi(Xd, Liaisons):
    x = Xd[0]
    y = Xd[1]
    z = Xd[2]
    L1 = Liaisons["Liaison 1"]
    L2 = Liaisons["Liaison 2"]
    L3 = Liaisons["Liaison 3"]
    solutions = []

    def calculer_solutions(q1):
        # Constantes données
        X = L2[1]
        Y = L3[1]
        Z1 = np.cos(q1) * x + np.sin(q1) * y - L1[0]
        Z2 = z - L1[1]

        # Calcul de q3
        c3 = (Z1 ** 2 + Z2 ** 2 - X ** 2 - Y ** 2) / (2 * X * Y)

        # Limiter c3 à l'intervalle [-1, 1]
        if c3 < -1 or c3 > 1:
            return []  # Aucune solution si c3 est hors de portée

        c3 = np.clip(c3, -1, 1)  # Forcer c3 dans l'intervalle [-1, 1]

        q31 = np.arctan2(np.sqrt(1 - c3 ** 2), c3)
        q32 = np.arctan2(-1 * np.sqrt(1 - c3 ** 2), c3)

        # Calcul de B1, B21 et B22
        B1 = X + Y * c3
        B21 = Y * np.sin(q31)
        B22 = Y * np.sin(q32)

        # Calcul de q2 pour les deux solutions de q3
        s21 = (B1 * Z2 - B21 * Z1) / (B1 ** 2 + B21 ** 2)
        c21 = (B1 * Z1 + B21 * Z2) / (B1 ** 2 + B21 ** 2)
        q21 = np.arctan2(s21, c21)

        s22 = (B1 * Z2 - B22 * Z1) / (B1 ** 2 + B22 ** 2)
        c22 = (B1 * Z1 + B22 * Z2) / (B1 ** 2 + B22 ** 2)
        q22 = np.arctan2(s22, c22)

        # Convertir les angles en degrés
        q1_deg = np.degrees(q1)
        q21 = np.degrees(q21)
        q22 = np.degrees(q22)
        q31 = np.degrees(q31)
        q32 = np.degrees(q32)

        # Retourner les deux solutions possibles pour ce q1
        return [
            [q1_deg, q21, q31],
            [q1_deg, q22, q32]
        ]

    # Premier ensemble de solutions en utilisant q1 = arctan2(y, x)
    q1_1 = np.arctan2(y, x)
    solutions.extend(calculer_solutions(q1_1))

    # Deuxième ensemble de solutions en utilisant q1 = arctan2(y, x) - pi
    q1_2 = q1_1 - np.pi
    solutions.extend(calculer_solutions(q1_2))

    return solutions



def verifier_solutions(Xd, Liaisons):
    # Obtenir toutes les combinaisons possibles d'angles avec la fonction mgi
    solutions = mgi(Xd, Liaisons)
    solutions = np.round(solutions, 2)  # Arrondir pour une meilleure lisibilité

    print("\nValeurs possibles des angles pour atteindre la configuration souhaitée Xd:", Xd)
    print(solutions)

    # Itérer sur chaque combinaison d'angles et vérifier avec la fonction mgd
    for i, q in enumerate(solutions):
        # Calculer les coordonnées (x, y, z) en utilisant la fonction mgd avec les angles q
        Xd_mgd = mgd(q, Liaisons)
        Xd_mgd = np.round(Xd_mgd, 2)  # Arrondir pour comparer

        # Calculer l'erreur entre les coordonnées souhaitées et celles obtenues
        erreur = np.linalg.norm(Xd_mgd - Xd)

        # Afficher le résultat pour chaque ensemble d'angles
        print(f"\nVérification de la solution {i + 1}: Angles = {q}")
        print(f"Coordonnées obtenues par MGD: {Xd_mgd}")
        print(f"Erreur par rapport à Xd: {np.round(erreur,3)}")

        if erreur < 0.1:  # Tolérance pour considérer que la solution est correcte
            print("Résultat : Correct !")
        else:
            print("Résultat : Incorrect")