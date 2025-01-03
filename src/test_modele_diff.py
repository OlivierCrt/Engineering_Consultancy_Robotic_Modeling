from const_v import *
from modele_differentiel import *
from matrice_tn import *
import numpy as np
import matplotlib.pyplot as plt


#juste pour tester othman donne la fonction après rempl

def x(t):
    """
    Fonction x(t) définie pour tester le MDI dans le plan x-z.
    Trajectoire circulaire entre deux points dans le plan x-z.
    """
    # Définir les points de départ et d'arrivée
    point_depart = np.array([800, 0, 600])
    point_arrivee = np.array([1400, 0, 600])

    # Calcul du rayon et du centre du cercle
    centre = (point_depart + point_arrivee) / 2
    rayon = np.linalg.norm(point_depart - centre)

    # Angle de rotation (0 à 2π pour un cercle complet)
    angle = 2 * np.pi * t

    # Calcul des coordonnées uniquement dans le plan x-z
    x_t = np.array([
        centre[0] + rayon * np.cos(angle),  # Coordonnée x
        centre[1],                          # Coordonnée y (constante)
        centre[2] + rayon * np.sin(angle)   # Coordonnée z
    ])
    return x_t



# Juste pour tester on donne la fonction après
def x(t):
    """
    Fonction x(t) définie pour tester le MDI.
    Trajectoire circulaire entre deux points dans un plan donné.
    """
    # Définir les points de départ et d'arrivée
    point_depart = np.array([800, 0, 600])
    point_arrivee = np.array([1400, 0, 600])

    # Calcul du rayon et du centre du cercle
    centre = (point_depart + point_arrivee) / 2
    rayon = np.linalg.norm(point_depart - centre)

    # Dérivée temporelle
    d_x = -2 * np.pi * rayon * np.sin(2 * np.pi * t)  # dérivée de x(t)
    d_z = 2 * np.pi * rayon * np.cos(2 * np.pi * t)   # dérivée de z(t)
    d_y = 0  # La dérivée de y(t) est toujours 0

    return np.array([d_x, d_y, d_z])
# Juste pour tester othman donne la fonction après
















# Liste pour stocker les matrices T(i, i+1)
T_matrices = []

# Calculer et afficher les matrices de transformation T(i, i+1)
for i in range(len(dh['sigma_i'])):
    print(f"Transformation T({i},{i + 1}):\n")
    t_i_ip1 = matrice_Tim1_Ti(dh["sigma_i"][i], dh["a_i_m1"][i], dh["alpha_i_m1"][i], dh["r_i"][i])
    if round_p:
        # Arrondi
        t_i_ip1_rounded = np.round(t_i_ip1, round_p[0])
        t_i_ip1_rounded[np.abs(t_i_ip1_rounded) < round_p[1]] = 0
        print(f"{t_i_ip1_rounded}\n")
        T_matrices.append(t_i_ip1_rounded)
    else:
        T_matrices.append(t_i_ip1)


jacob =calculer_jacobien(T_matrices[:3] ,[0,0,0])
print("Jacobienne:\n",jacob)

vitesses_ot = MDD([100,100,100],jacob)
print("\nVitesse ot:\n",vitesses_ot)

vitesse_q =MDI(vitesses_ot , jacob)
print("\nVitesse q:\n",vitesse_q)



