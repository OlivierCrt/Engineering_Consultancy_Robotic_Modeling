from const_v import *
from modele_differentiel import *
from matrice_tn import *


#juste pour tester, on donne la fonction après rempl
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

    # Angle de rotation (0 à 2pi pour un cercle complet)
    angle = 2 * np.pi * t

    # Calcul des coordonnées
    x_t = np.array([
        centre[0] + rayon * np.cos(angle),  # Coordonnée x
        centre[1] + rayon * np.sin(angle),  # Coordonnée y
        centre[2]                          # Coordonnée z (constant)
    ])
    return x_t


import numpy as np
import matplotlib.pyplot as plt
from const_v import *
from modele_differentiel import *
from matrice_tn import *

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

    # Angle de rotation (0 à 2pi pour un cercle complet)
    angle = 2 * np.pi * t

    # Calcul des coordonnées
    x_t = np.array([
        centre[0] + rayon * np.cos(angle),  # Coordonnée x
        centre[1],                         # Coordonnée y (constant)
        centre[2] + rayon * np.sin(angle)  # Coordonnée z
    ])
    return x_t

def dx_dt(t):
    """
    Dérivée de la fonction x(t) par rapport au temps t.
    """
    # Définir les points de départ et d'arrivée
    point_depart = np.array([800, 0, 600])
    point_arrivee = np.array([1400, 0, 600])

    # Calcul du rayon et du centre du cercle
    centre = (point_depart + point_arrivee) / 2
    rayon = np.linalg.norm(point_depart - centre)

    # Angle de rotation (0 à 2pi pour un cercle complet)
    angle = 2 * np.pi * t

    # Vitesse angulaire
    omega = 2 * np.pi

    # Calcul des dérivées des coordonnées
    dx_t = np.array([
        -rayon * omega * np.sin(angle),  # Dérivée de x
        0,                              # Dérivée de y (constant)
        rayon * omega * np.cos(angle)   # Dérivée de z
    ])
    return dx_t













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
print(jacob)

vitesses_ot = MDD([100,100,100],jacob)
print(vitesses_ot)

vitesse_q =MDI(vitesses_ot , jacob)
print(vitesse_q)


def plot_mdi():
    """
    Fonction pour tracer la trajectoire et les vitesses générées par le MDI.
    """
    # Discrétisation du temps
    t_values = np.linspace(0, 2, 100)  # Temps de 0 à 2 secondes

    # Initialiser les trajectoires et vitesses
    positions = []
    vitesses = []
    
    for t in t_values:
        pos = dx_dt(t)  # Position désirée
        positions.append(pos)
        
        # Calcul des vitesses en sortie
        vitesses_ot = MDD(pos, jacob)
        vitesses_q = MDI(vitesses_ot, jacob)
        vitesses.append(vitesses_q)

    positions = np.array(positions)
    vitesses = np.array(vitesses)

    # Tracer les positions (trajectoire en 3D)
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], label="Trajectoire x(t)")
    ax1.set_title("Trajectoire en 3D")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.legend()

    # Tracer les vitesses générées par le MDI
    ax2 = fig.add_subplot(122)
    for i in range(vitesses.shape[1]):
        ax2.plot(t_values, vitesses[:, i], label=f"vitesse_q[{i}]")
    ax2.set_title("Vitesses générées par MDI")
    ax2.set_xlabel("Temps (s)")
    ax2.set_ylabel("Vitesses")
    ax2.legend()

    plt.tight_layout()
    plt.show()

# Appeler la fonction de tracé
plot_mdi()