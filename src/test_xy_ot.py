from matrice_tn import *
from const_v import *
import numpy as np
from scipy.optimize import minimize
# Input data

# Afficher chaque transformation pour suivre le calcul
for i in range(len(dh['sigma_i'])):
    print(f"Transformation T({i},{i+1}):\n")
    t_i_ip1 = matrice_Tim1_Ti(dh["sigma_i"][i], dh["a_i_m1"][i], dh["alpha_i_m1"][i], dh["r_i"][i])
    if round_p :
        # Arrondi
        t_i_ip1_rounded = np.round(t_i_ip1, round_p[0])

        #Si tres petit = 0
        t_i_ip1_rounded[np.abs(t_i_ip1) < round_p[1]] = 0
        print(f"{t_i_ip1_rounded}\n")


# Calcul de la transformation complète T(0,3)
print(f"Transformation T(0,{len(dh['sigma_i'])}):\n")
matrice_T0Tn = matrice_Tn(dh)
print(f"{matrice_T0Tn}\n")
if round_p:
    # Arrondi
    matrice_T0Tn_rounded = np.round(matrice_T0Tn, round_p[0])

    # Si tres petit = 0
    matrice_T0Tn_rounded[np.abs(matrice_T0Tn) < round_p[1]] = 0
    print(f"{matrice_T0Tn_rounded}\n")


# Extraction des coordonnées (x, y, z) de la transformation finale
xyz = xy_Ot(matrice_T0Tn)
print("Coordonnées finales (x, y, z):\n")
print(f"{xyz}\n")
print(f"{H(xyz,Xd,rayon_max_p=rayon_max1_5)}\n")




# Définition de la position cible
Xe = np.array([700, 200, 500])

# Valeurs initiales pour qi (exemple)
qi_initiale = [0, 0, 0]  # À ajuster en fonction du robot

# Bornes pour chaque angle
bounds = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]

# Optimisation pour trouver le MGI
result = minimize(fonction_cout, x0=qi_initiale, args=(Xe, dh), bounds=bounds)

# Résultats
if result.success:
    qi_solution = result.x
    print("Solution pour les angles qi :", qi_solution)
else:
    print("La solution n'a pas convergé.")