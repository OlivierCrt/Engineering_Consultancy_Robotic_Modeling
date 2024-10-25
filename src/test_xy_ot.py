from matrice_tn import *
from const_v import *
import numpy as np

# Input data

# Afficher chaque transformation pour suivre le calcul
for i in range(len(dh['sigma_i'])):
    print(f"Transformation T({i},{i+1}):\n")
    t_i_ip1 = matrice_Tim1_Ti(dh["sigma_i"][i], dh["a_i_m1"][i], dh["alpha_i_m1"][i], dh["r_i"][i], round_m=round_p)
    print(f"{t_i_ip1}\n")


# Calcul de la transformation complète T(0,3)
print(f"Transformation T(0,{len(dh['sigma_i'])}):\n")
matrice_T0Tn = matrice_Tn(dh, round_m=round_p)
print(f"{matrice_T0Tn}\n")

# Extraction des coordonnées (x, y, z) de la transformation finale
xyz = xy_Ot(matrice_T0Tn)
print("Coordonnées finales (x, y, z):\n")
print(f"{xyz}\n")
print(f"{H(xyz,Xd,rayon_max_p=rayon_max1_5)}\n")
