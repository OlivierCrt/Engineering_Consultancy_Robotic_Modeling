from matrice_tn import *
from const_v import *
import numpy as np

# Input data

# Afficher chaque transformation pour suivre le calcul
for i in range(len(dh['sigma_i'])):
    print(f"Transformation T({i},{i+1}):")
    t_i_ip1 = matrice_Tim1_Ti(dh["sigma_i"][i], dh["a_i_m1"][i], dh["alpha_i_m1"][i], dh["r_i"][i], round_m=round_p)
    print(t_i_ip1)
    print("\n")

# Calcul de la transformation complète T(0,3)
print("Transformation T(0,3):")
matrice_T0Tn = matrice_Tn(dh, round_m=round_p)
print(matrice_T0Tn)

#Pour ce TP Z0 représente l'axe vertical et Y0 celui de la profondeur
print("\nCoordonnées finales grace a matrice T(0,n) en fonction de X0,Y0,Z0:\n",xy_Ot(matrice_T0Tn))
# print(H(xyz,Xd,rayon_max_p=rayon_max1_5))

print("\nCoordonnées (x, y, z) en mm en fonction des angles de la liste q et X0, Y0, Z0:")
mgd(q,Lxz,p1)

