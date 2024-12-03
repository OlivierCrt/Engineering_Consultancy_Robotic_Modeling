from matrice_tn import *
from const_v import *
import numpy as np
from trajectory_generation import *

# Afficher chaque transformation pour suivre le calcul
for i in range(len(dh['sigma_i'])):
    print(f"Transformation T({i},{i + 1}):\n")
    t_i_ip1 = matrice_Tim1_Ti(dh["sigma_i"][i], dh["a_i_m1"][i], dh["alpha_i_m1"][i], dh["r_i"][i])
    if round_p:
        # Arrondi
        t_i_ip1_rounded = np.round(t_i_ip1, round_p[0])
        t_i_ip1_rounded[np.abs(t_i_ip1) < round_p[1]] = 0
        print(f"{t_i_ip1_rounded}\n")

# Calcul de la transformation complète T(0,3)
print(f"Transformation T(0,{len(dh['sigma_i'])}) :\n")
matrice_T0Tn = matrice_Tn(dh)
print(f"{matrice_T0Tn}\n")


# Pour ce TP Z0 représente l'axe vertical et Y0 celui de la profondeur
print("\nCoordonnées finales grace a matrice T(0,n) en fonction de X0,Y0,Z0:\n", xy_Ot(matrice_T0Tn))
print("\nCoordonnées (x, y, z) en mm en fonction des angles de la liste q:")
Xd_mgd = mgd(q, Liaisons)
x_mgd = Xd_mgd[0]
y_mgd = Xd_mgd[1]
z_mgd = Xd_mgd[2]
ray = round(np.sqrt(x_mgd ** 2 + y_mgd ** 2 + z_mgd ** 2), 2)
if ray <= rayon_max1_5 and z_mgd >= 0 + 5:
    print("Valeurs de q correctes, coordonnés finales (x,y,z): \n", Xd_mgd)
else:
    print("Valeurs de q incorrectes, dépasement du rayon de 1600 mm (rayon actuel= ", ray,
          "mm) ou valeur de z négative, coordonnés finales (x,y,z): \n", Xd_mgd)

verifier_solutions(Xd, Liaisons)

#Genération de trajectoire
# Test génération de trajectoire
V1 = 10  # Vitesse 1 (par exemple)
V2 = 20  # Vitesse 2 (par exemple)

A = np.array([5, 5, 0])  # (par exemple)
B = np.array([10, 20, 0])  # (par exemple)

a = traj(A,B,V1,V2,Debug=True)