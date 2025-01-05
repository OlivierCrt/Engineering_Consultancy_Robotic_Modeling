from matrice_tn import *
from const_v import *
import numpy as np

from src.calcul_matrix import MDI_analytique
from trajectory_generation import *
from modele_differentiel import *

# Afficher chaque matrice de transformation pour suivre le calcul et enregistrer dans une liste les matrices
q = [0, 0, 0]
transformation_matrices = generate_transformation_matrices(q,dh, round_p=(2, 1e-6))

# Calcul de la transformation complète T(0,4)
print(f"Transformation T(0,{len(dh['sigma_i'])}) :\n")
matrice_T0Tn = matrice_Tn(dh,q)
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

# Calcule de Jacobienne geometrique
J_geo = Jacob_geo(transformation_matrices)
print("\nJacobienne geométrique:")
print(J_geo)
print(len(J_geo[0]))
# Calcule de Jacobienne analytique
# Matrices sous forme analytique
Mats = Mat_T_analytiques()
Jacob_an = Jacob_analytique(Mats)
print("\nJacobienne analytique:")
sp.pprint(Jacob_an)


# MDD pour dq1=0.1, dq2=0.2, dq3=0.3 appliqué à la position initiale q1=0, q2=0 et q3=0
dq = [0.1, 0.3, 0.2]
dX = MDD(dq, J_geo)
dX_vert = np.array(dX).reshape(-1, 1)
print("\nValeurs des vitesses linéaires et angulaires du robot pour sa position initiale lorsqu'on applique dq1 =",
      dq[0], ", dq2 =", dq[1], ", dq3 =", dq[2])
print(dX_vert)

# Verification en utilisant MDI inversant la Jacobienne
dq = MDI(dX, J_geo)
dq_vert = np.array(dq).reshape(-1, 1)
print("\nCalcul GEOMETRIQUE des valeurs des vitesses articulaires du robot pour sa position initiale lorsqu'on applique dx =", dX[0], ", dy=",
      dX[1], ", dz=", dX[2], ", wx=", dX[3], ", wy=", dX[4], ", wz=", dX[5])
print(dq_vert)

dq=MDI_analytique(J_geo, dX, q_initial)
dq_vert = np.array(dq).reshape(-1, 1)
print("\nCalcul ANALYTIQUE des valeurs des vitesses articulaires du robot pour sa position initiale lorsqu'on applique dx =", dX[0], ", dy=",
      dX[1], ", dz=", dX[2], ", wx=", dX[3], ", wy=", dX[4], ", wz=", dX[5])
print(dq_vert)

print("\n")
# Genération de trajectoire
# Test génération de trajectoire
V1 = 10  # Vitesse 1 (par exemple)
V2 = 20  # Vitesse 2 (par exemple)

A = np.array([500, 0, 600])  # Ajusté pour respecter z_min
B = np.array([500, 0, 900])  # Ajusté pour respecter z_min

result_a, message_a = est_point_atteignable(A)
result_b, message_b = est_point_atteignable(B)

print(f"Point A: {message_a}")
print(f"Point B: {message_b}")

(q, qp) = traj(A, B, V1, V2, Debug=True)
