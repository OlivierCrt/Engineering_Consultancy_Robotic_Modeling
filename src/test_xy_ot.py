from matrice_tn import *
from const_v import *
import numpy as np

# Input data

# Afficher chaque transformation pour suivre le calcul
print("Transformation T(0,1):")
t01 = matrice_Tim1_Ti(qi[0], ai_m1[0], alphai_m1[0], ri[0], round_m=round_p)
print(t01)
print("\n")

print("Transformation T(1,2):")
t12 = matrice_Tim1_Ti(qi[1], ai_m1[1], alphai_m1[1], ri[1], round_m=round_p)
print(t12)
print("\n")

print("Transformation T(2,3):")
t23 = matrice_Tim1_Ti(qi[2], ai_m1[2], alphai_m1[2], ri[2], round_m=round_p)
print(t23)
print("\n")

# Calcul de la transformation complète T(0,3)
print("Transformation T(0,3):")
matrice_T0Tn = matrice_Tn(qi, alphai_m1, ri, ai_m1, round_m=round_p)
print(matrice_T0Tn)

# Extraction des coordonnées (x, y, z) de la transformation finale
xyz = xy_Ot(matrice_T0Tn)
print("\nCoordonnées finales (x, y, z):")
print(xyz)
print(H(xyz,Xd,rayon_max_p=rayon_max1_5))
