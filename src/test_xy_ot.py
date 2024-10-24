from matrice_tn import *
import numpy as np

<<<<<<< HEAD
# Input data

qi = [0, 0, 0]
ri = [550, 0, 0]
ai_m1 = [0, 150, 825]
L = [550, 348.5, 825 + 735]

t01 = matrice_Tim1_Ti(0, 0, 0, 550)
print(t01)
print("\n")
t12 = matrice_Tim1_Ti(0, 150, np.pi / 2, 0)
print("t12:\n",t12)
print("\n")
print("t23:")
t23 = matrice_Tim1_Ti(0, 825, 0, 0)
print(t23)
alphai_m1 = [0, np.pi / 2, 0]
print("\nT(0,3):")
matrice_T0Tn = matrice_Tn(qi, alphai_m1, ri, ai_m1)
print(matrice_T0Tn)

x, y = xy_Ot(qi, L)
print("\n")
print(x, y)
=======
#Input data

qi=[0,0,0]
ri=[550,0,0]
ai_m1=[0,150,825]
L=[550,348.5,825+735]

t01=matrice_Tim1_Ti(0,0,0,550)
print(t01)
print("\n")
t12=matrice_Tim1_Ti(0,150,np.pi/2,0)
print(t12)
print("\n")
print("t23:\n")
t23=matrice_Tim1_Ti(0,825,0,0)
print(t23)
alphai_m1=[0,np.pi/2,0]
matrice_T0Tn=matrice_Tn(qi,alphai_m1,ri,ai_m1)
print (matrice_T0Tn)

x,y=xy_Ot(qi,L)
print ("\n\n")
print(x,y)

   



>>>>>>> 319975e397d9aa9b4a37ca114d824bb449eb2492

