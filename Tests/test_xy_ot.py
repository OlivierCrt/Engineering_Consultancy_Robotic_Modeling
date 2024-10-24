from matrice_tn import matrice_Tim1_Ti,matrice_Tn
import numpy as np

#Input data

qi=[0,0,0]
ri=[550,0,-225]
ai_m1=[0,150,225]

alphai_m1=[0,np.pi/2,0]
matrice_T0Tn=matrice_Tn(qi,alphai_m1,ri,ai_m1)
print (matrice_T0Tn)

   




