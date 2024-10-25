import numpy as np

#Parametres de Denavit Hartenberg modifié
dh = {}
dh["sigma_i"] = [0, 0, 0]
dh["a_i_m1"] = [0, 150, 825]
dh["alpha_i_m1"] =[0,np.pi/2,0]
dh["r_i"] = [550, 0, 0]

#Parametres de l'arrondi
decimals=2
threshold=1e-7
round_p=(decimals,threshold)

L = [np.sqrt(550**2+150**2), 348.5, 825 + 735]

rayon_max1_5=1600#en mm
Xd = [700, 200, 500]



