import numpy as np

# Parametres de Denavit Hartenberg modifié
dh = {}
dh["sigma_i"] = [0, 0, 0]
dh["a_i_m1"] = [0, 150, 825]
dh["alpha_i_m1"] = [0, np.pi / 2, 0]
dh["r_i"] = [550, 0, -3.5]
max_dist = 2110  # Distance maximale lorsque le bras est totalement tendu vers le haut (en mm)

# Parametres de l'arrondi
decimals = 2
threshold = 1e-7
round_p = (decimals, threshold)

# Normes des longueurs du bras
# Param de horiz/vert
l1 = np.sqrt(dh["r_i"][0] ** 2 + dh["a_i_m1"][1] ** 2)
l2 = dh["a_i_m1"][2]
l3 = max_dist - (dh["r_i"][0] + l2)
Lxz = [l1, l2, l3]

# Param de profondeur
p1 = dh["r_i"][2]

# Inclinaison horizontale de l1
inclin_horiz = (np.pi / 2) - np.arctan(dh["r_i"][0] / dh["a_i_m1"][1])

# Angles des liaisons en Degrés
q = [0, 40, -90]
rayon_max1_5 = 1600  # en mm

# Pour le MGI
Xd = [800, 0, 400]
# q_initial=[0,-90,0]
Nb_iter = 10000

# Pour modélisation 3D du bras robot
Liaisons = {}
"""Dans les listes on a les parametres horizontaux, verticaux et de profondeur, dans cette ordre"""
Liaisons["Liaison 1"] = [150, 550, 0]
Liaisons["Liaison 2"] = [825, 0, 348.5]
Liaisons["Liaison 3"] = [735, 0, 352]
