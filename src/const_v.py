import numpy as np

# Parametres de Denavit Hartenberg modifié
dh = {}
dh["sigma_i"] = [0, 0, 0, 0]
dh["a_i_m1"] = [0, 150, 825, 735]
dh["alpha_i_m1"] = [0, np.pi / 2, 0, 0]
dh["r_i"] = [550, 0, 0, 0]
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
q = [ 0,45 , -90]
rayon_max1_5 = 1600  # en mm

# Pour le MGI
Xd = [800, 0, 600]
q_initial=[0,0,0]

# Pour modélisation 3D du bras robot
Liaisons = {}
"""Dans les listes on a les parametres horizontaux (X), verticaux (Z) et de profondeur (Y), dans cette ordre"""
Liaisons["Liaison 1"] = [150, 550, 0]
Liaisons["Liaison 2"] = [0, 825, 352]
Liaisons["Liaison 3"] = [0, 735, 352]

Robot_pos=[
    [500, 0, 600],
    [1000, -1000, 600],
    [1400, 0, 600],
    [1000, 1000, 600],
    [0, 0, 1600],
    [-1000, 0, 1300],
    [-1000, 0, 900]
]