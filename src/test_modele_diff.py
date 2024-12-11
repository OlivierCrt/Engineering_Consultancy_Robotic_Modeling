from const_v import *
from modele_differentiel import *
from matrice_tn import *





# Liste pour stocker les matrices T(i, i+1)
T_matrices = []

# Calculer et afficher les matrices de transformation T(i, i+1)
for i in range(len(dh['sigma_i'])):
    print(f"Transformation T({i},{i + 1}):\n")
    t_i_ip1 = matrice_Tim1_Ti(dh["sigma_i"][i], dh["a_i_m1"][i], dh["alpha_i_m1"][i], dh["r_i"][i])
    if round_p:
        # Arrondi
        t_i_ip1_rounded = np.round(t_i_ip1, round_p[0])
        t_i_ip1_rounded[np.abs(t_i_ip1_rounded) < round_p[1]] = 0
        print(f"{t_i_ip1_rounded}\n")
        T_matrices.append(t_i_ip1_rounded)
    else:
        T_matrices.append(t_i_ip1)


jacob =calculer_jacobien(T_matrices[:3] ,[0,0,0])
print(jacob)

vitesses_ot = MDD([100,100,100],jacob)
print(vitesses_ot)

vitesse_q =MDI(vitesses_ot , jacob)
print(vitesse_q)