from matrice_tn import *
from const_v import *
import numpy as np
from scipy.optimize import minimize

# Afficher chaque transformation pour suivre le calcul
for i in range(len(dh['sigma_i'])):
    print(f"Transformation T({i},{i + 1}):\n")
    t_i_ip1 = matrice_Tim1_Ti(dh["sigma_i"][i], dh["a_i_m1"][i], dh["alpha_i_m1"][i], dh["r_i"][i])
    if round_p:
        # Arrondi
        t_i_ip1_rounded = np.round(t_i_ip1, round_p[0])

        # Si tres petit = 0
        t_i_ip1_rounded[np.abs(t_i_ip1) < round_p[1]] = 0
        print(f"{t_i_ip1_rounded}\n")

# Calcul de la transformation complète T(0,3)
print(f"Transformation T(0,{len(dh['sigma_i'])}) :\n")
#Matrice avec des parametres sans arrondir
matrice_T0Tn = matrice_Tn(dh)
print(f"{matrice_T0Tn}\n")
if round_p:
    # Arrondi
    matrice_T0Tn_rounded = np.round(matrice_T0Tn, round_p[0])

    # Si tres petit = 0
    matrice_T0Tn_rounded[np.abs(matrice_T0Tn) < round_p[1]] = 0
    print(f"{matrice_T0Tn_rounded}\n")

# Pour ce TP Z0 représente l'axe vertical et Y0 celui de la profondeur
print("\nCoordonnées finales grace a matrice T(0,n) en fonction de X0,Y0,Z0:\n", xy_Ot(matrice_T0Tn))
print("\nCoordonnées (x, y, z) en mm en fonction des angles de la liste q et X0, Y0, Z0:")
Xd_mgd = mgd(q, Lxz, p1, inclin_horiz)
x_mgd = Xd_mgd[0]
y_mgd = Xd_mgd[1]
z_mgd = Xd_mgd[2]
ray = round(np.sqrt(x_mgd ** 2 + y_mgd ** 2 + z_mgd ** 2), 2)
if ray <= rayon_max1_5 and z_mgd >= 0 + 5:
    print("Valeurs de q correctes, coordonnés finales (x,y,z): \n", Xd_mgd)
else:
    print("Valeurs de q incorrectes, dépasement du rayon de 1600 mm (rayon actuel= ", ray,
          "mm) ou valeur de z négative, coordonnés finales (x,y,z): \n", Xd_mgd)

# Tester plusieurs valeurs du pas
valeurs_pas = [1.5, 2, 2.5, 3,4,5]
q_initial=q
print("Configuracion initiale calculée:", q_initial)
resultats = evaluer_plusieurs_pas(Xd, q_initial, Lxz, p1, inclin_horiz, valeurs_pas, Nb_iter)
# Courbe(resultats, "Évolution de l'erreur en fonction du pas", "Itération", "Erreur")


"""TERMINER CORRECTEMENT CETTE DERNIERE PARTIE, VALEURS DU MGI INCORRECTES"""
# Execution du MGI (METHODE CLASSIQUE)
normal_angle = np.vectorize(lambda angle: angle % 360)
q_final, historique_erreur = mgi(Xd, q_initial, Lxz, p1, inclin_horiz, Nb_iter)

q_final_deg = normal_angle(np.degrees(q_final))
error_final = historique_erreur[-1]
print("Coordonnées con souhaite atteindre:", Xd)
print("Configuration finale des articulations:", q_final_deg)
print("Avec une erreur finale de:", error_final)
print("Verification de ces angles avec la fonction MGD:", mgd(q_final, Lxz, p1, inclin_horiz))


# Execution du MGI (METHODE OPTIMIZE)
Xe = np.array([700, 200, 500])

# Valeurs initiales pour qi (exemple)
qi_initiale = [0, 0, 0]  # À ajuster en fonction du robot

# Bornes pour chaque angle
bounds = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]

# Optimisation pour trouver le MGI
result = minimize(fonction_cout, x0=qi_initiale, args=(Xe, dh), bounds=bounds)

# Résultats
if result.success:
    qi_solution = result.x
    print("\nCoordonnées con souhaite atteindre:", Xe)
    print("Solution pour les angles qi :", qi_solution)
    print("Verification de ces angles avec la fonction MGD:", mgd(qi_solution, Lxz, p1, inclin_horiz))
else:
    print("La solution n'a pas convergé.")