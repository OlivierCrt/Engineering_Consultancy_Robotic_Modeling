import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from matrice_tn import *
from const_v import *
from modele_differentiel import *

def traj(A, B, V1, V2, Debug=False):
    """
    Génère une trajectoire circulaire dans R^3 entre deux points A et B.
    Args:
        A (np.ndarray): Point de départ [x, y, z].
        B (np.ndarray): Point d'arrivée [x, y, z].
        V1 (float): Vitesse initiale.
        V2 (float): Vitesse finale.
        Debug (bool): Affiche les détails pour le débogage.
    Returns:
        tuple: (q, qp, positions) Trajectoires articulaires, vitesses et positions opérationnelles.
    """
    try:
        K = float(input("Quelle valeur d'accélération (K) voulez-vous appliquer ?\n"))
        if K <= 0:
            raise ValueError("L'accélération K doit être positive.")
    except ValueError as e:
        print(f"Erreur : {e}")
        return None

    # Vecteur directeur AB et point médian
    AB = B - A
    center = (A + B) / 2
    ray = np.linalg.norm(AB) / 2

    arbitrary_vector = np.array([1, 0, 0]) if AB[2] != 0 else np.array([0, 0, 1])
    normal = np.cross(AB, arbitrary_vector).astype(float)
    normal /= np.linalg.norm(normal)  # Normalisation

    # Base locale du plan
    u = AB / np.linalg.norm(AB)
    v = np.cross(normal, u)

    # Temps de transition
    t1 = V1 / K
    t2 = t1 + (np.pi * ray - (V1**2 / (2 * K))) / V1
    t3 = t2 + (V2 - V1) / K
    t4 = t3 + (np.pi * ray - (V2**2 / (2 * K))) / V2
    tf = t4 + V2 / K

    # Génération du temps
    time = np.linspace(0, tf, 1000)
    vitesse = np.piecewise(
        time,
        [time < t1, (time >= t1) & (time < t2), (time >= t2) & (time < t3), (time >= t3) & (time < t4), time >= t4],
        [lambda t: K * t,
         lambda t: V1,
         lambda t: V1 + K * (t - t2),
         lambda t: V2,
         lambda t: V2 - K * (t - t4)]
    )

    acceleration = np.piecewise(
    time,
    [time < t1, (time >= t1) & (time < t2), (time >= t2) & (time < t3), (time >= t3) & (time < t4), time >= t4],
    [K,  # Accélération
     0,  # Vitesse constante
     K,  # Accélération
     0,  # Vitesse constante
     -K]  # Décélération
)
    #print(f"ACCEL = {acceleration}")
    s = np.cumsum(vitesse * (time[1] - time[0]))

    theta = s / ray
    positions = np.array([
        center[0] + ray * (np.cos(theta) * u[0] + np.sin(theta) * v[0]),
        center[1] + ray * (np.cos(theta) * u[1] + np.sin(theta) * v[1]),
        center[2] + ray * (np.cos(theta) * u[2] + np.sin(theta) * v[2]),
    ]).T

    # Conversion articulaire et vitesses
    q, qp, xp, yp, zp = [], [], [], [], []


    # Calcul des vitesses opérationnelles
    delta_t = time[1:] - time[:-1]
    velocities = (positions[1:, :] - positions[:-1, :]) / delta_t[:, np.newaxis]

    # Interpolation pour aligner les vitesses avec les positions
    interp_vel_x = interp1d(time[:-1], velocities[:, 0], kind='linear', fill_value="extrapolate")
    interp_vel_y = interp1d(time[:-1], velocities[:, 1], kind='linear', fill_value="extrapolate")
    interp_vel_z = interp1d(time[:-1], velocities[:, 2], kind='linear', fill_value="extrapolate")

    xp = interp_vel_x(time)
    yp = interp_vel_y(time)
    zp = interp_vel_z(time)

    # Calcul des accélérations opérationnelles
    try:
        acc_x = np.gradient(xp, time)
        acc_y = np.gradient(yp, time)
        acc_z = np.gradient(zp, time)
    except Exception as e:
        print(f"Erreur lors du calcul des gradients : {e}")
        acc_x, acc_y, acc_z = np.zeros_like(xp), np.zeros_like(yp), np.zeros_like(zp)

    # Vérifiez que acc_x, acc_y, acc_z ont la même taille que time
    if len(acc_x) != len(time):
        print(f"Dimensions mismatch detected: acc_x({len(acc_x)}) != time({len(time)})")
        time_adjusted = np.linspace(0, tf, len(acc_x))
    else:
        time_adjusted = time

    # Interpolation pour aligner les accélérations avec le vecteur temps original
    interp_acc_x = interp1d(time_adjusted, acc_x, kind='linear', fill_value="extrapolate")
    interp_acc_y = interp1d(time_adjusted, acc_y, kind='linear', fill_value="extrapolate")
    interp_acc_z = interp1d(time_adjusted, acc_z, kind='linear', fill_value="extrapolate")

    # Appliquer l'interpolation pour xpp, ypp et zpp
    xpp = interp_acc_x(time)
    ypp = interp_acc_y(time)
    zpp = interp_acc_z(time)


    for i, X in enumerate(positions):
        solutions = mgi(X, Liaisons)
        if solutions:
            q_i = solutions[0]
            q.append(q_i)

            # Calcul de la matrice Jacobienne pour la configuration courante
            T_matrices = t_mat(q_i, dh)
            J = Jacob_geo(T_matrices)
            J_translation = J[:3, :]  # Jacobienne translationnelle

            # Calcul des vitesses opérationnelles et articulaires
            v_operational = np.array([xp[i], yp[i], zp[i]])
            qp_i = MDI(v_operational, J_translation)  # Vitesses articulaires
            qp.append(qp_i)

            # Calcul des accélérations opérationnelles
            acc_operational = np.array([xpp[i], ypp[i], zpp[i]])
            #print(acc_operational)



        else:
            print(f"Erreur : MGI échoué pour X={X}")
            q.append(None)
            qp.append(None)

    # Convertir les listes en tableaux numpy
    q = np.array(q)
    qp = np.array(qp)



    if Debug:
        # Trajectoire 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label="Trajectoire opérationnelle", color='b')
        ax.scatter(A[0], A[1], A[2], color='g', label="Point A (Départ)")
        ax.scatter(B[0], B[1], B[2], color='r', label="Point B (Arrivée)")
        ax.set_title("Trajectoire 3D")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

        # Lois de mouvement temporel
        plt.figure()
        plt.plot(time, s, label="s(t)")
        plt.plot(time, vitesse, label="s'(t)")
        plt.plot(time, acceleration, label="s''(t)", color='r')
        for t_transition, label in zip([t1, t2, t3, t4], ['t1', 't2', 't3', 't4']):
            plt.axvline(x=t_transition, color='r', linestyle='--', label=label)
        plt.title("Lois de mouvement temporel")
        plt.xlabel("Temps (s)")
        plt.ylabel("Valeur")
        plt.legend()
        plt.grid()

        # Trajectoires opérationnelles
        plt.figure()
        plt.plot(time, positions[:, 0], label="x(t)")
        plt.plot(time, positions[:, 1], label="y(t)")
        plt.plot(time, positions[:, 2], label="z(t)")
        for t_transition, label in zip([t1, t2, t3, t4], ['t1', 't2', 't3', 't4']):
            plt.axvline(x=t_transition, color='r', linestyle='--', label=label)
        plt.title("Trajectoire opérationnelle")
        plt.xlabel("Temps (s)")
        plt.ylabel("Coordonnées")
        plt.legend()
        plt.grid()

        # Vitesses opérationnelles
        plt.figure()
        plt.plot(time, xp, label="x'(t)")
        plt.plot(time, yp, label="y'(t)")
        plt.plot(time, zp, label="z'(t)")
        for t_transition, label in zip([t1, t2, t3, t4], ['t1', 't2', 't3', 't4']):
            plt.axvline(x=t_transition, color='r', linestyle='--', label=label)
        plt.title("Vitesses opérationnelles")
        plt.xlabel("Temps (s)")
        plt.ylabel("Vitesses")
        plt.legend()
        plt.grid()

            # Accélérations opérationnelles
        plt.figure()
        plt.plot(time, xpp, label="x''(t)")
        plt.plot(time, ypp, label="y''(t)")
        plt.plot(time, zpp, label="z''(t)")
        for t_transition, label in zip([t1, t2, t3, t4], ['t1', 't2', 't3', 't4']):
            plt.axvline(x=t_transition, color='r', linestyle='--', label=label)
        plt.title("Accélérations opérationnelles")
        plt.xlabel("Temps (s)")
        plt.ylabel("Accélération (mm/s²)")
        plt.legend()
        plt.grid()

        # Profils articulaires
        plt.figure()
        plt.plot(time, q[:, 0], label="q1(t)")
        plt.plot(time, q[:, 1], label="q2(t)")
        plt.plot(time, q[:, 2], label="q3(t)")
        for t_transition, label in zip([t1, t2, t3, t4], ['t1', 't2', 't3', 't4']):
            plt.axvline(x=t_transition, color='r', linestyle='--', label=label)
        plt.title("Trajectoires articulaires")
        plt.xlabel("Temps (s)")
        plt.ylabel("Angles (°)")
        plt.legend()
        plt.grid()

        # Vitesses articulaires avec marqueurs
        plt.figure()
        markers = ['o', 's', 'x', '^', 'v', '*']  # Liste de marqueurs
        for joint in range(qp.shape[1]):
            plt.plot(
                time, 
                qp[:, joint], 
                label=f"q{joint+1}'(t)", 
                marker=markers[joint % len(markers)],  # Marqueur cyclique
                markevery=50  # Ajouter des marqueurs tous les 50 points
            )
        for t_transition, label in zip([t1, t2, t3, t4], ['t1', 't2', 't3', 't4']):
            plt.axvline(x=t_transition, color='r', linestyle='--', label=label)
        plt.title("Vitesses articulaires")
        plt.xlabel("Temps (s)")
        plt.ylabel("Vitesses articulaires (rad/s)")
        plt.legend()
        plt.grid()

        plt.show()


    return q, qp


def t_mat(q, dh):
    """
    Génère les matrices de transformation homogène pour une configuration articulaire donnée.
    Args:
        q (list): Configuration articulaire.
        dh (dict): Paramètres DH.
    Returns:
        list of np.ndarray: Matrices de transformation homogène.
    """
    T_matrices = []

    for i in range(len(dh['sigma_i'])-1):
        t_i_ip1 = matrice_Tim1_Ti(q[i], dh["a_i_m1"][i], dh["alpha_i_m1"][i], dh["r_i"][i])
        T_matrices.append(t_i_ip1)
        #print(t_i_ip1)
    return T_matrices

def est_point_atteignable(point):
    """
    Vérifie si un point est atteignable par le robot en fonction de son espace opérationnel.

    Args:
        point (tuple): Coordonnées (x, y, z) du point à vérifier.

    Returns:
        bool: True si le point est atteignable, False sinon.
        str: Message expliquant la raison si le point n'est pas atteignable.
    """
    x, y, z = point

    # Extraire les longueurs des liaisons depuis le dictionnaire
    longueur_bras = sum([np.linalg.norm(liaison) for liaison in Liaisons.values()])  # Norme 3D de chaque liaison
    z_min = min([liaison[1] for liaison in Liaisons.values()])  # Hauteur minimale
    z_max = max([liaison[1] for liaison in Liaisons.values()]) + longueur_bras  # Hauteur maximale atteignable
    rayon_max = longueur_bras  # Rayon maximum atteint en projection 2D (xy)

    # Calcul de la distance en projection 2D
    rayon_xy = np.sqrt(x**2 + y**2)

    # Vérifier les contraintes
    if z < z_min or z > z_max:
        return False, f"Le point est hors des limites verticales : {z_min} <= z <= {z_max}."
    if rayon_xy > rayon_max:
        return False, f"Le point est hors du rayon maximum atteignable dans le plan XY : r <= {rayon_max}."
    
    # Vérifier avec MGI
    solutions = mgi(np.array([x, y, z]), Liaisons)
    if not solutions:
        return False, "Le MGI n'a trouvé aucune solution pour atteindre ce point."

    return True, "Le point est atteignable."