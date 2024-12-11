import numpy as np
import matplotlib.pyplot as plt
from matrice_tn import *
from const_v import Liaisons


def traj(A, B, V1, V2, Debug=False):
    """
    Génère une trajectoire circulaire dans \( \mathbb{R}^3 \) entre deux points A et B.
    Args:
        A (np.ndarray): Point de départ [x, y, z].
        B (np.ndarray): Point d'arrivée [x, y, z].
        V1 (float): Vitesse initiale.
        V2 (float): Vitesse finale.
        Debug (bool): Affiche les détails pour le débogage.
    Returns:
        tuple: (q, qp, qpp) Trajectoires articulaires, vitesses et accélérations.
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

    # Trouver un vecteur non colinéaire à AB pour définir le plan
    arbitrary_vector = np.array([1, 0, 0]) if AB[2] != 0 else np.array([0, 0, 1])

    # Produit vectoriel pour obtenir le vecteur normal
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

    # Profils de vitesse et accélération
    vitesse = np.piecewise(
        time,
        [time < t1, (time >= t1) & (time < t2), (time >= t2) & (time < t3), (time >= t3) & (time < t4), time >= t4],
        [lambda t: K * t,  # Accélération
         lambda t: V1,     # Vitesse constante à V1
         lambda t: V1 + K * (t - t2),  # Accélération à V2
         lambda t: V2,     # Vitesse constante à V2
         lambda t: V2 - K * (t - t4)]  # Décélération
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
    
    s = np.cumsum(vitesse * (time[1] - time[0]))

    # Calcul de theta(s) en fonction de s
    theta = s / ray  # Relation entre s et l'angle paramétrique

    # Calcul des positions en fonction de s(t)
    positions = np.array([
        center[0] + ray * (np.cos(theta) * u[0] + np.sin(theta) * v[0]),
        center[1] + ray * (np.cos(theta) * u[1] + np.sin(theta) * v[1]),
        center[2] + ray * (np.cos(theta) * u[2] + np.sin(theta) * v[2]),
    ]).T



    # Conversion en espace articulaire
    q, qp, qpp = [], [], []
    for pos in positions:
        solutions = mgi(pos, Liaisons)
        if solutions:
            q.append(solutions[0])
        else:
            print(f"Erreur : aucune solution MGI pour la position {pos}.")
            return None

    q = np.array(q)
    dt = time[1] - time[0]
    qp = np.gradient(q, axis=0, edge_order=2) / dt
    qpp = np.gradient(qp, axis=0, edge_order=2) / dt

    # Calcul des vitesses et accélérations des points (opérationnel)
    xp = np.gradient(positions[:, 0], time)
    yp = np.gradient(positions[:, 1], time)
    zp = np.gradient(positions[:, 2], time)
    xpp = np.gradient(xp, time)
    ypp = np.gradient(yp, time)
    zpp = np.gradient(zp, time)

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
       
        # Affichage des lois de mouvement temporel
        plt.figure()
        #plt.plot(time, s, label="s(t)")
        plt.plot(time, vitesse, label="s'(t)")
        plt.plot(time, acceleration, label="s''(t)")
        plt.title("Lois de mouvement temporel")
        plt.xlabel("Temps (s)")
        plt.ylabel("Valeur")
        plt.legend()
        plt.grid()

        # Affichage des trajectoires opérationnelles
        plt.figure()
        plt.plot(time, positions[:, 0], label="x(t)")
        plt.plot(time, positions[:, 1], label="y(t)")
        plt.plot(time, positions[:, 2], label="z(t)")
        plt.title("Trajectoire opérationnelle")
        plt.xlabel("Temps (s)")
        plt.ylabel("Coordonnées")
        plt.legend()
        plt.grid()

        # Vitesses et accélérations opérationnelles
        plt.figure()
        plt.plot(time, xp, label="x'(t)")
        plt.plot(time, yp, label="y'(t)")
        plt.plot(time, zp, label="z'(t)")
        plt.title("Vitesses opérationnelles")
        plt.xlabel("Temps (s)")
        plt.ylabel("Vitesses")
        plt.legend()
        plt.grid()

        plt.figure()
        plt.plot(time, xpp, label="x''(t)")
        plt.plot(time, ypp, label="y''(t)")
        plt.plot(time, zpp, label="z''(t)")
        plt.title("Accélérations opérationnelles")
        plt.xlabel("Temps (s)")
        plt.ylabel("Accélérations")
        plt.legend()
        plt.grid()

        # Profils articulaires
        plt.figure()
        plt.plot(time, q[:, 0], label="q1(t)")
        plt.plot(time, q[:, 1], label="q2(t)")
        plt.plot(time, q[:, 2], label="q3(t)")
        plt.title("Trajectoires articulaires")
        plt.xlabel("Temps (s)")
        plt.ylabel("Angles (°)")
        plt.legend()
        plt.grid()

        plt.show()

    return q, qp, qpp
