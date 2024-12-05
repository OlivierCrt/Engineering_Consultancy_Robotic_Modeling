import numpy as np
import matplotlib.pyplot as plt
from matrice_tn import *
from const_v import Liaisons


def traj(A, B, V1, V2, Debug=False):
    # Demander la valeur de l'accélération constante K
    K = float(input("Quelle valeur d'accélération (K) voulez-vous appliquer ?\n"))

    # Définir le cercle à partir des points A et B
    diam = np.linalg.norm(B - A)  # Diamètre
    ray = diam / 2  # Rayon
    center = (A + B) / 2  # Centre du cercle

    # Temps de transition
    t1 = V1 / K
    t2 = t1 + (np.pi * ray - (V1**2 / (2 * K))) / V1
    t3 = t2 + (V2 - V1) / K
    t4 = t3 + (np.pi * ray - (V2**2 / (2 * K))) / V2
    tf = t4 + V2 / K

    if Debug:
        print(f"t1 = {t1}, t2 = {t2}, t3 = {t3}, t4 = {t4}, tf = {tf}")

    # Temps échantillonné
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

    # Calcul de l'abscisse curviligne s(t)
    s = np.cumsum(vitesse * (time[1] - time[0]))  # Intégrale de la vitesse

    # Positions sur le cercle (X(s), Y(s), Z(s))
    theta = 2 * np.pi * s / (2 * np.pi * ray)  # Relation entre s et l'angle paramétrique
    positions = np.array([
        center[0] + ray * np.cos(theta),
        center[1] + ray * np.sin(theta),
        np.zeros_like(theta)  # Plaque verticale
    ]).T

    # Conversion en espace articulaire
    q, qp, qpp = [], [], []

    for pos in positions:
        solutions = mgi(pos, Liaisons)  # Utiliser le MGI
        if solutions:
            q.append(solutions[0])  # Choisir une solution
        else:
            print(f"Erreur : aucune solution MGI pour la position {pos}.")
            return None

    q = np.array(q)
    dt = time[1] - time[0]
    qp = np.gradient(q, axis=0, edge_order=2) / dt
    qpp = np.gradient(qp, axis=0, edge_order=2) / dt
    # Calcul du jerk articulaire
    jerk = np.gradient(qpp, axis=0, edge_order=2) / dt  # Dérivée de l'accélération


    if Debug:
        # Tracer les profils de vitesse, d'accélération et les trajectoires
        plt.figure()
        plt.plot(time, vitesse, label="Vitesse")
        plt.plot(time, acceleration, label="Accélération")
        plt.title("Profils de vitesse et d'accélération")
        plt.xlabel("Temps (s)")
        plt.ylabel("Valeur")
        plt.legend()
        plt.grid()

        plt.figure()
        plt.plot(time, q[:, 0], label="q1")
        plt.plot(time, q[:, 1], label="q2")
        plt.plot(time, q[:, 2], label="q3")
        plt.title("Trajectoires articulaires")
        plt.xlabel("Temps (s)")
        plt.ylabel("Angles (°)")
        plt.legend()
        plt.grid()

                # Tracer le jerk articulaire
        plt.figure()
        plt.plot(time, jerk[:, 0], label="jerk q1")
        plt.plot(time, jerk[:, 1], label="jerk q2")
        plt.plot(time, jerk[:, 2], label="jerk q3")
        plt.title("Jerk articulaire")
        plt.xlabel("Temps (s)")
        plt.ylabel("Jerk (°/s³)")
        plt.legend()
        plt.grid()
        plt.show()

        plt.show()

    return q, qp, qpp