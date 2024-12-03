import numpy as np
import matplotlib.pyplot as plt

def traj(A,B,V1,V2, Debug = False):
    # Demander la valeur de l'accélération constante K
    K = float(input("Quelle valeur d'accélération (K) voulez-vous appliquer ?\n"))



    diam = np.linalg.norm(B - A)
    ray = diam/2

    # Temps des transitions

    t0 = 0
    t1 = V1/K
    t2 = np.pi*ray - (((t1*V1/2) + (((V2-V1)/K)*(V1+ V2/2)))/V1) + t1
    t3 = t2 + (V2-V1)/K
    t4 = np.pi - (V2/K)/2 + t3
    tf = (V2/K) + t4

    if Debug :
        print(f"{t1},{t2},{t3},{t4},{tf}")
        
    # Temps échantillonné
    time = np.linspace(t0, tf, 1000)

    # Profils de vitesse et d'accélération
    vitesse = np.piecewise(
        time,
        [time < t1, (time >= t1) & (time < t2), (time >= t2) & (time < t3), (time >= t3) & (time < t4), time >= t4],
        [lambda t: K * t,  # Phase d'accélération (0 -> t1)
        lambda t: V1,     # Vitesse constante (t1 -> t2)
        lambda t: V1 + K * (t - t2),  # Accélération (t2 -> t3)
        lambda t: V2,     # Vitesse constante (t3 -> t4)
        lambda t: V2 - K * (t - t4)]  # Décélération (t4 -> tf)
    )

    acceleration = np.piecewise(
        time,
        [time < t1, (time >= t1) & (time < t2), (time >= t2) & (time < t3), (time >= t3) & (time < t4), time >= t4],
        [K,  # Accélération constante (0 -> t1)
        0,  # Vitesse constante (t1 -> t2)
        K,  # Accélération constante (t2 -> t3)
        0,  # Vitesse constante (t3 -> t4)
        -K]  # Décélération constante (t4 -> tf)
    )
    if Debug:
        # Tracer les graphiques
        plt.figure()
        plt.plot(time, vitesse, label="Vitesse", lw=2)
        plt.title("Profil de vitesse")
        plt.xlabel("Temps (s)")
        plt.ylabel("Vitesse (m/s)")
        plt.grid()
        plt.legend()

        plt.figure()
        plt.plot(time, acceleration, label="Accélération", lw=2, color="orange")
        plt.title("Profil d'accélération")
        plt.xlabel("Temps (s)")
        plt.ylabel("Accélération (m/s²)")
        plt.grid()
        plt.legend()

        plt.show()

        q = 0
        qp = 0
        qpp = 0

    return 0

V1 = 10  # Vitesse 1 (par exemple)
V2 = 20  # Vitesse 2 (par exemple)

A = np.array([5, 5, 0])  # (par exemple)
B = np.array([10, 20, 0])  # (par exemple)

a = traj(A,B,V1,V2,Debug=True)
