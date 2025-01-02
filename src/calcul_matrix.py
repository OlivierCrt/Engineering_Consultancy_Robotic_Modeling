import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Definir los símbolos para cosenos y senos
c1, s1, c2, s2, c3, s3, c4, s4 = sp.symbols('c1 s1 c2 s2 c3 s3 c4 s4')

# Definir las matrices simbólicas
matrix01 = sp.Matrix([
    [c1, -s1, 0, 0],
    [s1, c1,  0, 0],
    [0,   0,  1, 550],
    [0,   0,  0, 1]
])

matrix12 = sp.Matrix([
    [c2, -s2, 0, 150],
    [0, 0, -1, 0],
    [s2, c2, 0, 0],
    [0,   0, 0, 1]
])

matrix23 = sp.Matrix([
    [c3,  -s3, 0, 825],
    [s3,  c3, 0, 0],
    [0,   0, 1, 0],
    [0,   0, 0, 1]
])

# Calcular matrices de transformación acumuladas
T_matrices = [matrix01, matrix01 * matrix12, matrix01 * matrix12 * matrix23]
matrix02=matrix01 * matrix12
matrix03=matrix01 * matrix12 * matrix23
print("\n",matrix02)
print("\n",matrix03,"\n")

# Función para calcular el Jacobiano
def calculer_jacobien_sympy(T_matrices, types_articulations):
    """
    Calcule le Jacobien symbolique d'un manipulateur à partir d'une liste de matrices de transformation 4x4.

    Paramètres:
    T_matrices : list of sp.Matrix
        Liste des matrices de transformation homogène 4x4 de chaque articulation jusqu'à l'effecteur final.
    types_articulations : list of int
        Liste des types d'articulations (0 pour rotoïde, 1 pour prismatique).

    Retourne:
    sp.Matrix
        Jacobien 6xN où N est le nombre d'articulations.
    """
    n = len(T_matrices)  # Nombre d'articulations

    # Position de l'effecteur final
    p_e = T_matrices[-1][:3, 3]  # Position de l'effecteur final (les trois premières lignes de la dernière colonne)
    J_P = []  # Partie de translation du Jacobien
    J_O = []  # Partie de rotation du Jacobien

    # Calcul des colonnes du Jacobien pour chaque articulation
    for i in range(n):
        T_i = T_matrices[i]  # Matrice de transformation pour l'articulation i
        p_i = T_i[:3, 3]  # Position de l'articulation i
        z_i_1 = T_i[:3, 2]  # Axe z de la rotation pour l'articulation i (troisième colonne)

        if types_articulations[i] == 0:  # Articulation rotoïde
            J_P_i = sp.Matrix.cross(z_i_1, p_e - p_i)
            J_O_i = z_i_1
        else:  # Articulation prismatique
            J_P_i = z_i_1
            J_O_i = sp.Matrix([0, 0, 0])

        J_P.append(J_P_i)
        J_O.append(J_O_i)

    # Combiner les parties translationnelle et rotationnelle
    J_P = sp.Matrix.hstack(*J_P)  # Transformer en une matrice 3xN
    J_O = sp.Matrix.hstack(*J_O)  # Transformer en une matrice 3xN

    J = sp.Matrix.vstack(J_P, J_O)  # Combiner en une matrice 6xN

    # Validar dimensiones
    if J.shape != (6, n):
        raise ValueError(f"Le Jacobien a une forme incorrecte : {J.shape}, attendu (6, {n})")

    return J

# Calcular el Jacobiano
types_articulations = [0, 0, 0]  # Todas rotoïdes
jacobien = calculer_jacobien_sympy(T_matrices, types_articulations)

# Mostrar el resultado
# sp.pprint(jacobien)

"""Calcul dq1, dq2, dq3"""
def MDI_analytique(vel):
    angle_range = np.linspace(0, 2 * np.pi, 300)  # Ángulos de 0 a 2π
    dq1_values = []
    dq2_values = []
    dq3_values = []
    q1_values = []
    q2_values = []

    for q1 in angle_range:
        for q2 in angle_range:
            dq1 = vel[5]
            if np.isclose(np.cos(q2), 0, atol=1e-6):  # Evitar división por cero
                dq2 = np.nan
            else:
                dq2 = vel[2] / (825 * np.cos(q2))
            if np.isclose(np.sin(q1), 0, atol=1e-6):  # Evitar división por cero
                dq3 = np.nan
            else:
                dq3 = (vel[3] - np.sin(q1) * dq2) / np.sin(q1)

            dq1_values.append(dq1)
            dq2_values.append(dq2)
            dq3_values.append(dq3)
            q1_values.append(q1)
            q2_values.append(q2)

    # Crear gráficos
    plt.figure(figsize=(15, 10))

    # Graficar dq1
    plt.subplot(3, 1, 1)
    plt.scatter(q1_values, dq1_values, c='blue', s=10, alpha=0.7, label='dq1')
    plt.xlabel('q1 (radianes)')
    plt.ylabel('dq1')
    plt.title('Evolución de dq1 respecto a q1')
    plt.grid()
    plt.legend()

    # Graficar dq2
    plt.subplot(3, 1, 2)
    plt.scatter(q2_values, dq2_values, c='green', s=10, alpha=0.7, label='dq2')
    plt.xlabel('q2 (radianes)')
    plt.ylabel('dq2')
    plt.title('Evolución de dq2 respecto a q2')
    plt.grid()
    plt.legend()

    # Graficar dq3
    plt.subplot(3, 1, 3)
    plt.scatter(q1_values, dq3_values, c='red', s=10, alpha=0.7, label='dq3')
    plt.xlabel('q1 (radianes)')
    plt.ylabel('dq3')
    plt.title('Evolución de dq3 respecto a q1')
    plt.grid()
    plt.legend()

    # Mostrar gráficos
    plt.tight_layout()
    plt.show()

# Valores de velocidad para la simulación
vel = [0, 0, 10, 15, 0, 5]  # Ejemplo de velocidades

# Calcular valores
MDI_analytique(vel)



