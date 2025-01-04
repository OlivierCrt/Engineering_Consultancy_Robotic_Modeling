import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Définir les symboles pour les cosinus et sinus
c1, s1, c2, s2, c3, s3, c4, s4 = sp.symbols('c1 s1 c2 s2 c3 s3 c4 s4')

# Définir les matrices symboliques
matrix01 = sp.Matrix([
    [c1, -s1, 0, 0],
    [s1, c1, 0, 0],
    [0, 0, 1, 550],
    [0, 0, 0, 1]
])

matrix12 = sp.Matrix([
    [c2, -s2, 0, 150],
    [0, 0, -1, 0],
    [s2, c2, 0, 0],
    [0, 0, 0, 1]
])

matrix23 = sp.Matrix([
    [c3, -s3, 0, 825],
    [s3, c3, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

matrix3T = sp.Matrix([
    [c4, 0, 0, 735],
    [0, c4, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# Calculer les matrices accumulées
T_01 = matrix01
T_02 = matrix01 * matrix12
T_03 = T_02 * matrix23
T_0T = T_03 * matrix3T

# Obtenir z2, OT et O3
z0 = T_01[:3, 2]  # Troisième colonne de T_01 (3 premières lignes)
z1 = T_02[:3, 2]
z2 = T_03[:3, 2]
O1 = T_01[:3, 3]  # Quatrième colonne de T_01 (3 premières lignes)
O2 = T_02[:3, 3]
O3 = T_03[:3, 3]
OT = T_0T[:3, 3]

# Calculer Jp3 = z2 x (OT - O3)
Jp1 = z0.cross(OT - O1)
Jp2 = z1.cross(OT - O2)
Jp2_simp = sp.Matrix([-c1 * (735 * (c2 * s3 + c3 * s2) + 825 * s2), -s1 * (735 * (c2 * s3 + c3 * s2) + 825 * s2),
                      735 * (c2 * c3 - s2 * s3) + 825 * c2])
Jp3 = z2.cross(OT - O3)
Jp3_simp = sp.Matrix([-735 * c1 * (c2 * s3 + c3 * s2), -735 * s1 * (c2 * s3 + c3 * s2), 735 * (c2 * c3 + s2 * s3)])

# Afficher le résultat
print("\nOT:")
sp.pprint(OT)
print("\nO1:")
sp.pprint(O1)
print("\nO2:")
sp.pprint(O2)
print("\nO3:")
sp.pprint(O3)
print("\nO1OT:")
sp.pprint(OT - O1)
print("\nO2OT:")
sp.pprint(OT - O2)
print("\nO3OT:")
sp.pprint(OT - O3)
print("\nz0:")
sp.pprint(z0)
print("\nz1:")
sp.pprint(z1)
print("\nz2:")
sp.pprint(z2)
print("\nJp1:")
sp.pprint(Jp1)
print("\nJp2:")
sp.pprint(Jp2)
print("\nJp2 simplifiée:")
sp.pprint(Jp2_simp)
print("\nJp3:")
sp.pprint(Jp3)
print("\nJp3 simplifiée:")
sp.pprint(Jp3_simp)

# Construire la Jacobienne
J = sp.Matrix([
    [Jp1[0], Jp2_simp[0], Jp3_simp[0]],
    [Jp1[1], Jp2_simp[1], Jp3_simp[1]],
    [Jp1[2], Jp2_simp[2], Jp3_simp[2]],
    [z0[0], z1[0], z2[0]],
    [z0[1], z1[1], z2[1]],
    [z0[2], z1[2], z2[2]],
])

# Afficher la Jacobienne
print("\nJacobienne analytique, sans remplacer les cos et sin:")
sp.pprint(J)

qifig = [0,0,0]
subs_dict = {
    c1: np.cos(qifig[0]), s1: np.sin(qifig[0]),
    c2: np.cos(qifig[1]), s2: np.sin(qifig[1]),
    c3: np.cos(qifig[2]), s3: np.sin(qifig[2]),
}

# Remplacer dans la Jacobienne
Jacobienne_numerique = J.subs(subs_dict)
print("\nJacobienne géométrique avec les angles qifig, q1=0, q2=0, q3=0:")
sp.pprint(Jacobienne_numerique)
dq1, dq2, dq3, dx, dy, dz, wx, wy, wz = sp.symbols('dq1 dq2 dq3 dx dy dz wx wy wz')

print("\nÉquations de la Jacobienne analytique :")
dx = J[0, 0] * dq1 + J[0, 1] * dq2 + J[0, 2] * dq3
dy = J[1, 0] * dq1 + J[1, 1] * dq2 + J[1, 2] * dq3
dz = J[2, 0] * dq1 + J[2, 1] * dq2 + J[2, 2] * dq3
wx = J[3, 0] * dq1 + J[3, 1] * dq2 + J[3, 2] * dq3
wy = J[4, 0] * dq1 + J[4, 1] * dq2 + J[4, 2] * dq3
wz = J[5, 0] * dq1 + J[5, 1] * dq2 + J[5, 2] * dq3

# Imprimir cada ecuación
print("dx =", dx)
print("dy =", dy)
print("dz =", dz)
print("wx =", wx)
print("wy =", wy)
print("wz =", wz)

def MDI_analytique(J, dX, q):
    """
        Calcula los valores de dq1, dq2 y dq3 para una Jacobiana, velocidades operativas y ángulos articulares dados.

        Args:
            J (np.ndarray): Matriz Jacobiana (6x3).
            dX (np.ndarray): Vector de velocidades operativas (longitud 6).
            q (list): Lista de ángulos articulares [q1, q2, q3] en radianes.

        Returns:
            np.ndarray: Vector [dq1, dq2, dq3].
        """
    q1 = q[0]

    # Manejar casos donde np.sin(q1) sea cercano a 0
    sin_q1 = np.sin(q1)
    if np.isclose(sin_q1, 0, atol=1e-6):
        sin_q1 = 1e-6  # Valor mínimo no cero para evitar división por cero

    # Manejar casos donde el denominador sea cercano a 0
    denominator = J[0][0] - 150 - J[0][2]
    if np.isclose(denominator, 0, atol=1e-6):
        denominator = 1e-6  # Valor mínimo no cero para evitar división por cero

    # Cálculo de dq
    dq1 = dX[5]  # wz
    dq2 = (dX[2] - ((J[0][2] * dX[3]) / sin_q1)) / denominator
    dq3 = (dX[3] / sin_q1) - dq2

    return np.array([dq1, dq2, dq3])
