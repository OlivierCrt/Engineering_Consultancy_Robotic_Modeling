import numpy as np
import sympy as sp
import time

def calculate_z_and_o(T):
    """
    Extrae el vector z y la posición o de una matriz de transformación homogénea.
    """
    z = T[:3, 2]  # Tercera columna (vectores z)
    o = T[:3, 3]  # Cuarta columna (posición)
    return z, o

def Jacob_geo(matrices):
    """
    Calcula Jp1, Jp2, Jp3, z0, z1, y z2 a partir de las matrices de transformación.
    """
    # Calcular matrices acumuladas
    T_01 = matrices[0]
    T_02 = np.dot(T_01, matrices[1])
    T_03 = np.dot(T_02, matrices[2])
    T_0T = np.dot(T_03, matrices[3])

    # Obtener vectores z y o
    z0, o0 = calculate_z_and_o(T_01)
    z1, o1 = calculate_z_and_o(T_02)
    z2, o2 = calculate_z_and_o(T_03)
    _, ot = calculate_z_and_o(T_0T)

    # Calcular Jp
    jp1 = np.cross(z0, ot - o0)
    jp2 = np.cross(z1, ot - o1)
    jp3 = np.cross(z2, ot - o2)

    J = np.array([
        [jp1[0], jp2[0], jp3[0]],
        [jp1[1], jp2[1], jp3[1]],
        [jp1[2], jp2[2], jp3[2]],
        [z0[0], z1[0], z2[0]],
        [z0[1], z1[1], z2[1]],
        [z0[2], z1[2], z2[2]],
    ])

    return J

def Mat_T_analytiques():
    """
    Define las matrices simbólicas con placeholders para cosenos y senos.
    """
    c1, s1, c2, s2, c3, s3, c4, s4 = sp.symbols('c1 s1 c2 s2 c3 s3 c4 s4')

    T01 = sp.Matrix([
        [c1, -s1, 0, 0],
        [s1,  c1, 0, 0],
        [ 0,   0, 1, 550],
        [ 0,   0, 0,   1]
    ])

    T12 = sp.Matrix([
        [c2, -s2,  0, 150],
        [ 0,   0, -1,   0],
        [s2,  c2,  0,   0],
        [ 0,   0,  0,   1]
    ])

    T23 = sp.Matrix([
        [c3, -s3, 0, 825],
        [s3,  c3, 0,   0],
        [ 0,   0, 1,   0],
        [ 0,   0, 0,   1]
    ])

    T34 = sp.Matrix([
        [c4,  0, 0, 735],
        [ 0, c4, 0,   0],
        [ 0,  0, 1,   0],
        [ 0,  0, 0,   1]
    ])

    return T01, T12, T23, T34

def Jacob_analytique(M):
    """
    Calcula la matriz Jacobiana simbólica.
    """
    c1, s1, c2, s2, c3, s3, c4, s4 = sp.symbols('c1 s1 c2 s2 c3 s3 c4 s4')
    T01, T12, T23, T34 = M

    # Calcular matrices acumuladas
    T_01 = T01
    T_02 = T01 * T12
    T_03 = T_02 * T23
    T_0T = T_03 * T34

    # Obtener vectores z y O
    z0 = T_01[:3, 2]
    z1 = T_02[:3, 2]
    z2 = T_03[:3, 2]
    o0 = T_01[:3, 3]
    o1 = T_02[:3, 3]
    o2 = T_03[:3, 3]
    ot = T_0T[:3, 3]

    # Calcular Jp
    Jp1 = z0.cross(ot - o0)
    Jp2 = z1.cross(ot - o1)
    Jp2_simp = sp.Matrix([-c1 * (735 * (c2 * s3 + c3 * s2) + 825 * s2), -s1 * (735 * (c2 * s3 + c3 * s2) + 825 * s2),
                          735 * (c2 * c3 - s2 * s3) + 825 * c2])
    Jp3 = z2.cross(ot - o2)
    Jp3_simp = sp.Matrix([-735 * c1 * (c2 * s3 + c3 * s2), -735 * s1 * (c2 * s3 + c3 * s2), 735 * (c2 * c3 + s2 * s3)])

    # Construir Jacobiano
    J = sp.Matrix([
        [Jp1[0], Jp2_simp[0], Jp3_simp[0]],
        [Jp1[1], Jp2_simp[1], Jp3_simp[1]],
        [Jp1[2], Jp2_simp[2], Jp3_simp[2]],
        [z0[0], z1[0], z2[0]],
        [z0[1], z1[1], z2[1]],
        [z0[2], z1[2], z2[2]],
    ])

    return J

def MDD(v,J) :
    """
    Return vitesses OT
    parametre Vitesses articulaires, J jacobienne

    """
    return np.dot(J,v)


def MDI(x,J):
    """return q vitesse
    param : x vitesse de l OT souhaitée , J jacobienne"""
    return np.dot(np.linalg.pinv(J),x)



