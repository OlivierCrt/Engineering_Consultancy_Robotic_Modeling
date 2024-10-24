import numpy as np


def matrice_Tim1_Ti(qi, ai_m1, alphai_m1, ri, decimals=2, threshold=1e-10):
    """
    valeur unique qi
    valeur unique ai-1
    valeur unique alphai-1
    valeur unique ri
    return matrice Ti-1,i avec les éléments arrondis
    """
    matrix_res = np.zeros((4, 4))

    matrix_res[0, 0] = np.cos(qi)
    matrix_res[0, 1] = -np.sin(qi)
    matrix_res[0, 2] = 0
    matrix_res[0, 3] = ai_m1

    matrix_res[1, 0] = np.sin(qi) * np.cos(alphai_m1)
    matrix_res[1, 1] = np.cos(qi) * np.cos(alphai_m1)
    matrix_res[1, 2] = -np.sin(alphai_m1)
    matrix_res[1, 3] = -ri * np.sin(alphai_m1)

    matrix_res[2, 0] = np.sin(qi) * np.sin(alphai_m1)
    matrix_res[2, 1] = np.cos(qi) * np.sin(alphai_m1)
    matrix_res[2, 2] = np.cos(alphai_m1)
    matrix_res[2, 3] = ri * np.cos(alphai_m1)

    matrix_res[3, 0] = 0
    matrix_res[3, 1] = 0
    matrix_res[3, 2] = 0
    matrix_res[3, 3] = 1

    # Redondear los valores a 'decimals' dígitos
    matrix_res = np.round(matrix_res, decimals)

    # Reemplazar valores pequeños (por debajo de 'threshold') por 0
    matrix_res[np.abs(matrix_res) < threshold] = 0

    return matrix_res


def matrice_Tn(qi, alphai_moins1, ri, ai_moins1, decimals=2, threshold=1e-3):
    """
    qi: liste des qi
    alphai-1 liste des alpha
    ri liste des r
    ai-1 liste
    return matrice T0,n correspondante avec les éléments arrondis.
    """
<<<<<<< HEAD
    nbliaison = len(qi)
    mat_list = []
    for i in range(nbliaison):
        mat_temp = matrice_Tim1_Ti(qi[i], ai_moins1[i], alphai_moins1[i], ri[i])
        mat_list.append(mat_temp)

    result_matrix = np.eye(4)
    for mat in mat_list:
        result_matrix = np.dot(result_matrix, mat)

    # Redondear los valores a 'decimals' dígitos
    result_matrix = np.round(result_matrix, decimals)

    # Reemplazar valores pequeños (por debajo de 'threshold') por 0
    result_matrix[np.abs(result_matrix) < threshold] = 0

    return result_matrix


def xy_Ot(qi, L):
    """qi list
        longueur list"""

    x = L[0] * np.cos(qi[0]) + L[1] * np.cos(qi[0] + qi[1]) + L[2] * np.cos(qi[0] + qi[1] + qi[2])
    y = L[0] * np.sin(qi[0]) + L[1] * np.sin(qi[0] + qi[1]) + L[2] * np.sin(qi[0] + qi[1] + qi[2])
    return x, y



=======


    nbliaison=len(qi)
    mat_list=[]
    for i in range (nbliaison):
        mat_temp=matrice_Tim1_Ti(qi[i],ai_moins1[i],alphai_moins1[i],ri[i])
        mat_list.append(mat_temp) 
    result_matrix = np.eye(4)
    for mat in mat_list:
        result_matrix = np.dot(result_matrix,mat)
    
    return result_matrix


def xy_Ot(qi,L):
    """qi list
        longueur list"""
    
    x=L[0]*np.cos(qi[0]) + L[1]*np.cos(qi[0]+qi[1]) + L[2]*np.cos(qi[0]+qi[1]+qi[2])
    y=L[0]*np.sin(qi[0]) + L[1]*np.sin(qi[0]+qi[1]) + L[2]*np.sin(qi[0]+qi[1]+qi[2])
    return x,y
    
   
 
>>>>>>> 319975e397d9aa9b4a37ca114d824bb449eb2492
