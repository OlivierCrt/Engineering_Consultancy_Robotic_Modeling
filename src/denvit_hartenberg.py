import numpy as np


def matrice_Tim1_Ti(qi, ai_m1, alphai_m1, ri):
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

    return matrix_res

    

def matrice_Tn(qi,alphai_moins1,ri,ai_moins1) :
    """
    qi: liste des qi
    alphai-1 liste des alpha
    ri liste des r
    ai-1 liste
    """
    nbliaison=len(qi)
    mat_list=[]
    for i in range (nbliaison):
        mat_temp=matrice_Tim1_Ti(qi[i],alphai_moins1[i],ri[i],ai_moins1[i])
        mat_list.append(mat_temp)  
    result_matrix = np.eye(4)
    for mat in mat_list:
        result_matrix = np.dot(result_matrix, mat)
    return result_matrix
   




        

    return 
