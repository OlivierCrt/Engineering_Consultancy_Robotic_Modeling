import numpy as np
from src.matrice_tn import *

# Paramètres DH fictifs pour tests
dh_test = {
    "a_i_m1": [0, 1, 1],
    "alpha_i_m1": [0, np.pi / 2, 0],
    "r_i": [0.5, 0, 0]
}

# Angles pour les tests
q_test = [0, np.pi / 4, -np.pi / 4]

# Dimensions des liaisons pour MGD et MGI
Liaisons_test = {
    "Liaison 1": [1, 0.5, 0],
    "Liaison 2": [0, 1, 0],
    "Liaison 3": [0, 1, 0]
}

# Position cible pour MGI
Xd_test = [1.5, 0, 1]

# Tests de chaque fonction
def test_matrice_Tim1_Ti():
    print("Test : matrice_Tim1_Ti")
    T = matrice_Tim1_Ti(q_test[0], dh_test["a_i_m1"][0], dh_test["alpha_i_m1"][0], dh_test["r_i"][0])
    assert T.shape == (4, 4), "La matrice retournée n'est pas de taille 4x4"
    print("Résultat : OK\n")

def test_generate_transformation_matrices():
    print("Test : generate_transformation_matrices")
    T_matrices = generate_transformation_matrices(q_test, dh_test, round_p=(2, 1e-6))
    assert len(T_matrices) == len(q_test), "Le nombre de matrices ne correspond pas au nombre de liaisons"
    for T in T_matrices:
        assert T.shape == (4, 4), "Une des matrices n'est pas de taille 4x4"
    print("Résultat : OK\n")

def test_matrice_Tn():
    print("Test : matrice_Tn")
    Tn = matrice_Tn(dh_test, q_test)
    assert Tn.shape == (4, 4), "La matrice T0,n n'est pas de taille 4x4"
    print("Résultat : OK\n")

def test_xy_Ot():
    print("Test : xy_Ot")
    Tn = matrice_Tn(dh_test, q_test)
    xyz = xy_Ot(Tn)
    assert len(xyz) == 3, "Les coordonnées extraites ne sont pas de taille 3"
    print("Résultat : OK\n")

def test_mgd():
    print("Test : mgd")
    Xd = mgd(q_test, Liaisons_test)
    assert len(Xd) == 3, "Les coordonnées retournées par le MGD ne sont pas de taille 3"
    print("Résultat : OK\n")

def test_mgi():
    print("Test : mgi")
    solutions = mgi(Xd_test, Liaisons_test)
    assert isinstance(solutions, list), "Le MGI ne retourne pas une liste"
    assert all(len(sol) == 3 for sol in solutions), "Une solution ne contient pas 3 angles"
    print("Résultat : OK\n")

def test_verifier_solutions():
    print("Test : verifier_solutions")
    verifier_solutions(Xd_test, Liaisons_test)
    print("Résultat : Vérification effectuée\n")

# Exécution des tests
if __name__ == "__main__":
    print("Lancement des tests...\n")
    test_matrice_Tim1_Ti()
    test_generate_transformation_matrices()
    test_matrice_Tn()
    test_xy_Ot()
    test_mgd()
    test_mgi()
    test_verifier_solutions()
    print("Tous les tests ont été passés avec succès.")
