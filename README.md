# Circular Trajectory Generation with a Robot Manipulator - RX160

## Description

Ce repository contient un modèle complet de robot RRR, incluant les paramètres de modélisation, les simulations et les tests associés. Il permet de réaliser des calculs de modélisation géométrique, cinématique et dynamique, ainsi que des simulations de trajectoires.

---

## Structure des fichiers

### `const_v.py`
- Contient les constantes définissant le robot RRR.
- Paramètres de distances des axes et paramètres Denavit-Hartenberg (DHM).

### `matrices_tn.py`
- **`matrice_Tim1_Ti(qi, ai_m1, alphai_m1, ri, Debug=False)`**  
  Calcule la matrice de transformation DH entre deux liaisons successives.

- **`generate_transformation_matrices(q, dh, round_p=False, Debug=False)`**  
  Génère une liste de matrices de transformation \( T(i, i+1) \) à partir des paramètres DH.

- **`matrice_Tn(dh, q, Debug=False)`**  
  Calcule la matrice globale \( T0,n \) en utilisant les paramètres DH et les angles articulaires \( q \).

- **`mgd(q, Liaisons, Debug=False)`**  
  Résout la modélisation géométrique directe.

- **`mgi(Xd, Liaisons, Debug=False)`**  
  Résout la modélisation géométrique inverse.

- **`xy_Ot(result_matrix)`**  
  Extrait les coordonnées opérationnelles obtenues à partir de la matrice \( T0,n \).

---

### `modele_differentiel.py`
- Contient les fonctions liées au modèle différentiel du robot, notamment :
  - Jacobiennes calculées géométriquement.
  - Jacobiennes calculées analytiquement.
  - Modèle Différentiel Direct (MDD).
  - Modèle Différentiel Inverse (MDI).

---

### `trajectory_generation.py`
- **`traj(A, B, V1, V2, K, Debug=False)`**  
  Génère une trajectoire circulaire dans l’espace \( \mathbb{R}^3 \) entre deux points \( A \) et \( B \).  
  **Arguments :**
  - `A`, `B` : Points de départ et d'arrivée \([x, y, z]\).
  - `V1`, `V2` : Vitesses initiale et finale (mm/s).
  - `K` : Accélération.
  - `Debug` : Affiche les détails pour le débogage.  
  **Retourne :**
  - Trajectoires articulaires, vitesses et positions opérationnelles.

---

### `main.py`
- Fichier principal à exécuter.
- Permet aux utilisateurs de tester et utiliser toutes les fonctionnalités via des interactions guidées.

---

### Tests et simulations
- Les fichiers annexes contiennent les simulations et tests nécessaires pour valider le modèle et les fonctions.

---

## Utilisation

### 1. Cloner le repository
```bash
git clone https://github.com/OlivierCrt/Engineering_Consultancy_Robotic_Modeling
```
### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```
### 3. Exécuter le fichier principal
```bash
python Test/main.py
```
Si une erreur de module non reconnu intervient :
Solution 1 :
```bash
python -m Tests.main.py
```
Solution 2 :
Ajouter le dossier principal à vos variables d’environnement (dépendant de votre OS).


### 4. Entrées attendues
Unités :
Vitesses linéaires : mm/s
Vitesses angulaires : rad/s
Vitesses articulaires : rad/s
Distances : mm
