import numpy as np
import plotly.graph_objects as go
from const_v import *  # Assurez-vous qu'il contient vos constantes
from matrice_tn import mgi  # Assurez-vous que `mgi` est correctement implémenté

def bras_rob_model3D_animation(Liaisons, Robot_pos):
    """
    Anime un bras robotique suivant une trajectoire circulaire définie par deux points (A et B).

    :param Liaisons: Dictionnaire contenant les dimensions des liaisons du bras.
    :param Robot_pos: Liste contenant deux points [A, B] qui définissent le cercle.
    """
    if len(Robot_pos) != 2:
        raise ValueError("Robot_pos doit contenir exactement deux points : [A, B].")

    point_a, point_b = Robot_pos

    # Valider que X est constant
    if point_a[0] != point_b[0]:
        raise ValueError("Les coordonnées X des deux points doivent être égales.")

    # Valider que seules les coordonnées Z ou Y changent
    if point_a[1] != point_b[1] and point_a[2] != point_b[2]:
        raise ValueError("Seule une des coordonnées Y ou Z doit varier entre les points A et B.")

    # Calculer le centre et le rayon du cercle
    y_center = (point_a[1] + point_b[1]) / 2
    z_center = (point_a[2] + point_b[2]) / 2
    x_center = point_a[0]  # X constant

    radius = np.sqrt((point_a[1] - point_b[1]) ** 2 + (point_a[2] - point_b[2]) ** 2) / 2

    # Générer des points pour le cercle
    theta = np.linspace(0, 2 * np.pi, 100)
    if point_a[1] != point_b[1]:  # Si Y change, Z est constant pour A et B
        if point_a[1] < y_center:
            y_circle = point_a[1] + radius * (1 - np.cos(theta))
            z_circle = point_a[2] + radius * np.sin(theta)
        else:
            y_circle = point_a[1] - radius * (1 - np.cos(theta))
            z_circle = point_a[2] + radius * np.sin(theta)
    else:  # Si Z change, Y est constant pour A et B
        if point_a[2] < z_center:
            y_circle = point_a[1] + radius * np.sin(theta)
            z_circle = point_a[2] + radius * (1 - np.cos(theta))
        else:
            y_circle = point_a[1] - radius * np.sin(theta)
            z_circle = point_a[2] - radius * (1 - np.cos(theta))
    x_circle = np.full_like(y_circle, x_center)

    # Initialiser les configurations du bras
    frames = []

    for i in range(len(theta)):
        # Calculer la position actuelle de l'extrémité du bras
        target_pos = [x_circle[i], y_circle[i], z_circle[i]]

        # Obtenir les angles à l'aide de la fonction mgi
        solutions = mgi(target_pos, Liaisons)
        if not solutions:
            raise ValueError(f"Aucune solution trouvée pour la position cible : {target_pos}")
        q1, q2, q3 = np.radians(solutions[1])  # Prendre la première solution et convertir en radians

        # Calculer les positions intermédiaires du bras
        L1, L2, L3 = Liaisons["Liaison 1"], Liaisons["Liaison 2"], Liaisons["Liaison 3"]
        x1, y1, z1 = 0, 0, L1[1]
        x2, y2, z2 = L1[0] * np.cos(q1), L1[0] * np.sin(q1), z1
        x3, y3, z3 = x2 + L2[2] * np.cos(q1 + np.pi / 2), y2 + L2[2] * np.sin(q1 + np.pi / 2), z2
        x4, y4, z4 = x3 + L2[1] * np.cos(q2) * np.cos(q1), y3 + L2[1] * np.cos(q2) * np.sin(q1), z3 + L2[1] * np.sin(q2)
        x5, y5, z5 = x4 + L3[2] * np.cos(q1 - np.pi / 2), y4 + L3[2] * np.sin(q1 - np.pi / 2), z4
        x6, y6, z6 = x5 + L3[1] * np.cos(q3 + q2) * np.cos(q1), y5 + L3[1] * np.cos(q3 + q2) * np.sin(q1), z5 + L3[1] * np.sin(q3 + q2)

        # Créer une frame avec la position actuelle du bras
        frames.append(go.Frame(data=[
            go.Scatter3d(
                x=[0, x1, x2, x3, x4, x5, x6],
                y=[0, y1, y2, y3, y4, y5, y6],
                z=[0, z1, z2, z3, z4, z5, z6],
                mode='lines+markers',
                marker=dict(size=4),
                line=dict(color='blue', width=5),
                showlegend=False
            ),
            go.Scatter3d(
                x=[point_a[0], point_b[0]],
                y=[point_a[1], point_b[1]],
                z=[point_a[2], point_b[2]],
                mode='markers+text',
                marker=dict(size=6, color=['red', 'green'], symbol='cross'),
                text=['A', 'B'],
                textposition='top center',
                showlegend=False
            ),
            go.Scatter3d(
                x=x_circle,
                y=y_circle,
                z=z_circle,
                mode='lines',
                line=dict(color='orange', width=2),
                showlegend=False
            )
        ]))

    # Créer la figure initiale
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=[0, x1, x2, x3, x4, x5, x6],
                y=[0, y1, y2, y3, y4, y5, y6],
                z=[0, z1, z2, z3, z4, z5, z6],
                mode='lines+markers',
                marker=dict(size=4),
                line=dict(color='blue', width=5),
                showlegend=False
            ),
            go.Scatter3d(
                x=[point_a[0], point_b[0]],
                y=[point_a[1], point_b[1]],
                z=[point_a[2], point_b[2]],
                mode='markers+text',
                marker=dict(size=6, color=['red', 'green'], symbol='cross'),
                text=['A', 'B'],
                textposition='top center',
                showlegend=False
            ),
            go.Scatter3d(
                x=x_circle,
                y=y_circle,
                z=z_circle,
                mode='lines',
                line=dict(color='orange', width=2),
                showlegend=False
            )
        ],
        frames=frames
    )

    # Configurer les boutons de lecture
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Rejouer",
                     method="animate",
                     args=[None, {"frame": {"duration": 40, "redraw": True}, "fromcurrent": True}]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
            ]
        )]
    )

    # Configurer le design du graphique
    fig.update_layout(
        scene_aspectmode='cube',
        scene=dict(
            xaxis=dict(title="Axe X", range=[-rayon_max1_5, rayon_max1_5]),
            yaxis=dict(title="Axe Y", range=[-rayon_max1_5, rayon_max1_5]),
            zaxis=dict(title="Axe Z", range=[0, 2 * rayon_max1_5])
        ),
        title="Animation 3D du Bras Robotique"
    )

    # Afficher la figure
    fig.show()

A = [300, 0, 100]
B = [300, 0, 2100]
Robot_pos = [A, B]
bras_rob_model3D_animation(Liaisons, Robot_pos)
