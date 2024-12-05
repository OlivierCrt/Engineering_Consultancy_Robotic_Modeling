import plotly.graph_objects as go
from const_v import *
from matrice_tn import mgi

def bras_rob_model3D_animation(Liaisons, Robot_pos, frames_per_transition=30, rayon_max1_5=1600):
    # Créer une semisphère
    def create_semisphere(rayon_max1_5, resolution=20):
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi / 2, resolution)  # Limiter à une demi-sphère (0 à π/2)
        x = rayon_max1_5 * np.outer(np.cos(u), np.sin(v))
        y = rayon_max1_5 * np.outer(np.sin(u), np.sin(v))
        z = rayon_max1_5 * np.outer(np.ones(np.size(u)), np.cos(v))
        return x, y, z

    # S'assurer que le dernier point est répété deux fois
    if Robot_pos:
        dernier_point = Robot_pos[-1]
        if len(Robot_pos) < 3 or Robot_pos[-1] != Robot_pos[-2] or Robot_pos[-2] != Robot_pos[-3]:
            Robot_pos.extend([dernier_point, dernier_point])  # Ajouter exactement deux points supplémentaires

    # Calculer les angles pour chaque position dans Robot_pos
    all_angles = []
    for pos in Robot_pos:
        solutions = mgi(pos, Liaisons)
        if solutions:
            all_angles.append(solutions[1])  # Choisir la première solution valide
        else:
            raise ValueError(f"Aucune solution trouvée pour la position {pos}")

    # Interpoler les angles pour une transition fluide entre chaque position
    interpolated_angles = []
    for i in range(len(all_angles) - 1):
        start = np.array(all_angles[i])
        end = np.array(all_angles[i + 1])
        for t in np.linspace(0, 1, frames_per_transition):
            interpolated_angles.append(start + t * (end - start))

    # Générer les points de la semisphère
    x, y, z = create_semisphere(rayon_max1_5)

    # Semisphère
    semisphere = go.Mesh3d(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        opacity=0.2,  # Transparence
        color='lightblue',  # Couleur
        alphahull=0,
        showlegend=False  # Masquer de la légende
    )

    # Listes des points de sortie (rouges) et d'entrée (verts)
    red_points = []
    green_points = []

    # Liste des croix avec numérotation pour les configurations valides
    labels = []

    # Ajouter des étiquettes pour toutes les configurations
    for idx, pos in enumerate(Robot_pos):
        x, y, z = pos
        labels.append(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode="text+markers",
            marker=dict(size=8, symbol="cross", color="black"),
            text=[str(idx + 1)],
            textposition="top center",
            showlegend=False  # Masquer de la légende
        ))

    # Variable pour suivre l'état précédent (dedans ou dehors)
    previous_exceeded = False

    # Générer des données pour l'animation
    frames = []

    for angles in interpolated_angles:
        q1, q2, q3 = np.radians(angles)  # Convertir les angles en radians

        L1 = Liaisons["Liaison 1"]
        L2 = Liaisons["Liaison 2"]
        L3 = Liaisons["Liaison 3"]

        # Calculer les coordonnées intermédiaires pour le tracé 3D
        x1, y1, z1 = 0, 0, L1[1]
        x2, y2, z2 = L1[0] * np.cos(q1), L1[0] * np.sin(q1), z1
        x3, y3, z3 = x2 + L2[2] * np.cos(q1 + np.pi / 2), y2 + L2[2] * np.sin(q1 + np.pi / 2), z2
        x4, y4, z4 = x3 + L2[1] * np.cos(q2) * np.cos(q1), y3 + L2[1] * np.cos(q2) * np.sin(q1), z3 + L2[1] * np.sin(q2)
        x5, y5, z5 = x4 + L3[2] * np.cos(q1 - np.pi / 2), y4 + L3[2] * np.sin(q1 - np.pi / 2), z4
        x6, y6, z6 = x5 + L3[1] * np.cos(q3 + q2) * np.cos(q1), y5 + L3[1] * np.cos(q3 + q2) * np.sin(q1), z5 + L3[1] * np.sin(q3 + q2)

        # Vérifier si la position dépasse le rayon
        distance = np.sqrt(x6**2 + y6**2 + z6**2)
        exceeded = distance > rayon_max1_5

        # Ajouter un point rouge si la limite est dépassée pour la première fois
        if exceeded and not previous_exceeded:
            red_points.append((x6, y6, z6))

        # Ajouter un point vert si revient à l'intérieur pour la première fois
        if not exceeded and previous_exceeded:
            green_points.append((x6, y6, z6))

        # Mettre à jour l'état précédent
        previous_exceeded = exceeded

        # Ajouter les données de la frame
        frames.append(go.Frame(data=[
            semisphere,  # Semisphère toujours visible
            go.Scatter3d(
                x=[0, x1, x2, x3, x4, x5, x6],
                y=[0, y1, y2, y3, y4, y5, y6],
                z=[0, z1, z2, z3, z4, z5, z6],
                mode='lines+markers',
                marker=dict(size=4),
                line=dict(color='blue', width=5),
                showlegend=False  # Masquer de la légende
            ),
            go.Scatter3d(
                x=[pt[0] for pt in red_points],
                y=[pt[1] for pt in red_points],
                z=[pt[2] for pt in red_points],
                mode='markers',
                marker=dict(size=8, symbol="cross", color="red"),
                name="Points de sortie"
            ),
            go.Scatter3d(
                x=[pt[0] for pt in green_points],
                y=[pt[1] for pt in green_points],
                z=[pt[2] for pt in green_points],
                mode='markers',
                marker=dict(size=8, symbol="cross", color="green"),
                name="Points d'entrée"
            )
        ] + labels))  # Ajouter les étiquettes pour chaque frame

    # Créer la figure initiale
    fig = go.Figure(
        data=[
            semisphere,  # Ajouter la semisphère
            go.Scatter3d(
                x=[0, x1, x2, x3, x4, x5, x6],
                y=[0, y1, y2, y3, y4, y5, y6],
                z=[0, z1, z2, z3, z4, z5, z6],
                mode='lines+markers',
                marker=dict(size=4),
                line=dict(color='blue', width=5),
                showlegend=False  # Masquer le bras robotique de la légende
            ),
            go.Scatter3d(
                x=[pt[0] for pt in red_points],
                y=[pt[1] for pt in red_points],
                z=[pt[2] for pt in red_points],
                mode='markers',
                marker=dict(size=8, symbol="cross", color="red"),
                name="Points de sortie"
            ),
            go.Scatter3d(
                x=[pt[0] for pt in green_points],
                y=[pt[1] for pt in green_points],
                z=[pt[2] for pt in green_points],
                mode='markers',
                marker=dict(size=8, symbol="cross", color="green"),
                name="Points d'entrée"
            )
        ] + labels,  # Ajouter les étiquettes dès le début
        frames=frames
    )

    # Configurer les boutons de lecture
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(label="Animer",
                     method="animate",
                     args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
            ]
        )]
    )

    # Configurer la mise en page des axes avec des plages constantes
    fig.update_layout(scene_aspectmode='cube', scene=dict(
        xaxis=dict(title="Axe X", range=[-rayon_max1_5, rayon_max1_5]),
        yaxis=dict(title="Axe Y", range=[-rayon_max1_5, rayon_max1_5]),
        zaxis=dict(title="Axe Z", range=[0, rayon_max1_5])
    ))

    # Afficher l'animation
    fig.show()


# Exemple d'utilisation
bras_rob_model3D_animation(Liaisons, Robot_pos)
