import plotly.graph_objects as go
from const_v import *

"""FONTION POUR MODELISER EN 3D LE BRAS ROBOT, LA FONCTION EST DECLARE A LA TOUTE FIN"""


def bras_rob_model3D(Liaisons, q):
    q_rad = np.radians(q)

    L1 = Liaisons["Liaison 1"]
    L2 = Liaisons["Liaison 2"]
    L3 = Liaisons["Liaison 3"]

    # Angles
    teta1 = q_rad[0]
    teta2 = q_rad[1]
    teta3 = q_rad[2]

    """Chaque segment du robot necessite ces propres coordonnées"""
    x1 = 0
    y1 = 0
    z1 = L1[1]

    x2 = L1[0] * np.cos(teta1)
    y2 = L1[0] * np.sin(teta1)
    z2 = z1

    x3 = x2 + L2[2] * np.cos(teta1 + np.pi / 2)
    y3 = y2 + L2[2] * np.sin(teta1 + np.pi / 2)
    z3 = z2

    x4 = x3 + L2[0] * np.cos(teta2) * np.cos(teta1)
    y4 = y3 + L2[0] * np.cos(teta2) * np.sin(teta1)
    z4 = z3 + L2[0] * np.sin(teta2)

    x5 = x4 + L3[2] * np.cos(teta1 - np.pi / 2)
    y5 = y4 + L3[2] * np.sin(teta1 - np.pi / 2)
    z5 = z4

    x6 = x5 + L3[0] * np.cos(teta3 + teta2) * np.cos(teta1)
    y6 = y5 + L3[0] * np.cos(teta3 + teta2) * np.sin(teta1)
    z6 = z5 + L3[0] * np.sin(teta3 + teta2)

    X = [x3, x4, x6]
    Z = [z3, z4, z6]
    # Definition du rang de l'axe y
    max_y = max(abs(y1), abs(y2), abs(y3), abs(y4), abs(y5), abs(y6))
    if max_y > 1000:
        y_range = [-1600, 1600]
    elif max_y > 500:
        y_range = [-1000, 1000]
    else:
        y_range = [-500, 500]

    max_x = max(abs(x1), abs(x2), abs(x3), abs(x4), abs(x5), abs(x6))
    if max_x > 1000:
        x_range = [-1600, 1600]
    elif max_x > 500:
        x_range = [-1000, 1000]
    else:
        x_range = [-500, 500]

    if max_x < max_y:
        x_range = y_range
    if max_y < max_x:
        y_range = x_range

    # Creation d'un graph en 3D
    fig = go.Figure(data=[go.Scatter3d(
        x=[0, x1, x2, x3, x4, x5, x6],
        y=[0, y1, y2, y3, y4, y5, y6],
        z=[0, z1, z2, z3, z4, z5, z6],
        mode='lines+markers',
        marker=dict(size=4),
        line=dict(color='blue', width=5)
    )])

    # Configuration d'axes personalisés (Z en vertical et Y en profondeur)
    fig.update_layout(scene=dict(
        xaxis=dict(title="Axe X", range=x_range),
        yaxis=dict(title="Axe Y", range=y_range),
        zaxis=dict(title="Axe Z")
    ),
        title="Modéle bras robot selon configurations de q données"
    )

    return fig.show()


bras_rob_model3D(Liaisons, q)
