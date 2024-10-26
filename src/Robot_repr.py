import numpy as np
import plotly.graph_objects as go

#Parametres de Denavit Hartenberg modifié
dh = {}
dh["sigma_i"] = [0, 0, 0]
dh["a_i_m1"] = [0, 150, 825]
dh["alpha_i_m1"] =[0,np.pi/2,0]
dh["r_i"] = [550, 0, -3.5]
max_dist=2110   #Distance maximale lorsque le bras est totalement tendu vers le haut (en mm)

#Parametres de l'arrondi
decimals=2
threshold=1e-7
round_p=(decimals,threshold)

#Normes des longueurs du bras
#Param de horiz/vert
l1=np.sqrt(dh["r_i"][0]**2+dh["a_i_m1"][1]**2)
l2=dh["a_i_m1"][2]
l3=max_dist-(dh["r_i"][0]+l2)
Lxz=[l1,l2,l3]

#Param de profondeur
p1=dh["r_i"][2]

#Inclinaison horizontale de l1
inclin_horiz=(np.pi/2)-np.arctan(dh["a_i_m1"][1]/ dh["r_i"][0])

#Angles des liaisons en Degrés
q = [0,-20,-120]
rayon_max1_5=1600#en mm

#Pour le MGI
Xd = [700, 200, 500]
#q_initial=[0,-90,0]
Nb_iter=10000

q_rad = np.radians(q)

# Angles
teta = inclin_horiz
teta1 = q_rad[0]
teta2 = q_rad[1]
teta3 = q_rad[2]

# Premier segment
x1 = Lxz[0] * np.cos(teta)
y1 = (x1 / np.cos(teta1)) * np.sin(teta1 + np.arcsin(p1 / (Lxz[0] * np.cos(teta) + Lxz[1] + Lxz[2])))
z1 = Lxz[0] * np.sin(teta)

# Deuxieme segment
x2 = x1 + Lxz[1] * np.cos(teta + teta2)
y2 = (x2 / np.cos(teta1)) * np.sin(teta1 + np.arcsin(p1 / (Lxz[0] * np.cos(teta) + Lxz[1] + Lxz[2])))
z2 = z1 + Lxz[1] * np.sin(teta + teta2)

# Troisieme segment
x3 = x2 + Lxz[2] * np.cos(teta + teta2 + teta3)
y3 = (x3 / np.cos(teta1)) * np.sin(teta1 + np.arcsin(p1 / (Lxz[0] * np.cos(teta) + Lxz[1] + Lxz[2])))
z3 = z2 + Lxz[2] * np.sin(teta + teta2 + teta3)

# Creation d'un graph en 3D
fig = go.Figure(data=[go.Scatter3d(
    x=[0, x1, x2, x3],
    y=[0, y1, y2, y3],
    z=[0, z1, z2, z3],
    mode='lines+markers',
    marker=dict(size=5),
    line=dict(color='blue', width=5)
)])

# Configuration d'axes personalisés (Z en vertical et Y en profondeur)
fig.update_layout(scene=dict(
    xaxis=dict(title="Axe X", range=[x3, 0]),  # Invertir el eje X
    yaxis=dict(title="Axe Y", range=[y3, 0]),  # Invertir el eje Y
    zaxis=dict(title="Axe Z")
),
    title="Modelo 3D del Brazo Robótico con Ejes Invertidos"
)

fig.show()
