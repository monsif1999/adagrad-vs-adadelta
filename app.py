
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt



def loss_function(x,y):
    return x**2 + 20 * (y**2)

def gradient(x,y):
    return np.array([2*x, 40*y])

def adagrad_optimizer(start_x, start_y, lr , iterations, epsilon = 1e-8):
    gradients = np.zeros(2)
    path = []
    current_pos = np.array([start_x, start_y], dtype=float)
    for i in range(iterations):
        path.append(current_pos.copy())
        gradient_output = gradient(current_pos[0], current_pos[1])
        gradients+= gradient_output**2
        adjust = (lr/np.sqrt(gradients+epsilon))*gradient_output
        current_pos -= adjust

    return np.array(path)
def adadelta_optimizer(start_x, start_y, rho, iterations, epsilon = 1e-6):
    u = np.zeros(2)
    v = np.zeros(2)
    current_pos = np.array([start_x, start_y], dtype=float)
    parameter = 0
    path = []

    for i in range(iterations):
        path.append(current_pos.copy())
        u = rho * u + (1-rho)*(gradient(current_pos[0], current_pos[1])**2)
        delta = -((np.sqrt(v+epsilon))/(np.sqrt(u+epsilon)))*gradient(current_pos[0], current_pos[1
                                                                                                  ])
        current_pos +=  delta
        v = rho * v + (1-rho) * delta**2


    return np.array(path)
st.set_page_config(page_title="AdaGrad vs AdaDelta", layout="wide")

st.title("🥊 Le Duel : AdaGrad vs AdaDelta")
st.markdown("""
**Objectif :** Montrer pourquoi AdaDelta est plus robuste qu'AdaGrad.
* **AdaGrad (Rouge)** : Souffre d'un learning rate qui diminue trop vite ou nécessite un réglage manuel précis.
* **AdaDelta (Bleu)** : S'adapte automatiquement grâce aux moyennes mobiles (EMA) et gère les unités.
""")


st.sidebar.header("Paramètres")

iterations = st.sidebar.slider("Nombre d'itérations", 10, 200, 50)
start_x = st.sidebar.slider("Position X départ", -100.0, 100.0, -8.0)
start_y = st.sidebar.slider("Position Y départ", -50.0, 50.0, 4.0)

st.sidebar.markdown("---")
lr_adagrad = st.sidebar.slider("AdaGrad : Learning Rate", 0.01, 0.1, 1.5, help="Si trop petit, il s'arrête. Si trop grand, il explose.")
rho_adadelta = st.sidebar.slider("AdaDelta : Rho (Décadence)", 0.80, 0.999, 0.95, help="L'inertie de la mémoire.")


path_grad = adagrad_optimizer(start_x, start_y, lr_adagrad, iterations)
path_delta = adadelta_optimizer(start_x, start_y, rho_adadelta, iterations)

fig, ax = plt.subplots(figsize=(10, 6))

x_grid = np.linspace(-12, 12, 100)
y_grid = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x_grid, y_grid)
Z = loss_function(X, Y)

ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.5)

ax.plot(path_grad[:,0], path_grad[:,1], 'o-', color='red', label=f'AdaGrad (LR={lr_adagrad})', markersize=4, alpha=0.8)
ax.plot(path_delta[:,0], path_delta[:,1], 'o-', color='blue', label=f'AdaDelta (Rho={rho_adadelta})', markersize=4, alpha=0.8)

ax.plot(start_x, start_y, 'k*', markersize=15, label="Départ")
ax.plot(0, 0, 'rx', markersize=15, markeredgewidth=3, label="Objectif (Min)")

ax.set_xlim(-12, 12)
ax.set_ylim(-6, 6)
ax.legend()
ax.set_title(f"Comparaison de la convergence ({iterations} itérations)")
ax.grid(True, linestyle='--', alpha=0.3)

st.pyplot(fig)

dist_grad = np.linalg.norm(path_grad[-1])
dist_delta = np.linalg.norm(path_delta[-1])

col1, col2 = st.columns(2)
with col1:
    st.metric("Distance finale AdaGrad", f"{dist_grad:.4f}", delta_color="inverse")
with col2:
    st.metric("Distance finale AdaDelta", f"{dist_delta:.4f}", delta_color="inverse")

if dist_delta < dist_grad:
    st.success("✅ AdaDelta est plus proche de l'objectif (0,0) !")
else:
    st.warning("⚠️ AdaGrad a gagné (probablement grâce à un gros Learning Rate manuel).")