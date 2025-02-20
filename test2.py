import numpy as np
import matplotlib.pyplot as plt

# Paramètres temporels
dt = 0.01             # pas de temps (min)
T_total = 200         # durée totale de la simulation (min)
n_steps = int(T_total / dt)
time = np.linspace(0, T_total, n_steps)

# Paramètres du modèle
s = 0.1               # taux de synthèse de cAMP (nM/min)
k_ec = 0.4            # taux de dégradation/dilution non lié à la PDE (min^-1)
alpha = 0.5           # efficacité de dégradation par PDE (min^-1)

# Paramètres pour la régulation de la PDE
k_syn = 0.5           # taux de synthèse de PDE (min^-1)
k_deg = 0.001         # taux de dégradation de PDE (min^-1)
K = 1.0               # concentration caractéristique (nM) pour la réponse en cloche
b = 0.5               # expression basale de PDE

# Fonction de régulation de PDE : courbe en cloche
def g_reg(g):
    return (g**2) / ((g**2 + K**2)**2) + b

# Initialisation des variables
g = np.zeros(n_steps)   # concentration extracellulaire de cAMP (nM)
c = np.zeros(n_steps)   # niveau de PDE (dimensionless, normalisé)
g[0] = 0.2              # condition initiale pour g (nM)
c[0] = 0.05             # condition initiale pour c

# Intégration par méthode d'Euler
for i in range(n_steps - 1):
    dg = s - k_ec * g[i] - alpha * c[i] * g[i]
    dc = k_syn * g_reg(g[i]) - k_deg * c[i]
    g[i+1] = g[i] + dt * dg
    c[i+1] = c[i] + dt * dc

# Tracé des résultats
plt.figure(figsize=(12, 5))

# Courbe de cAMP en fonction du temps
plt.subplot(1, 2, 1)
plt.plot(time, g, label='cAMP (g)')
plt.xlabel('Temps (min)')
plt.ylabel('Concentration de cAMP (nM)')
plt.title('Concentration de cAMP vs Temps')
plt.legend()

# Courbe de PDE en fonction du temps
plt.subplot(1, 2, 2)
plt.plot(time, c, label='PDE (c)', color='orange')
plt.xlabel('Temps (min)')
plt.ylabel('Niveau de PDE')
plt.title('Concentration de PDE vs Temps')
plt.legend()

plt.tight_layout()
plt.show()

# Diagramme de phase (c vs g)
plt.figure(figsize=(6,5))
plt.plot(g, c, lw=2)
plt.xlabel('cAMP (g) [nM]')
plt.ylabel('PDE (c)')
plt.title('Diagramme de phase')
plt.show()