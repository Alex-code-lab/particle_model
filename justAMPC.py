import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------------
# Paramètres de la simulation
# -------------------------------
dt = 0.01              # pas de temps (min)
T_total = 100.0        # durée totale de la simulation (min)
n_steps = int(T_total/dt)

# -------------------------------
# Paramètres du modèle FitzHugh–Nagumo
# -------------------------------
a = 0.7                # paramètre d'excitation
b = 0.8                # paramètre de récupération
epsilon = 0.6          # accélère la dynamique de récupération (valeur ajustée)

pulse_threshold = 1.0  # seuil de déclenchement d'un pulse
pulse_strength = 400.0 # quantité de cAMP injectée lors d'un pulse

# -------------------------------
# Paramètres du champ extracellulaire d'AMPc
# -------------------------------
space_size = 100.0     # taille du domaine (μm)
dx = 1.0               # résolution spatiale (μm)
grid_size = int(space_size/dx)
D = 10.0                # coefficient de diffusion (μm²/min)
decay_rate = 0.5       # taux de dégradation du cAMP

# Initialisation du champ d'AMPc sur une grille 2D
camp_field = np.zeros((grid_size, grid_size))

# -------------------------------
# Définition des cellules
# -------------------------------
n_cells = 10000
# Positions aléatoires (entiers entre 0 et grid_size-1)
np.random.seed(42)  # pour reproductibilité
cell_positions = np.column_stack((
    np.random.randint(0, grid_size, size=n_cells),
    np.random.randint(0, grid_size, size=n_cells)
))

# Pour chaque cellule, on stocke ses variables d'état A et R
A_cells = np.zeros(n_cells)
R_cells = np.zeros(n_cells)

# Pour chaque cellule, on définit un courant d'entrée I_cell.
# On répartit aléatoirement 50% de cellules "déclenchées" (I=1.0) et 50% "latentes" (I=0.0)
I_cells = np.zeros(n_cells)
triggered = np.zeros(n_cells, dtype=bool)  # True si la cellule est déjà déclenchée

for i in range(n_cells):
    if np.random.rand() < 0.05:
        I_cells[i] = 1.0  # Cellule spontanée
        triggered[i] = True
    else:
        I_cells[i] = 0.0  # Cellule en état latent

# Tracé des positions initiales des cellules
plt.figure(figsize=(6,6))
# Séparation des cellules activées (triggered=True) et latentes (triggered=False)
activated = cell_positions[triggered]
latent = cell_positions[~triggered]

plt.scatter(activated[:, 0] * dx, activated[:, 1] * dx, color='green', s=10, label='Cellules activées')
plt.scatter(latent[:, 0] * dx, latent[:, 1] * dx, color='red', s=10, label='Cellules latentes')

plt.xlabel("x (μm)")
plt.ylabel("y (μm)")
plt.title("Positions initiales des cellules")
plt.legend()
plt.show()

# Seuil local de cAMP pour déclencher une cellule latente
camp_trigger_threshold = 100.0

# -------------------------------
# Préparation pour enregistrer les graphiques
# -------------------------------
output_dir = "camp_field_plots"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# Simulation
# -------------------------------
time_series = []      # pour tracer la moyenne de A, par exemple
save_interval = int(0.25/dt)  # enregistrer le champ de cAMP toutes les 1 minute

for step in range(n_steps):
    t = step * dt

    # Mise à jour de chaque cellule (FitzHugh–Nagumo)
    for i in range(n_cells):
        # Pour les cellules latentes, on vérifie le signal local de cAMP.
        # Si camp_field à la position de la cellule dépasse le seuil, on "déclenche" la cellule.
        x, y = cell_positions[i]
        if not triggered[i] and camp_field[x, y] > camp_trigger_threshold:
            I_cells[i] = 1.0
            triggered[i] = True

        dA = (A_cells[i] - (A_cells[i]**3)/3 - R_cells[i] + I_cells[i]) * dt
        dR = (A_cells[i] + a - b * R_cells[i]) * epsilon * dt
        A_old = A_cells[i]
        A_cells[i] += dA
        R_cells[i] += dR

        # Déclenchement d'un pulse si A franchit le seuil à la montée
        if A_cells[i] >= pulse_threshold and A_old < pulse_threshold:
            camp_field[x, y] += pulse_strength

    # Mise à jour du champ d'AMPc par diffusion et dégradation (schéma en différences finies)
    laplacian = (
        np.roll(camp_field, 1, axis=0) +
        np.roll(camp_field, -1, axis=0) +
        np.roll(camp_field, 1, axis=1) +
        np.roll(camp_field, -1, axis=1) -
        4 * camp_field
    ) / (dx**2)
    camp_field += dt * (D * laplacian - decay_rate * camp_field)

    time_series.append(np.mean(A_cells))

    # Sauvegarde du champ de cAMP toutes les 1 minute
    if step % save_interval == 0:
        plt.figure(figsize=(6,5))
        im = plt.imshow(camp_field, origin='lower', cmap='viridis', extent=[0, space_size, 0, space_size], vmin=0, vmax=110)
        plt.colorbar(im, label='Concentration de cAMP')
        plt.title(f"Champ d'AMPc à t = {t:.1f} min")
        plt.xlabel("x (μm)")
        plt.ylabel("y (μm)")
        filename = os.path.join(output_dir, f"camp_field_{t:05.2f}min.png")
        plt.savefig(filename)
        plt.close()

# Optionnel : tracer l'évolution moyenne de A sur le temps
plt.figure(figsize=(6,4))
plt.plot(np.linspace(0, T_total, n_steps), time_series, label="A moyenne")
plt.xlabel("Temps (min)")
plt.ylabel("Activation moyenne")
plt.title("Évolution moyenne de A sur toutes les cellules")
plt.legend()
plt.tight_layout()
plt.show()