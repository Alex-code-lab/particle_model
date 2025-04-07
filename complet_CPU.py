#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Version optimisée pour CPU du modèle complet avec vectorisation et utilisation de Numba.

Ce script simule :
- Les déplacements des cellules Dictyostelium (forces, mouvement, etc.)
- La dynamique du cAMP/PDE sur une grille 2D (diffusion, production, dégradation)
- La mise à jour de l'état interne (modèle Martiel–Goldbeter)

Les parties critiques (diffusion, calculs sur grille) sont compilées avec Numba.
"""

import math
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from datetime import datetime
from numba import njit, prange

# ============================================
# 1) PARAMÈTRES GLOBAUX
# ============================================

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
GENERAL_PATH = "/Users/souchaud/Desktop/simulations/"
if not os.path.exists(GENERAL_PATH):
    os.makedirs(GENERAL_PATH)
PATH = os.path.join(GENERAL_PATH, f"simu_{timestamp}")
os.makedirs(PATH, exist_ok=True)
print(f"📁 Dossier de simulation créé : {PATH}")

use_saved_state = True
save_new_initial_state = False

# ------------------
# Espace / Temps
# ------------------
SPACE_SIZE = 500.0       # µm, taille du domaine
TIME_SIMU = 120.0        # min, durée de la simulation
PLOT_INTERVAL = 250      # fréquence d'affichage (en itérations)

# ------------------
# Paramètres "physiques" (forces)
# ------------------
MU = 1.0
F_REP = 0               # Force répulsive maximale
F_ADH = 0               # Force adhésive maximale
R_EQ = 4.0              # Distance d'équilibre
R_0 = 3.8               # Distance maximale d'interaction
COEFF_CARRE = 50.0      # Coefficient pour l'adhésion
COEFF_REP = 0.5         # Coefficient pour la répulsion
FLUCTUATION_FACTOR = 3  # Intensité du bruit sur la vitesse

# ------------------
# Paramètres "biochimiques" (cAMP / PDE)
# ------------------
GRID_RESOLUTION = 1.0          # µm, taille d'une case
D_CAMP = 150.0                 # µm²/min, diffusion du cAMP
D_PDE = 100.0                  # µm²/min, diffusion de la PDE
PDE_threshold = 1e-8           # M, seuil cAMP pour production de PDE
PDE_rate = 2.0                 # facteur de production de PDE
PDE_decay = 0.4                # min⁻¹, dégradation de la PDE
rho = 1e-5                     # M/min, production basale de cAMP (augmentée)
alpha0 = 9.8e-6                # M/min, facteur de production cAMP
J = 0.01                       # min⁻¹, dégradation globale du cAMP
k_PDE = 1.0                    # min⁻¹, dégradation du cAMP par la PDE
PDE_inhibition_threshold = 5e-6  # M

# ------------------
# Martiel–Goldbeter (paramètres cellulaires)
# ------------------
F1_base = 1.4     # taux de désensibilisation de base
F2_base = 1.7     # taux de réactivation de base
n = 4             # exposant de Hill (récepteur)
K_h = 0.8         # constante demi-saturation pour désensibilisation
hill_n = 3        # exposant de Hill pour rétroaction de production
hill_K_h = 1e-6   # constante demi-saturation pour rétroaction

# ------------------
# Paramètres population
# ------------------
PACKING_FRACTION = 0.8
estimated_cell_area = math.pi * (R_EQ)**2
N_CELLS = 2500
print(f"Nombre de cellules estimé = {N_CELLS}")

# ------------------
# Paramètres delta T (CFL pour diffusion)
# ------------------
DELTA_T = 0.5 * (GRID_RESOLUTION**2) / min(D_CAMP, D_PDE)
DELTA_T = min(DELTA_T, 0.001)
print("Pas de temps :", DELTA_T)

# ------------------
# Paramètres cinétiques (mouvement)
# ------------------
velocity_magnitude_pop1 = 0        
ECART_TYPE_POP1 = 0.3               
NOISE_POP_1 = 0                    
TAU_POP_1 = 5                      
PERSISTENCE_POP1 = 0               
SENSITIVITY_cAMP_THRESHOLD_POP1 = 2  

velocity_magnitude_pop2 = 0.1      
ECART_TYPE_POP2 = 0.5               
NOISE_POP_2 = 0                    
TAU_POP_2 = 5                      
PERSISTENCE_POP2 = 0               
SENSITIVITY_cAMP_THRESHOLD_POP2 = 2  

MIN_DISTANCE_INIT = 2 * R_EQ         

pop1 = N_CELLS // 2
pop2 = N_CELLS - pop1

# ============================================
# 2) FONCTIONS UTILITAIRES (NUMPY et NUMBA)
# ============================================

def adhesion_force(R, Req, R0, Fadh):
    """
    Calcule la force d'adhésion linéaire simple.
    
    Args:
        R (float): Distance.
        Req (float): Distance d'équilibre.
        R0 (float): Distance maximale d'interaction.
        Fadh (float): Force adhésive maximale.
        
    Returns:
        float: Force d'adhésion.
    """
    return -((Fadh / (R0 - Req)) * R - Fadh * Req / (R0 - Req))

def force_field_inbox(coordinates_diff, Req, R0, Frep, Fadh, coeff_a, coeff_rep):
    """
    Calcule le champ de force 2D agissant sur chaque cellule.
    
    Args:
        coordinates_diff (np.array): Tableau (N, N, 2) des différences de positions.
        Req (float): Distance d'équilibre.
        R0 (float): Distance maximale d'interaction.
        Frep (float): Force répulsive maximale.
        Fadh (float): Force adhésive maximale.
        coeff_a (float): Coefficient associé à l'adhésion.
        coeff_rep (float): Coefficient pour la répulsion.
    
    Returns:
        np.array: Tableau (N,2) représentant la force totale sur chaque cellule.
    """
    Rlim = 1e-6
    R = np.linalg.norm(coordinates_diff, axis=2)
    R = np.maximum(R, Rlim)
    mask_adh = (R < R0) & (R > Req)
    mask_rep = (R <= Req)
    force_adh = np.zeros_like(R)
    force_adh[mask_adh] = -((Fadh / (R0 - Req)) * R[mask_adh] - Fadh * Req / (R0 - Req))
    force_rep = np.zeros_like(R)
    force_rep[mask_rep] = -Frep * coeff_rep * (1.0/Req - 1.0/R[mask_rep])
    force = force_adh + force_rep
    norms = np.linalg.norm(coordinates_diff, axis=2, keepdims=True)
    directions = coordinates_diff / (norms + 1e-10)
    force_vectors = force[:, :, np.newaxis] * directions
    force_field = np.sum(force_vectors, axis=1)
    return force_field

def autovel(dX, n, tau, noise, dt, persistence):
    """
    Met à jour la direction d'une cellule en fonction du déplacement.
    
    Args:
        dX (np.array): Déplacement (2,).
        n (np.array): Ancienne direction (2,).
        tau (float): Temps de persistance.
        noise (float): Intensité du bruit.
        dt (float): Pas de temps.
        persistence (float): Facteur de persistance.
    
    Returns:
        np.array: Nouvelle direction (2,).
    """
    norm_dX = np.linalg.norm(dX)
    if norm_dX == 0:
        norm_dX = 1.0
    dX_norm = dX / norm_dX
    theta = np.arctan2(dX_norm[1], dX_norm[0])
    dtheta = np.arcsin(n[0] * dX_norm[1] - n[1] * dX_norm[0]) * dt / tau
    rnd = (2.0 * math.pi * (np.random.rand() - 0.5)) * noise * math.sqrt(dt)
    theta_update = theta + dtheta + rnd
    return np.array([math.cos(theta_update), math.sin(theta_update)])

# ============================================
# 3) FONCTIONS OPTIMISÉES AVEC NUMBA (diffusion et production)
# ============================================

@njit(parallel=True)
def diffuse_np_numba(grid, D, dt):
    rows, cols = grid.shape
    new_grid = np.empty_like(grid)
    for i in prange(rows):
        for j in prange(cols):
            up = grid[(i - 1) % rows, j]
            down = grid[(i + 1) % rows, j]
            left = grid[i, (j - 1) % cols]
            right = grid[i, (j + 1) % cols]
            new_grid[i, j] = grid[i, j] + D * (up + down + left + right - 4 * grid[i, j]) * dt
    return new_grid

@njit
def compute_PDE_production_numba(local_cAMP, PDE_threshold, PDE_rate):
    if local_cAMP < PDE_threshold:
        return 0.0
    normalized = (local_cAMP - PDE_threshold) / (1e-6 - PDE_threshold)
    if normalized < 0:
        normalized = 0.0
    elif normalized > 1:
        normalized = 1.0
    sin_factor = math.sin(math.pi * normalized / 2)**2
    return PDE_rate * sin_factor

@njit
def compute_cAMP_production_numba(local_cAMP, local_PDE, rho, alpha0, PDE_inhibition_threshold):
    cAMP_max = 1e-6
    if local_cAMP < cAMP_max:
        feedback = math.sin(math.pi * local_cAMP / (2 * cAMP_max))**2
    else:
        feedback = 1.0
    inhibition = 1.0
    if local_PDE > 0:
        inhibition = 1.0 / (1.0 + (local_PDE / PDE_inhibition_threshold)**2)
    return rho * alpha0 + feedback * inhibition

def produce_cAMP(cell, local_cAMP, local_PDE, dt):
    """
    Production locale de cAMP autour d'une cellule.
    """
    return compute_cAMP_production_numba(local_cAMP, local_PDE, rho, alpha0, PDE_inhibition_threshold) * dt

def update_cell_PDE_production(cell, local_cAMP, dt):
    """
    Met à jour la production de PDE pour une cellule.
    """
    ramp_up = 0.002
    ramp_down = 0.7
    if local_cAMP > PDE_threshold:
        cell.pde_production_level += ramp_up * dt * compute_PDE_production_numba(local_cAMP, PDE_threshold, PDE_rate)
    else:
        cell.pde_production_level -= ramp_down * dt
    if cell.pde_production_level < 0.0:
        cell.pde_production_level = 0.0
    elif cell.pde_production_level > 1.0:
        cell.pde_production_level = 1.0
    return cell.pde_production_level * PDE_rate * local_cAMP

def update_cell_MG(cell, local_cAMP, dt):
    """
    Met à jour l'état interne de la cellule selon le modèle Martiel–Goldbeter.
    """
    if local_cAMP > 0:
        f1_eff = F1_base * local_cAMP / (K_h + local_cAMP)
    else:
        f1_eff = 0.0
    dr_T = -f1_eff * cell.r_T + F2_base * (1 - cell.r_T)
    if local_cAMP > 0:
        F = cell.r_T / (1 + (K_h / local_cAMP)**n)
    else:
        F = 0.0
    q_s = 1.0
    k_t = 0.5
    db = q_s * F - k_t * cell.b
    cell.r_T += dr_T * dt
    cell.b += db * dt

# ============================================
# 4) CLASSES ET INITIALISATION DES CELLULES
# ============================================
class CellAgent:
    """
    Représente une cellule Dictyostelium.
    """
    _id_counter = 0
    def __init__(self, position, velocity, velocity_magnitude, tau, noise, persistence, pop_tag="Unknown"):
        self.id = CellAgent._id_counter
        CellAgent._id_counter += 1
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.velocity_magnitude = velocity_magnitude
        self.tau = tau
        self.noise = noise
        self.persistence = persistence
        norm = np.linalg.norm(self.velocity)
        self.direction = self.velocity / norm if norm != 0 else np.array([1.0, 0.0])
        self.pop = pop_tag
        self.b = 0.0
        self.r_T = 1.0
        self.pde_production_level = 0.0

class Population:
    """
    Génère une population de cellules en respectant une distance minimale.
    """
    def __init__(self, num_cells, space_size, velocity_magnitude, tau, noise,
                 ecart_type, persistence, min_distance, pop_tag, existing_cells=None):
        self.num_cells = num_cells
        self.space_size = space_size
        self.velocity_magnitude = velocity_magnitude
        self.tau = tau
        self.noise = noise
        self.ecart_type = ecart_type
        self.persistence = persistence
        self.min_distance = min_distance
        self.pop_tag = pop_tag
        self.existing_cells = existing_cells if existing_cells is not None else []
        self.cells = []
        self.initialize_cells()

    def initialize_cells(self):
        max_attempts = 100
        for i in range(self.num_cells):
            attempt = 0
            placed = False
            while attempt < max_attempts and not placed:
                attempt += 1
                candidate = np.random.rand(2) * self.space_size
                conflict = False
                for other in self.cells + self.existing_cells:
                    if np.linalg.norm(candidate - other.position) < self.min_distance:
                        conflict = True
                        break
                if not conflict:
                    direction = np.random.rand(2) - 0.5
                    norm = np.linalg.norm(direction)
                    direction = direction / norm if norm != 0 else np.array([1.0, 0.0])
                    speed = np.random.normal(self.velocity_magnitude, self.ecart_type)
                    velocity = direction * speed
                    new_cell = CellAgent(candidate, velocity, speed, self.tau, self.noise, self.persistence, pop_tag=self.pop_tag)
                    self.cells.append(new_cell)
                    placed = True
            if not placed:
                print(f"Avertissement: Impossible de placer une cellule dans {self.pop_tag}")

# ============================================
# 5) FONCTIONS DE VISUALISATION
# ============================================
def plot_cells_and_fields(cells, camp_grid, pde_grid, iteration, time_now, space_size, grid_resolution, path_saving=None):
    fig, axes = plt.subplots(1, 3, figsize=(18,6), constrained_layout=True)
    
    # Positions des cellules
    ax = axes[0]
    ax.set_title(f"Positions (t={time_now:.2f} min)")
    ax.set_xlim(0, space_size)
    ax.set_ylim(0, space_size)
    ax.set_aspect('equal')
    x_pop1 = [c.position[0] for c in cells if c.pop=="Population 1"]
    y_pop1 = [c.position[1] for c in cells if c.pop=="Population 1"]
    x_pop2 = [c.position[0] for c in cells if c.pop=="Population 2"]
    y_pop2 = [c.position[1] for c in cells if c.pop=="Population 2"]
    ax.scatter(x_pop1, y_pop1, s=10, color="blue", alpha=0.6, label="Population 1")
    ax.scatter(x_pop2, y_pop2, s=10, color="red", alpha=0.6, label="Population 2")
    ax.legend(loc="upper right")
    
    # Champ de cAMP
    ax1 = axes[1]
    ax1.set_title("Champ de cAMP")
    ax1.set_aspect('equal')
    extent = [0, space_size, 0, space_size]
    cimg = ax1.imshow(camp_grid.T, origin='lower', extent=extent, cmap="viridis")
    plt.colorbar(cimg, ax=ax1)
    
    # Champ de PDE
    ax2 = axes[2]
    ax2.set_title("Champ de PDE")
    ax2.set_aspect('equal')
    pimg = ax2.imshow(pde_grid.T, origin='lower', extent=extent, cmap="plasma")
    plt.colorbar(pimg, ax=ax2)
    
    if path_saving:
        filename = os.path.join(path_saving, f"frame_{iteration}.png")
        plt.savefig(filename, dpi=200)
    plt.close(fig)

# ============================================
# 6) SAUVEGARDE/CHARGEMENT DE L'ÉTAT INITIAL
# ============================================
def save_initial_state(cells, filename="initial_state.pkl"):
    state_data = [{
        'id': cell.id,
        'position': cell.position,
        'velocity': cell.velocity,
        'direction': cell.direction,
        'velocity_magnitude': cell.velocity_magnitude,
        'b': cell.b,
        'r_T': cell.r_T,
        'pop': cell.pop
    } for cell in cells]
    with open(filename, "wb") as f:
        pickle.dump(state_data, f)
    print(f"✅ État initial sauvegardé dans '{filename}'")

def load_initial_state(filename="initial_state.pkl"):
    with open(filename, "rb") as f:
        state_data = pickle.load(f)
    loaded_cells = []
    for data in state_data:
        cell = CellAgent(
            position=data['position'],
            velocity=data['velocity'],
            velocity_magnitude=data['velocity_magnitude'],
            tau=TAU_POP_1 if data['pop']=="Population 1" else TAU_POP_2,
            noise=NOISE_POP_1 if data['pop']=="Population 1" else NOISE_POP_2,
            persistence=PERSISTENCE_POP1 if data['pop']=="Population 1" else PERSISTENCE_POP2,
            pop_tag=data['pop']
        )
        cell.direction = data['direction']
        cell.b = data['b']
        cell.r_T = data['r_T']
        loaded_cells.append(cell)
    print(f"✅ État initial chargé depuis '{filename}'")
    return loaded_cells

# ============================================
# 7) SAUVEGARDE DES PARAMÈTRES DE SIMULATION
# ============================================
def save_simulation_parameters(filename="simulation_parameters.txt"):
    with open(filename, "w") as f:
        f.write(f"SPACE_SIZE = {SPACE_SIZE} µm\n")
        f.write(f"TIME_SIMU = {TIME_SIMU} min\n")
        f.write(f"DELTA_T = {DELTA_T} min\n")
        f.write(f"D_CAMP = {D_CAMP} µm²/min, D_PDE = {D_PDE} µm²/min\n")
        f.write(f"PDE_threshold = {PDE_threshold} M, PDE_rate = {PDE_rate}, PDE_decay = {PDE_decay} min⁻¹\n")
        f.write(f"rho = {rho} M/min, alpha0 = {alpha0} M/min, J = {J} min⁻¹, k_PDE = {k_PDE} min⁻¹\n")
        f.write(f"PDE_inhibition_threshold = {PDE_inhibition_threshold} M\n")
        f.write(f"F1_base = {F1_base}, F2_base = {F2_base}, n = {n}, K_h = {K_h}\n")
        f.write(f"hill_n = {hill_n}, hill_K_h = {hill_K_h}\n")
        f.write(f"N_CELLS = {N_CELLS}, MIN_DISTANCE_INIT = {MIN_DISTANCE_INIT} µm\n")
    print(f"Les paramètres de simulation ont été enregistrés dans '{filename}'.")

save_simulation_parameters(os.path.join(PATH, "simulation_parameters.txt"))

# ============================================
# 8) BOUCLE PRINCIPALE DE SIMULATION
# ============================================
def main():
    # Chargement ou création de l'état initial
    if use_saved_state:
        cells = load_initial_state(os.path.join(GENERAL_PATH, "initial_state.pkl"))
    else:
        CellAgent._id_counter = 0
        population1 = Population(
            num_cells=pop1,
            space_size=SPACE_SIZE,
            velocity_magnitude=velocity_magnitude_pop1,
            tau=TAU_POP_1,
            noise=NOISE_POP_1,
            ecart_type=ECART_TYPE_POP1,
            persistence=PERSISTENCE_POP1,
            min_distance=MIN_DISTANCE_INIT,
            pop_tag="Population 1"
        )
        population2 = Population(
            num_cells=pop2,
            space_size=SPACE_SIZE,
            velocity_magnitude=velocity_magnitude_pop2,
            tau=TAU_POP_2,
            noise=NOISE_POP_2,
            ecart_type=ECART_TYPE_POP2,
            persistence=PERSISTENCE_POP2,
            min_distance=MIN_DISTANCE_INIT,
            pop_tag="Population 2",
            existing_cells=population1.cells
        )
        cells = population1.cells + population2.cells
        if save_new_initial_state:
            save_initial_state(cells, os.path.join(GENERAL_PATH, "initial_state.pkl"))
    
    GRID_SIZE = int(np.ceil(SPACE_SIZE / GRID_RESOLUTION))
    print("GRID_SIZE =", GRID_SIZE)
    
    # Création des grilles cAMP et PDE
    camp_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    pde_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    
    D_camp_eff = D_CAMP
    D_pde_eff = D_PDE

    output_path = PATH
    os.makedirs(output_path, exist_ok=True)
    
    time_sim = 0.0
    iteration = 0
    n_steps = int(TIME_SIMU / DELTA_T)
    data_log = []
    
    # Initialisation des positions, directions et vitesses
    positions = np.array([cell.position for cell in cells])
    directions = np.array([cell.direction for cell in cells])
    v0 = np.array([cell.velocity_magnitude for cell in cells]).reshape(-1, 1)
    
    while iteration < n_steps:
        R = max(1, int(R_EQ / GRID_RESOLUTION))
        sigma = R / GRID_RESOLUTION
        
        cAMP_prod_grid = np.zeros_like(camp_grid)
        PDE_prod_grid = np.zeros_like(pde_grid)
        
        for cell in cells:
            x_idx = int(cell.position[0] // GRID_RESOLUTION)
            y_idx = int(cell.position[1] // GRID_RESOLUTION)
            if x_idx < 0 or x_idx >= GRID_SIZE or y_idx < 0 or y_idx >= GRID_SIZE:
                continue
            cAMP_local = camp_grid[x_idx, y_idx]
            PDE_local = pde_grid[x_idx, y_idx]
            
            cAMP_brut = produce_cAMP(cell, cAMP_local, PDE_local, DELTA_T)
            PDE_brut = update_cell_PDE_production(cell, cAMP_local, DELTA_T)
            
            production_scaling = GRID_RESOLUTION**2
            cAMP_brut /= production_scaling
            PDE_brut /= production_scaling
            
            indices_voisins = []
            weights = []
            for dx in range(-R, R+1):
                for dy in range(-R, R+1):
                    nx = x_idx + dx
                    ny = y_idx + dy
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                        w = math.exp(- (dx**2 + dy**2) / (2 * sigma**2))
                        indices_voisins.append((nx, ny))
                        weights.append(w)
            sum_w = sum(weights)
            if sum_w > 0:
                weights = [w/sum_w for w in weights]
                for (nx, ny), w_norm in zip(indices_voisins, weights):
                    cAMP_prod_grid[nx, ny] += cAMP_brut * w_norm
                    PDE_prod_grid[nx, ny] += PDE_brut * w_norm
        
        camp_grid += cAMP_prod_grid
        pde_grid += PDE_prod_grid
        
        camp_grid = diffuse_np_numba(camp_grid, D_camp_eff, DELTA_T)
        pde_grid = diffuse_np_numba(pde_grid, D_pde_eff, DELTA_T)
        
        camp_grid -= (J * camp_grid + k_PDE * pde_grid * camp_grid) * DELTA_T
        pde_grid -= PDE_decay * pde_grid * DELTA_T
        
        camp_grid = np.clip(camp_grid, 0, None)
        pde_grid = np.clip(pde_grid, 0, None)
        
        for cell in cells:
            x_idx = int(cell.position[0] // GRID_RESOLUTION)
            y_idx = int(cell.position[1] // GRID_RESOLUTION)
            if 0 <= x_idx < GRID_SIZE and 0 <= y_idx < GRID_SIZE:
                local_cAMP = camp_grid[x_idx, y_idx]
                update_cell_MG(cell, local_cAMP, DELTA_T)
        
        positions = np.array([cell.position for cell in cells])
        coords_diff = positions[:, None, :] - positions[None, :, :]
        coords_diff = (coords_diff + SPACE_SIZE/2) % SPACE_SIZE - SPACE_SIZE/2
        
        force_field = force_field_inbox(coords_diff, Req=R_EQ, R0=R_0,
                                         Frep=F_REP, Fadh=F_ADH,
                                         coeff_a=COEFF_CARRE, coeff_rep=COEFF_REP)
        
        fluctuations = (np.random.rand(*v0.shape) - 0.5) * FLUCTUATION_FACTOR
        displacement = (MU * force_field) * DELTA_T + (v0 + fluctuations) * directions * DELTA_T
        
        positions = positions + displacement
        positions = np.mod(positions, SPACE_SIZE)
        for i, cell in enumerate(cells):
            cell.position = positions[i]
            cell.direction = autovel(displacement[i], cell.direction, cell.tau, cell.noise, DELTA_T, cell.persistence)
        
        for cell in cells:
            data_log.append({
                'frame': iteration,
                'time': time_sim,
                'cell_id': cell.id,
                'pop_tag': cell.pop,
                'x': cell.position[0],
                'y': cell.position[1],
                'dir_x': cell.direction[0],
                'dir_y': cell.direction[1],
                'b': cell.b,
                'r_T': cell.r_T
            })
        
        if iteration % PLOT_INTERVAL == 0:
            plot_cells_and_fields(cells, camp_grid, pde_grid, iteration, time_sim,
                                  SPACE_SIZE, GRID_RESOLUTION, path_saving=output_path)
        
        directions = np.array([cell.direction for cell in cells])
        iteration += 1
        time_sim += DELTA_T
    
    df_log = pd.DataFrame(data_log)
    df_log.to_csv(os.path.join(output_path, "simulation_data.csv"), index=False)
    print("Simulation terminée. Résultats sauvegardés.")

if __name__ == "__main__":
    main()