#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation de cellules de Dictyostelium avec :
 - Dynamique FitzHugh–Nagumo (A, R)
 - Liaison récepteur-cAMP (L) avec désensibilisation/adaptation
 - Production pulsée de cAMP lors des oscillations
 - Diffusion globale et dégradation du cAMP
 - Dégradation locale supplémentaire (phosphodiestérase de surface)
 - Interactions mécaniques (adhésion/répulsion) et déplacement
 - Sauvegarde d'images pour visualiser le champ de cAMP et les positions cellulaires
 - Tracé pas à pas de la production de cAMP d'une cellule cible
"""

import math
import random
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Choix CPU ou GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
#                          PARAMÈTRES GLOBAUX
###############################################################################
SPACE_SIZE = 200.0      # Taille du domaine (μm)
DELTA_T    = 0.01       # Pas de temps (min)
TIME_SIMU  = 50.0       # Durée totale de la simulation (min)
R_EQ       = 4.0        # Rayon cellulaire d'équilibre (μm)
R_0        = 6.0        # Distance max d'interaction adhésive (μm)
F_REP      = 40.0       # Intensité de la répulsion
F_ADH      = 7.0        # Intensité de l'adhésion
MIN_DIST   = 2 * R_EQ   # Distance minimale de placement initial des cellules
N_CELLS    = 20         # Nombre de cellules
SAVE_EVERY = 100        # Sauvegarde d'image toutes les 100 itérations
SEED       = 42         # Pour la reproductibilité

random.seed(SEED)
torch.manual_seed(SEED)

###############################################################################
#                          CLASSES PRINCIPALES
###############################################################################
class CellAgent:
    """
    Représente une cellule Dictyostelium avec :
     - Dynamique interne FitzHugh–Nagumo (A, R)
     - Cinétique de liaison récepteur-cAMP (L)
     - Désensibilisation à forte exposition cAMP (adaptation)
     - Production pulsée de cAMP déclenchée par un seuil
    """

    def __init__(self, idx: int, position: torch.Tensor, velocity: torch.Tensor,
                 cell_params: dict, space_size: float):
        """
        Initialise la cellule.

        Args:
            idx (int): Identifiant unique de la cellule.
            position (torch.Tensor): Position (x, y) initiale (μm).
            velocity (torch.Tensor): Vitesse initiale (vx, vy) (μm/min).
            cell_params (dict): Paramètres internes (FitzHugh–Nagumo, liaison, adaptation,
                                production pulsée, etc.).
            space_size (float): Taille du domaine (pour conditions périodiques).
        """
        self.id = idx
        self.position = position.clone().to(device)
        self.velocity = velocity.clone().to(device)
        self.direction = torch.nn.functional.normalize(self.velocity, dim=0)
        self.cell_params = cell_params
        self.space_size = space_size

        # Dynamique FitzHugh–Nagumo
        self.A = torch.tensor(cell_params['A_init'], device=device, dtype=torch.float)
        self.R = torch.tensor(cell_params['R_init'], device=device, dtype=torch.float)
        self.L = torch.tensor(0.0, device=device, dtype=torch.float)  # Fraction de récepteurs liés [0,1]

        # Désensibilisation / adaptation
        self.receptor_sensitivity = torch.tensor(1.0, device=device, dtype=torch.float)

        # Pour la détection de pulse (production pulsée)
        self.last_A = self.A.clone()

        # Suivi de la production cumulée (optionnel)
        self.camp_production_cumulative = 0.0

    def update_state(self, local_camp: float, dt: float):
        """
        Met à jour l'état interne (L, adaptation, FitzHugh–Nagumo) sur un pas de temps dt.
        """
        p = self.cell_params

        # Adaptation / désensibilisation
        if local_camp > p['cAMP_adapt_threshold']:
            target = p['adaptation_min']
        else:
            target = 1.0
        dr = p['adaptation_rate'] * (target - self.receptor_sensitivity) * dt
        self.receptor_sensitivity += dr
        self.receptor_sensitivity = torch.clamp(self.receptor_sensitivity, 0.0, 1.0)

        # Cinétique de liaison L
        kon, koff = p['kon'], p['koff']
        dL = (kon * local_camp * (1.0 - self.L) - koff * self.L) * self.receptor_sensitivity * dt
        self.L += dL
        self.L = torch.clamp(self.L, 0.0, 1.0)

        # FitzHugh–Nagumo
        a_val = p['a']
        Kd = p['Kd']
        gamma = p['gamma']
        epsilon = p['epsilon']
        sigma = p['sigma']
        noise_flag = p.get('noise', True)
        I_S = a_val * torch.log1p(self.L / Kd)
        dA = (self.A - (self.A ** 3)/3.0 - self.R + I_S)
        if noise_flag:
            dA += sigma * math.sqrt(dt) * torch.randn((), device=device)
        self.A += dA * dt
        dR = (self.A - gamma * self.R + local_camp) * epsilon
        self.R += dR * dt

    def produce_camp_pulse(self, dt: float) -> float:
        """
        Déclenche une production pulsée de cAMP lorsque A franchit le seuil.
        
        Returns:
            float: Quantité de cAMP produite lors de ce pulse.
        """
        p = self.cell_params
        pulse = 0.0
        if (self.last_A < p['pulse_threshold']) and (self.A >= p['pulse_threshold']):
            pulse = p['pulse_strength']
        self.last_A = self.A.clone()
        return pulse

    def __repr__(self):
        return f"<Cell {self.id} : pos=({self.position[0]:.2f},{self.position[1]:.2f}) A={self.A:.2f} R={self.R:.2f} L={self.L:.2f}>"

class cAMPField:
    """
    Gère le champ de cAMP par diffusion et dégradation (globale et locale).
    La dégradation locale simule l'action de la phosphodiestérase de surface.
    """

    def __init__(self, space_size: float, grid_resolution: float, params: dict):
        """
        Initialise le champ de cAMP sur une grille 2D.

        Args:
            space_size (float): Taille du domaine (μm).
            grid_resolution (float): Taille d'une case (μm).
            params (dict): Contient D_cAMP, aPDE et local_pde_strength.
        """
        self.space_size = space_size
        self.grid_resolution = grid_resolution
        self.grid_size = int(space_size / grid_resolution)
        self.dx = grid_resolution
        self.D_cAMP = params['D_cAMP']
        self.aPDE = params['aPDE']
        self.local_pde_strength = params['local_pde_strength']

        self.signal = torch.zeros((self.grid_size, self.grid_size), device=device)

        # Noyau gaussien pour dépôt/dégradation locale
        self.prod_radius = 3
        self.kernel_size = 2 * self.prod_radius + 1
        sigma = self.prod_radius / 2.0
        kernel = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float32)
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                dx_val = i - self.prod_radius
                dy_val = j - self.prod_radius
                kernel[i, j] = math.exp(-(dx_val**2 + dy_val**2) / (2 * sigma**2))
        kernel /= np.sum(kernel)
        self.kernel = torch.tensor(kernel, device=device)

    def get_signal_at_position(self, pos: torch.Tensor) -> float:
        """
        Retourne la concentration de cAMP à la position donnée.
        """
        x_idx = int(pos[0].item() / self.grid_resolution) % self.grid_size
        y_idx = int(pos[1].item() / self.grid_resolution) % self.grid_size
        return self.signal[x_idx, y_idx].item()

    def compute_laplacian_9point(self) -> torch.Tensor:
        """
        Calcule le Laplacien via un stencil 9-points.
        """
        S = self.signal
        dx2 = self.dx**2
        S_up    = torch.roll(S, shifts=+1, dims=0)
        S_down  = torch.roll(S, shifts=-1, dims=0)
        S_left  = torch.roll(S, shifts=+1, dims=1)
        S_right = torch.roll(S, shifts=-1, dims=1)
        S_upleft    = torch.roll(S_up,    shifts=+1, dims=1)
        S_upright   = torch.roll(S_up,    shifts=-1, dims=1)
        S_downleft  = torch.roll(S_down,  shifts=+1, dims=1)
        S_downright = torch.roll(S_down,  shifts=-1, dims=1)
        lap = (
            -20.0 * S
            + 4.0 * (S_up + S_down + S_left + S_right)
            + 2.0 * (S_upleft + S_upright + S_downleft + S_downright)
        ) / (6.0 * dx2)
        return lap

    def update(self, cells: list, dt: float, current_time: float = None, production_trace: dict = None):
        """
        Met à jour le champ cAMP en :
          1) Récoltant la production pulsée de chaque cellule,
          2) Appliquant une dégradation locale via la PDE de surface,
          3) Diffusant et dégradant globalement le champ.
          
        Args:
            cells (list): Liste des cellules.
            dt (float): Pas de temps.
            current_time (float, optionnel): Temps courant (pour log).
            production_trace (dict, optionnel): Dictionnaire pour tracer la production d'une cellule cible.
                Doit contenir les clés 'target', 'trace' et 'time'.
        """
        A_grid = torch.zeros_like(self.signal)
        PDE_grid = torch.zeros_like(self.signal)

        for cell in cells:
            pulse_amount = cell.produce_camp_pulse(dt)
            # Si on souhaite tracer la production pour une cellule cible
            if production_trace is not None and cell.id == production_trace['target']:
                if current_time is not None:
                    production_trace['time'].append(current_time)
                production_trace['trace'].append(pulse_amount)
            if pulse_amount > 0:
                x_idx = int(cell.position[0].item() / self.grid_resolution) % self.grid_size
                y_idx = int(cell.position[1].item() / self.grid_resolution) % self.grid_size
                for dx_idx in range(-self.prod_radius, self.prod_radius+1):
                    for dy_idx in range(-self.prod_radius, self.prod_radius+1):
                        xx = (x_idx + dx_idx) % self.grid_size
                        yy = (y_idx + dy_idx) % self.grid_size
                        w = self.kernel[dx_idx + self.prod_radius, dy_idx + self.prod_radius]
                        A_grid[xx, yy] += pulse_amount * w

            # Dégradation locale (PDE de surface)
            pde_strength = self.local_pde_strength
            x_idx = int(cell.position[0].item() / self.grid_resolution) % self.grid_size
            y_idx = int(cell.position[1].item() / self.grid_resolution) % self.grid_size
            for dx_idx in range(-self.prod_radius, self.prod_radius+1):
                for dy_idx in range(-self.prod_radius, self.prod_radius+1):
                    xx = (x_idx + dx_idx) % self.grid_size
                    yy = (y_idx + dy_idx) % self.grid_size
                    w = self.kernel[dx_idx + self.prod_radius, dy_idx + self.prod_radius]
                    PDE_grid[xx, yy] -= pde_strength * w

        laplacian = self.compute_laplacian_9point()
        self.signal += dt * (
            self.D_cAMP * laplacian
            - self.aPDE * self.signal
            + A_grid
            + PDE_grid * self.signal
        )
        self.signal = torch.clamp(self.signal, min=0.0)
        if torch.isnan(self.signal).any() or torch.isinf(self.signal).any():
            print("NaN/Inf détecté dans le champ cAMP, arrêt.")
            sys.exit(1)

###############################################################################
#                  FONCTIONS UTILES POUR LA MOTILITÉ
###############################################################################
def force_field_inbox(coordinates_diff: torch.Tensor, distances: torch.Tensor,
                      Req: float, R0: float, Frep: float, Fadh: float) -> torch.Tensor:
    """
    Calcule la force nette subie par chaque cellule via répulsion et adhésion pair-à-pair.
    """
    Rlim = 1e-6
    R = torch.where(distances > Rlim, distances, torch.full_like(distances, Rlim))
    force_adh = -((Fadh/(R0 - Req)) * R - (Fadh*Req/(R0 - Req)))
    force_rep = -(Frep * (1.0/Req - 1.0/R))
    force = torch.zeros_like(R)
    in_adh_zone = (R > Req) & (R < R0)
    in_rep_zone = (R <= Req)
    force = torch.where(in_adh_zone, force_adh, force)
    force = torch.where(in_rep_zone, force_rep, force)
    norm_diff = torch.nn.functional.normalize(coordinates_diff, dim=2)
    force_field = torch.sum(force[:, :, None] * norm_diff, dim=1)
    return force_field

def autovel(dX: torch.Tensor, old_dir: torch.Tensor, tau: float, noise: float, dt: float) -> torch.Tensor:
    """
    Met à jour la direction de déplacement d'une cellule.
    """
    dX_n = torch.nn.functional.normalize(dX, dim=1)
    theta_new = torch.atan2(dX_n[:, 1], dX_n[:, 0])
    rnd = (2 * math.pi * (torch.rand(len(dX), 1, device=device) - 0.5)) * noise * math.sqrt(dt)
    theta_final = theta_new + rnd.squeeze(1) * dt / tau
    new_dir = torch.stack((torch.cos(theta_final), torch.sin(theta_final)), dim=1)
    return new_dir

def save_frame(camp_field: cAMPField, cells: list, iteration: int, time: float):
    """
    Sauvegarde une image montrant le champ de cAMP et les positions des cellules.
    """
    plt.figure(figsize=(8, 6))
    field = camp_field.signal.cpu().numpy()
    plt.imshow(field.T, origin='lower', extent=[0, camp_field.space_size, 0, camp_field.space_size],
               cmap='viridis')
    xs = [cell.position[0].item() for cell in cells]
    ys = [cell.position[1].item() for cell in cells]
    plt.scatter(xs, ys, color='red', s=50, label="Cells")
    plt.title(f"Time: {time:.2f} min, Iteration: {iteration}")
    plt.xlabel("X (μm)")
    plt.ylabel("Y (μm)")
    plt.legend()
    plt.colorbar(label="cAMP concentration")
    os.makedirs("frames", exist_ok=True)
    plt.savefig(f"frames/frame_{iteration:04d}.png")
    plt.close()

###############################################################################
#                      SCRIPT PRINCIPAL DE SIMULATION
###############################################################################
def main():
    # 1) Paramètres internes pour les cellules
    cell_params = {
        'A_init':  -1.0,
        'R_init':  -1.0,
        'a': 0.4,
        'gamma': 0.1,
        'epsilon': 0.1,
        'sigma': 0.01,
        'Kd': 0.5,
        'noise': True,
        'kon': 0.5,
        'koff': 1.0,
        'adaptation_rate': 0.1,
        'cAMP_adapt_threshold': 1.5,
        'adaptation_min': 0.5,
        'pulse_threshold':  1.0,
        'pulse_strength':  400.0,
    }

    # 2) Paramètres du champ cAMP
    camp_params = {
        'D_cAMP': 1.0,
        'aPDE': 0.5,
        'local_pde_strength': 0.2,
    }

    # 3) Création de la population de cellules
    cells = []
    placed_positions = []
    max_attempts = 100

    for i_cell in range(N_CELLS):
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            pos = torch.rand(2, device=device) * SPACE_SIZE
            conflict = False
            for p0 in placed_positions:
                if torch.norm(pos - p0) < MIN_DIST:
                    conflict = True
                    break
            if not conflict:
                placed_positions.append(pos)
                direction = torch.nn.functional.normalize(torch.rand(2, device=device) - 0.5, dim=0)
                speed = 0.5
                vel = direction * speed
                cell = CellAgent(i_cell, pos, vel, cell_params, SPACE_SIZE)
                cells.append(cell)
                break

    print(f"{len(cells)} cellules placées sur {N_CELLS} demandées.")

    # 4) Création du champ de cAMP
    grid_resolution = 1.0
    camp_field = cAMPField(SPACE_SIZE, grid_resolution, camp_params)

    # 5) Initialisation du trace de production pour la cellule cible (ici id=0)
    production_trace = {'target': 0, 'trace': [], 'time': []}

    # 6) Boucle principale de simulation
    time = 0.0
    iteration = 0

    while time < TIME_SIMU:
        # A) Mise à jour de l'état interne de chaque cellule
        for cell in cells:
            loc_val = camp_field.get_signal_at_position(cell.position)
            cell.update_state(loc_val, DELTA_T)

        # B) Mise à jour du champ cAMP (production pulsée, diffusion, dégradation)
        camp_field.update(cells, DELTA_T, current_time=time, production_trace=production_trace)

        # C) Calcul des interactions mécaniques et déplacement
        positions = torch.stack([c.position for c in cells])
        coords_diff = positions[:, None, :] - positions[None, :, :]
        coords_diff = torch.remainder(coords_diff + SPACE_SIZE/2, SPACE_SIZE) - SPACE_SIZE/2
        dists = torch.norm(coords_diff, dim=2)
        forces = force_field_inbox(coords_diff, dists, R_EQ, R_0, F_REP, F_ADH)
        displacement = forces * DELTA_T
        new_dirs = autovel(displacement, torch.stack([c.direction for c in cells]), tau=5.0, noise=0.3, dt=DELTA_T)
        for i_c, cell in enumerate(cells):
            cell.position += displacement[i_c]
            cell.position = torch.remainder(cell.position, SPACE_SIZE)
            cell.direction = new_dirs[i_c]

        # Enregistrement périodique d'images
        if iteration % SAVE_EVERY == 0:
            save_frame(camp_field, cells, iteration, time)

        time += DELTA_T
        iteration += 1

    print("Fin de la simulation.")

    # Tracé de la production de cAMP pour la cellule cible
    plt.figure(figsize=(8, 4))
    plt.plot(production_trace['time'], production_trace['trace'], marker='o')
    plt.xlabel("Temps (min)")
    plt.ylabel("Production de cAMP (pulse)")
    plt.title("Production de cAMP pas à pas pour la cellule cible (id=0)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()