#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended model of cAMP/PDE signaling in Dictyostelium inspired by Halloy et al. (1998).
- Cells produce cAMP basally and via a positive feedback mechanism.
- When local cAMP exceeds a PDE_threshold, the production of PDE becomes proportional to (cAMP - PDE_threshold).
- Extracellular cAMP is degraded at a rate enhanced by the local PDE concentration.
- Each cell’s internal dynamics are modeled with receptor state (r_T), intracellular cAMP (b), 
  and fractions of active Gs (gs) and Gi (gi) proteins.
- Diffusion of cAMP and PDE is handled on a spatial grid with periodic boundary conditions.
- At the end, a time-series plot is generated for a selected cell.
Author: souchaud
"""

import math
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device for torch operations:", device)

#####################################
# Global plotting functions
#####################################

def plot_environment(cells, space_size, req, path_saving, iteration):
    """Plot the spatial distribution of cells (cells are immobile)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, space_size)
    ax.set_ylim(0, space_size)
    
    x = [cell.position[0].item() for cell in cells]
    y = [cell.position[1].item() for cell in cells]
    colors = ["blue" if cell.pop=="Population 1" else "red" for cell in cells]
    
    ax.scatter(x, y, s=3, color=colors, alpha=0.5)
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.axis('off')
    
    filename = os.path.join(path_saving, f"image_{iteration}.png")
    plt.savefig(filename, bbox_inches='tight', dpi=400, pad_inches=0)
    plt.close(fig)

def plot_combined_state(cells, camp_field, pde_field, SPACE_SIZE: float, iteration: float, PATH: str):
    """Plot a combined figure: cell positions, extracellular cAMP, and PDE fields."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)
    extent = [0, SPACE_SIZE, 0, SPACE_SIZE]
    
    # Cell positions
    axes[0].set_xlim(0, SPACE_SIZE)
    axes[0].set_ylim(0, SPACE_SIZE)
    axes[0].set_aspect('equal', adjustable='box')
    x = [cell.position[0].item() for cell in cells]
    y = [cell.position[1].item() for cell in cells]
    colors = ["blue" if cell.pop=="Population 1" else "red" for cell in cells]
    axes[0].scatter(x, y, s=30, color=colors, alpha=0.8)
    axes[0].set_title("Positions des cellules")
    axes[0].set_xlabel("X (μm)")
    axes[0].set_ylabel("Y (μm)")
    
    # Extracellular cAMP field
    im1 = axes[1].imshow(
        camp_field.camp_grid.cpu().numpy().T,
        origin='lower',
        extent=extent,
        cmap='viridis',
        alpha=0.8,
        vmin=0, vmax=0.5
    )
    axes[1].set_title(f"Champ de cAMP (itération {iteration})")
    axes[1].set_xlabel("X (μm)")
    axes[1].set_ylabel("Y (μm)")
    fig.colorbar(im1, ax=axes[1], shrink=0.6, aspect=20, label="cAMP")
    
    # Extracellular PDE field
    im2 = axes[2].imshow(
        pde_field.PDE_grid.cpu().numpy().T,
        origin='lower',
        extent=extent,
        cmap='plasma',
        alpha=0.8,
        vmin=0, vmax=0.01
    )
    axes[2].set_title(f"Champ de PDE (itération {iteration})")
    axes[2].set_xlabel("X (μm)")
    axes[2].set_ylabel("Y (μm)")
    fig.colorbar(im2, ax=axes[2], shrink=0.6, aspect=20, label="PDE")
    
    filename = os.path.join(PATH, f"combined_{iteration}.png")
    plt.savefig(filename, bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close()

def plot_cell_time_series_four(data_frame, cell_id, path):
    """
    For a given cell (cell_id), plot four time series in separate subplots:
      1) Local extracellular cAMP concentration.
      2) Local extracellular PDE concentration.
      3) Production of cAMP.
      4) Production of PDE.
    """
    cell_data = data_frame[data_frame['cell_id'] == cell_id].copy()
    cell_data.sort_values('time', inplace=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    axs[0, 0].plot(cell_data['time'], cell_data['local_cAMP'], color='blue')
    axs[0, 0].set_title('Concentration locale de cAMP')
    axs[0, 0].set_xlabel('Temps (min)')
    axs[0, 0].set_ylabel('cAMP')
    axs[0, 0].grid(True)
    
    axs[0, 1].plot(cell_data['time'], cell_data['local_PDE'], color='red')
    axs[0, 1].set_title('Concentration locale de PDE')
    axs[0, 1].set_xlabel('Temps (min)')
    axs[0, 1].set_ylabel('PDE')
    axs[0, 1].grid(True)
    
    axs[1, 0].plot(cell_data['time'], cell_data['cAMP_prod'], color='cyan', linestyle='--')
    axs[1, 0].set_title('Production de cAMP')
    axs[1, 0].set_xlabel('Temps (min)')
    axs[1, 0].set_ylabel('Production cAMP')
    axs[1, 0].grid(True)
    
    axs[1, 1].plot(cell_data['time'], cell_data['PDE_prod'], color='magenta', linestyle='--')
    axs[1, 1].set_title('Production de PDE')
    axs[1, 1].set_xlabel('Temps (min)')
    axs[1, 1].set_ylabel('Production PDE')
    axs[1, 1].grid(True)
    
    fig.suptitle(f"Évolution temporelle pour la cellule {cell_id}", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(path, f"cell_{cell_id}_timeseries.png"), bbox_inches='tight', dpi=300, pad_inches=0)
    plt.show()
    plt.close()

#######################################
# Extended internal cell update function
#######################################

def update_cell_extended(cell, local_cAMP, dt):
    """
    Extended update function for the cell's internal state, based on the model by Martiel & Goldbeter (1987) and further refined
    by Halloy et al. (1998). This version includes additional variables for the active Gs and Gi proteins.
    
    The equations used here are a simplified version of the extended model:
    
      dr_T/dt = -r_T * f1(local_cAMP) + (1 - r_T) * f2(local_cAMP)
      dgs/dt = J1 * x(r_T, local_cAMP) * (1 - gs) - k3' * gs
      dgi/dt = J2 * y(r_T, local_cAMP) * (1 - gi) - k4' * gi
      db/dt  = j_q * (gs - gi) - (k_i + k_t) * b
      
    where f1 and f2 are functions (using a Hill form) that control receptor desensitization and resensitization,
    and x and y describe the activation of Gs and Gi.
    
    The parameter values below are indicative and should be calibrated:
      - k1 = 0.5, k2 = 0.5 for f1; k_minus1 = 0.1, k_minus2 = 0.1 for f2.
      - J1 = 25, k3' = 5 for Gs dynamics.
      - J2 = 5, k4' = 0.1 for Gi dynamics.
      - j_q = 1.0, total_degradation_rate = 0.6 for intracellular cAMP.
    
    Note: This is a simplified version and may require further refinement to fully capture the biological oscillations.
    """
    # Define parameters for receptor dynamics
    k1, k2 = 0.5, 0.5
    k_minus1, k_minus2 = 0.1, 0.1
    # Hill functions for receptor inactivation and reactivation
    f1 = (k1 + k2 * local_cAMP) / (1 + local_cAMP)
    f2 = (k_minus1 + k_minus2 * local_cAMP) / (1 + local_cAMP)
    
    drT = -cell.r_T * f1 + (1 - cell.r_T) * f2
    cell.r_T += drT * dt
    
    # Update Gs and Gi dynamics
    # x = r_T * local_cAMP / (1 + local_cAMP)
    # y = (1 - r_T) * local_cAMP / (1 + local_cAMP)
    x_val = cell.r_T * local_cAMP / (1 + local_cAMP)
    y_val = (1 - cell.r_T) * local_cAMP / (1 + local_cAMP)
    J1, k3_prime = 25.0, 5.0
    J2, k4_prime = 5.0, 0.1
    dgs = J1 * x_val * (1 - cell.gs) - k3_prime * cell.gs
    dgi = J2 * y_val * (1 - cell.gi) - k4_prime * cell.gi
    cell.gs += dgs * dt
    cell.gi += dgi * dt
    
    # Update intracellular cAMP (b)
    j_q = 1.0
    total_deg = 0.6  # effective degradation rate (sum of ki and kt)
    db = j_q * (cell.gs - cell.gi) - total_deg * cell.b
    cell.b += db * dt

#############################################
# Extended CellAgent class with new variables
#############################################

class ExtendedCellAgent(CellAgent):
    def __init__(self, pop, position, tau, noise, qs=1.0, kt=0.5, f1=0.1, f2=0.5,
                 PDE_threshold=0.4, PDE_rate=5.0):
        super().__init__(pop, position, tau, noise)
        # Use parameters from parent for b and r_T
        self.qs = qs
        self.kt = kt
        self.f1 = f1
        self.f2 = f2
        self.PDE_threshold = PDE_threshold
        self.PDE_rate = PDE_rate
        # New variables for the extended model
        self.gs = 0.0  # Fraction of active Gs proteins
        self.gi = 0.0  # Fraction of active Gi proteins

####################################
# Population using ExtendedCellAgent
####################################

class ExtendedPopulation(Population):
    def initialize_cells(self):
        for i in range(self.num_cells):
            candidate = torch.rand(2, device=device) * self.space_size
            # Instantiate ExtendedCellAgent with default or provided parameters
            self.cells.append(ExtendedCellAgent(self.pop_tag, candidate, self.tau, self.noise,
                                                 qs=1.0, kt=0.5, f1=0.1, f2=0.5,
                                                 PDE_threshold=0.4, PDE_rate=5.0))

class CellAgent:
    _id_counter = 0  # Compteur unique pour chaque cellule
    
    def __init__(self, pop, position, tau, noise):
        self.id = CellAgent._id_counter
        CellAgent._id_counter += 1
        self.pop = pop
        self.position = position.clone().to(device)  # Position fixe
        # Variables internes du modèle Martiel & Goldbeter
        self.b = 0.0    # concentration d'AMPc intracellulaire
        self.r_T = 0.1  # fraction de récepteurs activés
        # Paramètres internes
        self.qs = 1.0   # taux de production d'AMPc intracellulaire
        self.kt = 0.5   # taux de dégradation d'AMPc intracellulaire
        self.f1 = 0.1   # facteur de désactivation des récepteurs
        self.f2 = 0.5   # facteur de réactivation des récepteurs
        # Paramètres pour la production de PDE
        self.PDE_threshold = 0.4  # seuil de cAMP pour déclencher la production de PDE
        # Ici, au lieu de produire un taux fixe, nous souhaitons que la production soit proportionnelle à (local_cAMP - PDE_threshold)
        self.PDE_rate = 5       # facteur multiplicatif pour la production de PDE
        self.PDE = 0.0            # quantité de PDE produite par la cellule (initialement 0)
        # Paramètres pour éventuel bruit (non utilisé ici)
        self.tau = tau
        self.noise = noise

class Population:
    def __init__(self, num_cells, space_size, pop_tag, tau, noise):
        self.num_cells = num_cells
        self.space_size = space_size
        self.pop_tag = pop_tag
        self.tau = tau
        self.noise = noise
        self.cells = []
        self.initialize_cells()
    
    def initialize_cells(self):
        for i in range(self.num_cells):
            candidate = torch.rand(2, device=device) * self.space_size
            self.cells.append(CellAgent(self.pop_tag, candidate, self.tau, self.noise))

class cAMP:
    def __init__(self, space_size, grid_resolution, rho, alpha0, D, J,
                 production_threshold=0.3, extra_production_rate=0.05,
                 k_PDE=1.0, production_inhibition_threshold=0.01):
        """
        Paramètres :
          - space_size : taille du domaine (μm)
          - grid_resolution : taille d'une case de la grille (μm)
          - rho : coefficient de production basale de cAMP par cellule
          - alpha0 : facteur de normalisation de la production basale
          - D : coefficient de diffusion du cAMP (μm²/min)
          - J : taux de dégradation global du cAMP (min⁻¹)
          - production_threshold : seuil de cAMP pour rétroaction positive
          - extra_production_rate : production additionnelle si le seuil est dépassé
          - k_PDE : coefficient augmentant la dégradation du cAMP en fonction de la PDE locale
          - production_inhibition_threshold : si la concentration locale de PDE dépasse ce seuil, la production de cAMP est bloquée
        """
        self.space_size = space_size
        self.grid_resolution = grid_resolution
        self.grid_size = int(space_size / grid_resolution)
        self.rho = rho
        self.alpha0 = alpha0
        self.D = D
        self.J = J
        self.production_threshold = production_threshold
        self.extra_production_rate = extra_production_rate
        self.k_PDE = k_PDE
        self.production_inhibition_threshold = production_inhibition_threshold
        self.camp_grid = torch.zeros((self.grid_size, self.grid_size), device=device)
    
    def update(self, cells, pde_field, dt):
        # Production locale de cAMP par les cellules
        for cell in cells:
            x_idx = int(cell.position[0].item() / self.grid_resolution)
            y_idx = int(cell.position[1].item() / self.grid_resolution)
            x_idx = min(x_idx, self.grid_size - 1)
            y_idx = min(y_idx, self.grid_size - 1)
            # Lire la concentration locale actuelle de cAMP et de PDE
            local_conc = self.camp_grid[x_idx, y_idx].item()
            local_PDE = pde_field.PDE_grid[x_idx, y_idx].item()
            # Calcul de la production de cAMP pour ce pas de temps :
            if local_PDE > self.production_inhibition_threshold:
                prod = 0.0
            else:
                prod = self.rho * self.alpha0 * dt
                if local_conc > self.production_threshold:
                    prod += self.extra_production_rate * dt
            self.camp_grid[x_idx, y_idx] += prod
        # Diffusion avec conditions périodiques
        self.camp_grid += self.D * self.laplacian(self.camp_grid) * dt
        # Dégradation du cAMP, renforcée localement par la PDE
        self.camp_grid -= (self.J + self.k_PDE * pde_field.PDE_grid) * self.camp_grid * dt
    
    def laplacian(self, grid):
        # Conditions aux bords périodiques via torch.roll
        return (torch.roll(grid, shifts=1, dims=0) +
                torch.roll(grid, shifts=-1, dims=0) +
                torch.roll(grid, shifts=1, dims=1) +
                torch.roll(grid, shifts=-1, dims=1) -
                4 * grid)

class PDE:
    def __init__(self, space_size, grid_resolution, D, decay):
        """
        Paramètres :
          - space_size : taille du domaine (μm)
          - grid_resolution : taille d'une case de la grille (μm)
          - D : coefficient de diffusion de la PDE (μm²/min)
          - decay : taux de dégradation de la PDE (min⁻¹)
        """
        self.space_size = space_size
        self.grid_resolution = grid_resolution
        self.grid_size = int(space_size / grid_resolution)
        self.D = D
        self.decay = decay
        self.PDE_grid = torch.zeros((self.grid_size, self.grid_size), device=device)
    
    def update(self, cells, dt):
        # Production de PDE par les cellules
        for cell in cells:
            x_idx = int(cell.position[0].item() / self.grid_resolution)
            y_idx = int(cell.position[1].item() / self.grid_resolution)
            x_idx = min(x_idx, self.grid_size - 1)
            y_idx = min(y_idx, self.grid_size - 1)
            self.PDE_grid[x_idx, y_idx] += cell.PDE * dt
        # Diffusion avec conditions périodiques
        self.PDE_grid += self.D * self.laplacian(self.PDE_grid) * dt
        # Dégradation de la PDE
        self.PDE_grid -= self.decay * self.PDE_grid * dt
    
    def laplacian(self, grid):
        return (torch.roll(grid, shifts=1, dims=0) +
                torch.roll(grid, shifts=-1, dims=0) +
                torch.roll(grid, shifts=1, dims=1) +
                torch.roll(grid, shifts=-1, dims=1) -
                4 * grid)

########################################
# Main simulation function (extended)
########################################

def main():
    # =======================
    # Simulation parameters
    # =======================
    SPACE_SIZE = 50         # μm
    TIME_SIMU = 100         # minutes
    DELTA_T = 0.01          # minutes per time step
    PLOT_INTERVAL = 100     # plotting interval (iterations)

    # Number of cells
    N_CELLS = 250
    print(f"{N_CELLS} cellules")

    # Parameters for populations
    TAU_POP = 5
    NOISE_POP = 8

    # Create two populations using the extended cell model
    pop1 = ExtendedPopulation(num_cells=int(N_CELLS/2), space_size=SPACE_SIZE,
                              pop_tag="Population 1", tau=TAU_POP, noise=NOISE_POP)
    pop2 = ExtendedPopulation(num_cells=int(N_CELLS/2), space_size=SPACE_SIZE,
                              pop_tag="Population 2", tau=TAU_POP, noise=NOISE_POP)
    cells = pop1.cells + pop2.cells

    # Set up save directory
    PATH = '/Users/souchaud/Desktop/simu/simplified/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    else:
        print("WARNING: Le dossier existe déjà!")
    
    # Create extracellular fields for cAMP and PDE
    camp = cAMP(
        space_size=SPACE_SIZE,         # Domain size (μm)
        grid_resolution=1,             # Grid cell size (μm)
        rho=0.5,                       # Basal production coefficient for cAMP
        alpha0=1.0,                    # Normalization factor for basal production
        D=2.0,                         # Diffusion coefficient for cAMP (μm²/min)
        J=0.05,                        # Global degradation rate for cAMP (min⁻¹)
        production_threshold=0.2,      # Threshold for extra production (nM)
        extra_production_rate=0.05,      # Additional production rate (nM/min)
        k_PDE=10.0,                    # PDE coupling coefficient for cAMP degradation (min⁻¹)
        production_inhibition_threshold=0.06  # PDE concentration threshold to block cAMP production
    )
    pde_field = PDE(
        space_size=SPACE_SIZE,         # Domain size (μm)
        grid_resolution=1,             # Grid cell size (μm)
        D=0.2,                         # Diffusion coefficient for PDE (μm²/min)
        decay=0.005                    # PDE degradation rate (min⁻¹)
    )
    
    # Save initial cell positions
    plot_environment(cells, space_size=SPACE_SIZE, req=1.1, path_saving=PATH, iteration=0)

    time = 0.0
    iteration = 1
    data_list = []

    # Main simulation loop
    while time < TIME_SIMU:
        for cell in cells:
            x_idx = int(cell.position[0].item() / camp.grid_resolution)
            y_idx = int(cell.position[1].item() / camp.grid_resolution)
            x_idx = min(x_idx, camp.grid_size - 1)
            y_idx = min(y_idx, camp.grid_size - 1)
            
            local_cAMP = camp.camp_grid[x_idx, y_idx].item()
            local_PDE = pde_field.PDE_grid[x_idx, y_idx].item()
            
            # Update extended internal cell dynamics using the new function
            update_cell_extended(cell, local_cAMP, DELTA_T)
            
            # Production of PDE is now proportional to (local_cAMP - PDE_threshold) if above threshold:
            if local_cAMP > cell.PDE_threshold:
                cell.PDE = cell.PDE_rate * (local_cAMP - cell.PDE_threshold)
            else:
                cell.PDE = 0.0
            
            # Compute production of cAMP for this cell (as in camp.update)
            if local_PDE > camp.production_inhibition_threshold:
                cAMP_prod = 0.0
            else:
                cAMP_prod = camp.rho * camp.alpha0 * DELTA_T
                if local_cAMP > camp.production_threshold:
                    cAMP_prod += camp.extra_production_rate * DELTA_T
            
            PDE_prod = cell.PDE * DELTA_T
            
            data_list.append({
                'time': time,
                'cell_id': cell.id,
                'pop_tag': cell.pop,
                'x': cell.position[0].item(),
                'y': cell.position[1].item(),
                'local_cAMP': local_cAMP,
                'local_PDE': local_PDE,
                'cAMP_prod': cAMP_prod,
                'PDE_prod': PDE_prod,
                'b': cell.b,
                'r_T': cell.r_T,
                'gs': cell.gs,
                'gi': cell.gi
            })
        
        # Update extracellular fields: PDE then cAMP
        pde_field.update(cells, DELTA_T)
        camp.update(cells, pde_field, DELTA_T)
        
        time += DELTA_T
        iteration += 1
        
        if iteration % PLOT_INTERVAL == 0:
            plot_combined_state(cells, camp, pde_field, SPACE_SIZE, iteration, PATH)
        
        if iteration % 1000 == 0:
            print(f"Temps simulé : {time:.2f} minutes")
    
    # Save simulation data
    data_frame = pd.DataFrame(data_list)
    csv_filename = os.path.join(PATH, "simulation_data.csv")
    data_frame.to_csv(csv_filename, index=False)
    print("Simulation terminée. Données sauvegardées.")
    
    # Plot time series for a given cell (e.g., cell_id = 0)
    plot_cell_time_series_four(data_frame, cell_id=0, path=PATH)

if __name__ == "__main__":
    main()