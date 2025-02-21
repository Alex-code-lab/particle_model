#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation du modèle de particules avec diffusion de cAMP, dynamique cellulaire de type FitzHugh–Nagumo
et cinétique de liaison récepteur-cAMP.

Ce script simule l'interaction entre des cellules dans un domaine 2D. 
Les cellules interagissent via des forces d'adhésion et de répulsion et communiquent grâce à la production et
diffusion du cAMP. Chaque cellule possède une dynamique interne décrite par :

    1. La cinétique de liaison des récepteurs :
         dL/dt = kon * [cAMP] * (1 - L) - koff * L
       où L ∈ [0, 1] est la fraction des récepteurs liés.

    2. Le modèle FitzHugh–Nagumo (après activation, c'est-à-dire lorsque L dépasse un seuil) :
         dA/dt = A - A³/3 - R + I_S   avec I_S = a * log(1 + L/Kd)
         dR/dt = ε * (A - γ R + [cAMP])
       Les variables A (activateur) et R (répresseur) modélisent l’oscillation cellulaire.

Les interactions entre cellules (force nette) sont calculées de la façon suivante :

  - Pour une distance d entre Req et R0 (adhésion) :
        F_adhesion(d) = -[(Fadh/(R0-Req)) * d - (Fadh*Req/(R0-Req))]
  - Pour d ≤ Req (répulsion) :
        F_repulsion(d) = -Frep * coeff_rep * (1/Req - 1/d)

Unités utilisées :
    - Position : μm
    - Temps : min
    - Diffusion : D_cAMP en μm²/min
    - Dégradation : aPDE en min⁻¹
    - Force : unités arbitraires (a.u.)
"""

import math
import os
import sys
import random
import torch             # Calculs tensoriels (CPU/GPU)
import matplotlib.pyplot as plt  # Visualisation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd      # Sauvegarde des données
import numpy as np       # Opérations mathématiques
import trackpy as tp     # Suivi de particules (non utilisé ici)
from scipy.signal import find_peaks

# =============================================================================
# Définition du device (GPU si disponible, sinon CPU)
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device for torch operations:", device)

# =============================================================================
# Fonctions utilitaires de calcul et de mise à jour
# =============================================================================

def force_field_inbox(coordinates_diff: torch.Tensor, distances: torch.Tensor, Req: float, R0: float,
                        Frep: float, Fadh: float, coeff_a: float = None, coeff_rep: float = 1.0) -> torch.Tensor:
    """
    Calcule la force nette agissant sur chaque cellule à partir des interactions pair à pair.

    Les forces appliquées sont définies par :
      - Si Req < d < R0 (zone d'adhésion) :
            F(d) = -[(Fadh/(R0-Req)) * d - (Fadh*Req/(R0-Req))]
      - Si d ≤ Req (zone de répulsion) :
            F(d) = -Frep * coeff_rep * (1/Req - 1/d)

    Paramètres:
        coordinates_diff (torch.Tensor): Différence de positions [N, N, 2] entre paires de cellules.
        distances (torch.Tensor): Matrice des distances (d) correspondantes.
        Req (float): Rayon d'équilibre (μm).
        R0 (float): Rayon maximal d'interaction (μm).
        Frep (float): Intensité de la force de répulsion (a.u.).
        Fadh (float): Intensité de la force d'adhésion (a.u.).
        coeff_a (float, optionnel): Coefficient pour force quadratique (non utilisé).
        coeff_rep (float, optionnel): Coefficient modifiant la force répulsive.
    
    Retourne:
        torch.Tensor: Force nette sur chaque cellule (dimensions [N, 2]).
    """
    Rlim = 1e-6  # Pour éviter une division par zéro
    R = torch.norm(coordinates_diff, dim=2)
    R = torch.where(R > Rlim, R, torch.full_like(R, Rlim))
    
    force = torch.where((R < R0) & (R > Req),
                        -((Fadh / (R0 - Req)) * R - Fadh * Req / (R0 - Req)),
                        torch.zeros_like(R))
    force = torch.where(R <= Req,
                        -Frep * coeff_rep * (1 / Req - 1 / R),
                        force)
    
    norm_diff = torch.nn.functional.normalize(coordinates_diff, dim=2)
    force_field = torch.sum(force[:, :, None] * norm_diff, dim=1)
    return force_field

def autovel(dX: torch.Tensor, n: torch.Tensor, tau: float, noise: float, dt: float, persistence: float) -> torch.Tensor:
    """
    Met à jour la direction d'une cellule en fonction de son déplacement, de sa direction précédente et d'un bruit aléatoire.

    La mise à jour de l'angle s'exprime par :
        θ(t+dt) = θ(t) + Δθ + bruit,
    où Δθ = (arcsin(n_x * dY - n_y * dX) * dt) / tau et le bruit est proportionnel à noise * √(dt).

    Paramètres:
        dX (torch.Tensor): Déplacement vectoriel sur dt.
        n (torch.Tensor): Direction précédente (normalisée).
        tau (float): Constante de temps pour la persistance (min).
        noise (float): Intensité du bruit angulaire.
        dt (float): Pas de temps (min).
        persistence (float): Facteur de persistance (utilisé pour moduler l'effet du passé).
    
    Retourne:
        torch.Tensor: Nouvelle direction normalisée (vecteur 2D).
    """
    dX_norm = torch.nn.functional.normalize(dX, dim=1) * 0.9999999
    if persistence == 1:
        persistence = 0.9999999
    theta = torch.atan2(dX_norm[:, 1], dX_norm[:, 0])
    dtheta = torch.arcsin((n[:, 0] * dX_norm[:, 1] - n[:, 1] * dX_norm[:, 0])) * dt / tau
    rnd = (2 * math.pi * (torch.rand(len(dX), 1, device=device) - 0.5)) * noise * np.sqrt(dt)
    theta_update = theta + dtheta + rnd.squeeze(1)
    new_direction = torch.stack((torch.cos(theta_update), torch.sin(theta_update)), dim=1)
    return new_direction

# =============================================================================
# Fonctions de visualisation
# =============================================================================

def plot_environment(cells, camp_field, space_size: float, axis, iteration: float = None):
    """
    Affiche l'environnement de simulation : champ de cAMP en arrière-plan et positions des cellules.

    Paramètres:
        cells: Liste d'instances CellAgent.
        camp_field: Instance de la classe cAMP.
        space_size (float): Taille du domaine (μm).
        axis: Axe matplotlib pour l'affichage.
        iteration (float, optionnel): Temps (min) affiché dans le titre.
    
    Retourne:
        im: Objet image du champ de cAMP affiché.
    """
    axis.set_xlim(0, space_size)
    axis.set_ylim(0, space_size)
    extent = [0, space_size, 0, space_size]
    im = axis.imshow(camp_field.signal.cpu().numpy().T, origin='lower', extent=extent,
                     cmap=plt.cm.viridis, alpha=0.5, vmin=0, vmax=15)
    # x = [cell.position[0].item() for cell in cells]
    # y = [cell.position[1].item() for cell in cells]
    # colors = ['blue' if cell.pop == 'Population 1' else 'red' for cell in cells]
    # axis.scatter(x, y, s=5, color=colors, alpha=0.5, edgecolors='k')
    for cell in cells:
        # Choix de la couleur selon la population
        col = 'blue' if cell.pop == 'Population 1' else 'red'
        # Crée un cercle de rayon R_EQ centré sur la position de la cellule
        circle = plt.Circle((cell.position[0].item(), cell.position[1].item()), R_EQ, color=col,
                            alpha=0.5, ec='k', lw=0.5)
        axis.add_patch(circle)
    axis.set_xlabel('Position X (μm)')
    axis.set_ylabel('Position Y (μm)')
    if iteration is not None:
        axis.set_title(f'Temps : {iteration * DELTA_T:.2f} min')
    return im

def plot_camp_field(camp_field, space_size: float, iteration: float, vmin=0, vmax=15):
    """
    Affiche une carte statique du champ de cAMP.

    Paramètres:
        camp_field: Instance de la classe cAMP.
        space_size (float): Taille du domaine (μm).
        iteration (float): Itération ou temps pour le titre.
        vmin, vmax: Limites de l'échelle de couleur.
    """
    extent = [0, space_size, 0, space_size]
    plt.figure(figsize=(6,6))
    im = plt.imshow(camp_field.signal.cpu().numpy().T, origin='lower', extent=extent,
                    cmap='viridis', alpha=0.8, vmin=vmin, vmax=vmax)
    plt.title(f'Champ de cAMP à l\'itération {iteration}')
    plt.xlabel('Position X (μm)')
    plt.ylabel('Position Y (μm)')
    plt.colorbar(im, label='Concentration de cAMP')
    plt.tight_layout()
    plt.close()

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

def plot_combined_state(cells, camp_field, SPACE_SIZE: float, iteration: float, PATH: str, device):
    """
    Trace une figure combinée composée de 4 sous-graphes :
      1) L'environnement (champ de cAMP + positions des cellules).
      2) Le champ complet de cAMP.
      3) La distribution spatiale de la concentration en A représentée par des cercles colorés.
      4) La distribution spatiale de la concentration en R représentée par des cercles colorés.

    Pour les axes[2] et axes[3], chaque cellule est représentée par un cercle (de rayon R_EQ)
    dont la couleur est obtenue via un colormap en fonction de sa valeur de A ou R.

    Paramètres:
        cells: Liste d'instances CellAgent.
        camp_field: Instance de la classe cAMP.
        SPACE_SIZE (float): Taille du domaine (μm).
        iteration (float): Itération ou temps pour le titre.
        PATH (str): Chemin de sauvegarde de l'image.
        device: Device utilisé pour Torch.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    
    # -- Axe 0 : Environnement (champ de cAMP + position des cellules) --
    im0 = plot_environment(cells, camp_field, SPACE_SIZE, axis=axes[0], iteration=iteration)
    fig.colorbar(im0, ax=axes[0], shrink=0.6, aspect=20, label='Concentration de cAMP')
    
    # -- Axe 1 : Champ complet de cAMP --
    extent = [0, SPACE_SIZE, 0, SPACE_SIZE]
    im1 = axes[1].imshow(
        camp_field.signal.cpu().numpy().T,
        origin='lower',
        extent=extent,
        cmap='viridis',
        alpha=0.8,
        vmin=0,     # Échelle fixée de 0...
        vmax=15     # ... à 15 pour le cAMP
    )
    axes[1].set_title(f'Champ de cAMP à l\'itération {iteration}')
    axes[1].set_xlabel('X (μm)')
    axes[1].set_ylabel('Y (μm)')
    fig.colorbar(im1, ax=axes[1], shrink=0.6, aspect=20, label='cAMP')
    
    # -- Axe 2 : Concentration de A par cellule (échelle -2 à 2) --
    values_A = [cell.A.item() for cell in cells]
    # On fixe l'échelle à [-2, 2], peu importe les valeurs réelles
    norm_A = mcolors.Normalize(vmin=-2, vmax=2)
    cmap_A = plt.get_cmap('YlOrBr')
    axes[2].set_xlim(0, SPACE_SIZE)
    axes[2].set_ylim(0, SPACE_SIZE)
    axes[2].set_aspect('equal', adjustable='box')  # Pour garder un aspect carré
    for cell in cells:
        val = cell.A.item()
        color = cmap_A(norm_A(val))
        circle = patches.Circle(
            (cell.position[0].item(), cell.position[1].item()),
            radius=R_EQ,  # Veillez à avoir R_EQ défini quelque part dans votre code
            facecolor=color,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )
        axes[2].add_patch(circle)
    axes[2].set_title(f'Concentration de A')
    axes[2].set_xlabel('X (μm)')
    axes[2].set_ylabel('Y (μm)')
    sm_A = plt.cm.ScalarMappable(cmap=cmap_A, norm=norm_A)
    sm_A.set_array([])
    fig.colorbar(sm_A, ax=axes[2], shrink=0.6, aspect=20, label='A ([-2, 2])')
    
    # -- Axe 3 : Concentration de R par cellule (échelle -2 à 2) --
    values_R = [cell.R.item() for cell in cells]
    norm_R = mcolors.Normalize(vmin=-2, vmax=2)
    cmap_R = plt.get_cmap('YlGn')
    axes[3].set_xlim(0, SPACE_SIZE)
    axes[3].set_ylim(0, SPACE_SIZE)
    axes[3].set_aspect('equal', adjustable='box')
    for cell in cells:
        val = cell.R.item()
        color = cmap_R(norm_R(val))
        circle = patches.Circle(
            (cell.position[0].item(), cell.position[1].item()),
            radius=R_EQ,
            facecolor=color,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )
        axes[3].add_patch(circle)
    axes[3].set_title(f'Concentration de R')
    axes[3].set_xlabel('X (μm)')
    axes[3].set_ylabel('Y (μm)')
    sm_R = plt.cm.ScalarMappable(cmap=cmap_R, norm=norm_R)
    sm_R.set_array([])
    fig.colorbar(sm_R, ax=axes[3], shrink=0.6, aspect=20, label='R ([-2, 2])')
    
    # Sauvegarde la figure
    plt.savefig(f'{PATH}combined_{iteration}.png', bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close()

def plot_function(pas: float, Req: float, R0: float, Frep: float, Fadh: float, a: float, coeff_rep: float):
    """
    Trace les courbes des forces de répulsion et d'adhésion en fonction de la distance.

    Paramètres:
        pas (float): Pas de discrétisation (μm).
        Req (float): Rayon d'équilibre (μm).
        R0 (float): Rayon maximal d'interaction (μm).
        Frep (float): Intensité de la force répulsive (a.u.).
        Fadh (float): Intensité de la force adhésive (a.u.).
        a (float): Coefficient supplémentaire (non utilisé ici).
        coeff_rep (float): Coefficient modifiant la force de répulsion.
    """
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.set_xlim(0, R0)
    axis.set_ylim(-Frep, Fadh)
    R_rep = np.arange(pas, Req, pas)
    force_rep = [Frep * coeff_rep * (1 / Req - 1 / R) for R in R_rep]
    axis.plot(R_rep, force_rep, label='Répulsion')
    R_adh = np.arange(Req, R0, pas)
    force_adh = [(Fadh / (R0 - Req)) * (R - Req) for R in R_adh]
    axis.plot(R_adh, force_adh, label='Adhésion')
    axis.set_xlabel('Distance (μm)')
    axis.set_ylabel('Force (a.u.)')
    axis.legend()
    plt.show()

# =============================================================================
# Initial state
# =============================================================================
def save_initial_state(cells, packing_fraction, filename):
    """
    Sauvegarde l'état initial de toutes les cellules dans un fichier CSV.

    Pour chaque cellule, on enregistre :
      - Les informations de base : id, population, position, vitesse.
      - Les paramètres de configuration individuels : velocity_magnitude, persistence, tau, noise, space_size, etc.
      - L'état interne courant : A, R, L, production cumulative de cAMP, état latent.
      - La direction de déplacement (ses composantes x et y).
      - Tous les paramètres contenus dans cell.cell_params (précédés de "param_").

    Une ligne additionnelle est ajoutée à la fin pour stocker la valeur du packing fraction.

    Paramètres:
      cells (list): Liste d'instances de CellAgent.
      packing_fraction (float): Valeur du packing fraction.
      filename (str): Chemin complet du fichier CSV de sauvegarde.
    """
    import pandas as pd

    data = []
    for cell in cells:
        # Construction d'un dictionnaire regroupant toutes les informations de la cellule.
        cell_data = {
            'id': cell.id,
            'pop': cell.pop,
            'x': cell.position[0].item(),
            'y': cell.position[1].item(),
            'vx': cell.velocity[0].item(),
            'vy': cell.velocity[1].item(),
            'velocity_magnitude': cell.velocity_magnitude,
            'persistence': cell.persistence,
            'tau': cell.tau,
            'noise': cell.noise,
            'space_size': cell.space_size,
            'sensitivity_cAMP_threshold': cell.sensitivity_threshold,
            'basal_value': cell.a0,
            'A': cell.A.item(),
            'R': cell.R.item(),
            'L': cell.L.item(),
            'camp_production': cell.camp_production,
            'is_latent': cell.is_latent,
            'dir_x': cell.direction[0].item(),
            'dir_y': cell.direction[1].item()
        }
        # Intégration de tous les paramètres contenus dans le dictionnaire cell_params,
        # en les préfixant par "param_" pour éviter toute confusion.
        for key, value in cell.cell_params.items():
            cell_data[f'param_{key}'] = value

        data.append(cell_data)

    # Ajout d'une ligne spéciale pour le packing fraction.
    data.append({"packing_fraction": packing_fraction})

    # Création du DataFrame et sauvegarde en CSV.
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def load_initial_state(filename: str) -> tuple:
    """
    Charge l'état initial des cellules depuis un fichier CSV et retourne
    à la fois la liste des objets CellAgent et la valeur du packing fraction.

    Le fichier CSV doit contenir, pour chaque cellule, les colonnes :
      - 'id', 'pop', 'x', 'y', 'vx', 'vy',
      - 'velocity_magnitude', 'persistence', 'tau', 'noise', 'space_size',
      - 'sensitivity_cAMP_threshold', 'basal_value',
      - 'A', 'R', 'L', 'camp_production', 'is_latent',
      - 'dir_x', 'dir_y'
    ainsi que toutes les clés de cell_params, sauvegardées avec le préfixe "param_".
    Une ligne additionnelle contenant uniquement la clé "packing_fraction" doit être présente.

    Paramètres:
        filename (str): Chemin vers le fichier CSV.
        
    Retourne:
        tuple: (cells, packing_fraction)
            - cells: liste d'objets CellAgent reconstruits.
            - packing_fraction: valeur lue dans le CSV (ou None si non trouvée).
    """
    import pandas as pd

    df = pd.read_csv(filename)

    # Extraire la ligne correspondant au packing fraction (où 'id' est NaN)
    pf_rows = df[df['id'].isnull()]
    if not pf_rows.empty:
        packing_fraction = pf_rows['packing_fraction'].iloc[0]
        # Conserver uniquement les lignes ayant une valeur dans 'id'
        df = df[df['id'].notnull()]
    else:
        packing_fraction = None

    cells = []
    # Parcourir chaque ligne pour reconstruire les cellules
    for _, row in df.iterrows():
        # Reconstruction de la position et de la vélocité sous forme de tenseurs Torch
        position = torch.tensor([float(row['x']), float(row['y'])], device=device, dtype=torch.float)
        velocity = torch.tensor([float(row['vx']), float(row['vy'])], device=device, dtype=torch.float)
        
        # Reconstruction du dictionnaire cell_params à partir des colonnes commençant par "param_"
        cell_params_reconstruit = {}
        for key in row.index:
            if key.startswith("param_"):
                # On enlève le préfixe "param_" pour retrouver la clé d'origine
                cell_params_reconstruit[key[6:]] = row[key]
        
        # Création de la cellule en utilisant les paramètres sauvegardés
        cell = CellAgent(
            id=int(row['id']),
            pop=row['pop'],
            position=position,
            velocity=velocity,
            velocity_magnitude=float(row['velocity_magnitude']),
            persistence=float(row['persistence']),
            space_size=float(row['space_size']),
            tau=float(row['tau']),
            noise=float(row['noise']),
            cell_params=cell_params_reconstruit,
            sensitivity_cAMP_threshold=float(row['sensitivity_cAMP_threshold']),
            basal_value=float(row['basal_value']),
            A_init=float(row['A']),
            R_init=float(row['R'])
        )
        # Restauration de l'état courant pour L, la production cumulée et l'état latent
        cell.L = torch.tensor(float(row['L']), device=device, dtype=torch.float)
        cell.camp_production = float(row['camp_production'])
        cell.is_latent = bool(row['is_latent'])
        # Reconstruction de la direction (normalisée)
        direction = torch.tensor([float(row['dir_x']), float(row['dir_y'])], device=device, dtype=torch.float)
        cell.direction = torch.nn.functional.normalize(direction, p=2, dim=0)
        
        cells.append(cell)
    
    return cells, packing_fraction
# =============================================================================
# Classes représentant les agents et le champ
# =============================================================================

class CellAgent:
    """
    Représente une cellule avec :
      - Dynamique interne selon FitzHugh–Nagumo
      - Cinétique de liaison récepteur-cAMP

    Équations résolues :
      • Cinétique de liaison :
             dL/dt = kon * [cAMP] * (1 - L) - koff * L
      • FitzHugh–Nagumo (après activation, L ≥ seuil) :
             dA/dt = A - A³/3 - R + I_S   avec I_S = a * log(1 + L/Kd)
             dR/dt = ε * (A - γ R + [cAMP])
    """
    def __init__(self, id: int, pop: str, position: torch.Tensor, velocity: torch.Tensor,
                 velocity_magnitude: float, persistence: float, space_size: float, tau: float,
                 noise: float, cell_params: dict, sensitivity_cAMP_threshold: float,
                 basal_value: float, A_init: float, R_init: float):
        """
        Initialise la cellule.

        Paramètres:
            id (int): Identifiant unique.
            pop (str): Étiquette de la population.
            position (torch.Tensor): Position initiale (2D, μm).
            velocity (torch.Tensor): Vecteur vitesse initiale (μm/min).
            velocity_magnitude (float): Norme de la vitesse (μm/min).
            persistence (float): Facteur de persistance.
            space_size (float): Taille du domaine (μm).
            tau (float): Constante de temps directionnelle (min).
            noise (float): Intensité du bruit dans A.
            cell_params (dict): Paramètres du modèle (voir ci-dessus).
            sensitivity_cAMP_threshold (float): Seuil de détection du cAMP (a.u.).
            basal_value (float): Production basale de cAMP (a.u.).
            A_init (float): État initial de A.
            R_init (float): État initial de R.
        """
        self.id = id
        self.pop = pop
        self.position_init = position.clone().to(device)
        self.position = position.clone().to(device)
        self.velocity = velocity.clone().to(device)
        self.velocity_magnitude = velocity_magnitude
        self.persistence = persistence
        self.space_size = space_size
        self.tau = tau
        self.noise = noise
        self.direction = torch.nn.functional.normalize(velocity, p=2, dim=0)
        self.A = torch.tensor(A_init, device=device, dtype=torch.float)
        self.R = torch.tensor(R_init, device=device, dtype=torch.float)
        self.cell_params = cell_params
        self.D = cell_params['D']
        self.a0 = basal_value
        self.af = cell_params['af']
        self.sensitivity_threshold = sensitivity_cAMP_threshold
        self.camp_production = 0.0
        self.is_latent = True  # État latent tant que L < seuil
        self.L = torch.tensor(0.0, device=device, dtype=torch.float)

    def update_state(self, signal_value: float, dt: float):
        """
        Met à jour l'état interne de la cellule en fonction du signal local de cAMP.

        1. Mise à jour de la liaison des récepteurs :
               dL/dt = kon * [cAMP] * (1 - L) - koff * L
        2. Tant que L < seuil, la cellule reste latente avec A et R nuls.
        3. Sinon, résolution du modèle de FitzHugh–Nagumo :
               dA/dt = A - A³/3 - R + I_S   avec I_S = a * log(1 + L/Kd)
               dR/dt = ε * (A - γR + [cAMP])
        
        Paramètres:
            signal_value (float): Concentration locale de cAMP (a.u.).
            dt (float): Pas de temps (min).
        """
        kon = self.cell_params.get('kon', 0.1)
        koff = self.cell_params.get('koff', 0.1)
        dL = (kon * signal_value * (1 - self.L) - koff * self.L) * dt
        self.L += dL
        self.L = torch.clamp(self.L, 0, 1)

        activation_threshold = self.cell_params['activation_threshold_cAMP']
        if self.is_latent:
            if self.L < activation_threshold:
                self.A = torch.tensor(0.0, device=device)
                self.R = torch.tensor(0.0, device=device)
                return
            else:
                self.is_latent = False

        a = self.cell_params['a']
        Kd = self.cell_params['Kd']
        gamma = self.cell_params['gamma']
        epsilon = self.cell_params['epsilon']
        sigma = self.cell_params['sigma']
        noise_flag = self.cell_params.get('noise', True)
        
        I_S = a * torch.log1p(self.L / Kd)
        dA = (self.A - (self.A ** 3) / 3 - self.R + I_S) * dt
        if noise_flag:
            dA += sigma * math.sqrt(dt) * torch.randn((), device=device)
        self.A += dA
        
        dR = (self.A - gamma * self.R + signal_value) * epsilon * dt
        self.R += dR

# On suppose que device et cell_id_counter sont définis globalement quelque part dans votre code.
# Exemple :
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cell_id_counter = 0

class Population:
    """
    Représente une population de cellules dans un domaine 2D.

    Cette classe permet d'initialiser un ensemble de cellules en veillant à respecter une
    distance minimale entre elles. Elle peut également tenir compte de cellules déjà
    existantes (provenant d'une autre population) pour éviter les collisions inter-populations.
    """

    def __init__(self,
                 num_cells: int,
                 space_size: float,
                 velocity_magnitude: float,
                 persistence: float,
                 min_distance: float,
                 pop_tag: str,
                 ecart_type: float,
                 tau: float,
                 noise: float,
                 cell_params: dict,
                 sensitivity_cAMP_threshold: float,
                 basal_fraction: float = 0.1,
                 A_init: float = 1.0,
                 R_init: float = 1.0,
                 existing_cells=None):
        """
        Initialise la population de cellules.

        Args:
            num_cells (int):
                Nombre de cellules à générer dans cette population.
            space_size (float):
                Taille du domaine (en μm). On considère un carré [0, space_size] x [0, space_size].
            velocity_magnitude (float):
                Vitesse moyenne des cellules (en μm/min).
            persistence (float):
                Facteur de persistance directionnelle.
            min_distance (float):
                Distance minimale entre deux cellules (en μm).
            pop_tag (str):
                Étiquette (nom) de la population (ex: "Population 1").
            ecart_type (float):
                Écart-type utilisé pour la distribution aléatoire des vitesses autour de velocity_magnitude.
            tau (float):
                Constante de temps pour la dynamique de la direction (persistance).
            noise (float):
                Intensité du bruit (peut influencer la dynamique de la cellule selon le modèle).
            cell_params (dict):
                Dictionnaire des paramètres cellulaires (ex: paramètres FitzHugh–Nagumo, production cAMP, etc.).
            sensitivity_cAMP_threshold (float):
                Seuil de détection du cAMP pour la cellule.
            basal_fraction (float, optionnel):
                Fraction de cellules ayant une production basale non nulle (défaut: 0.1).
            A_init (float, optionnel):
                Valeur initiale de la variable A (FHN) pour chaque cellule (défaut: 1.0).
            R_init (float, optionnel):
                Valeur initiale de la variable R (FHN) pour chaque cellule (défaut: 1.0).
            existing_cells (list, optionnel):
                Liste d'objets CellAgent déjà placés (dans une autre population par exemple).
                Permet d'éviter les collisions inter-populations. (défaut: None)

        Remarques:
            - Cette classe repose sur un objet global `cell_id_counter` pour l'attribution
              d'IDs uniques aux cellules.
            - Les cellules nouvellement créées sont stockées dans `self.cells`.
        """
        self.num_cells = num_cells
        self.space_size = space_size
        self.velocity_magnitude = velocity_magnitude
        self.persistence = persistence
        self.min_distance = min_distance
        self.pop_tag = pop_tag
        self.ecart_type = ecart_type
        self.tau = tau
        self.noise = noise
        self.cell_params = cell_params
        self.sensitivity_cAMP_threshold = sensitivity_cAMP_threshold
        self.basal_fraction = basal_fraction
        self.cells = []
        self.A_init = A_init
        self.R_init = R_init

        # Si aucune liste de cellules existantes n'est fournie, on crée une liste vide.
        if existing_cells is None:
            self.existing_cells = []
        else:
            self.existing_cells = existing_cells

        # Initialisation des cellules de la population.
        self.initialize_cells()

    def initialize_cells(self) -> None:
        """
        Initialise la population de cellules en respectant la distance minimale
        par rapport aux cellules déjà existantes (existing_cells) et par rapport
        aux cellules de cette même population.

        Cette fonction effectue jusqu'à 50 tentatives par cellule pour trouver
        une position aléatoire valide. Si aucune position n'est trouvée, la
        cellule est ignorée et un message d'avertissement est affiché.

        Remarque:
            - Utilise la variable globale `cell_id_counter` pour attribuer un ID unique
            à chaque nouvelle cellule.
        """
        global cell_id_counter

        # Liste interne pour stocker les cellules placées dans CETTE population.
        placed_cells = []

        # On récupère les positions des cellules "existantes" (issues d'autres populations).
        existing_positions = [c.position.clone() for c in self.existing_cells]

        max_attempts = 50
        for i in range(self.num_cells):
            attempt = 0
            placed = False

            while attempt < max_attempts and not placed:
                attempt += 1

                # On tire une position aléatoire dans [0, space_size].
                pos = torch.rand(2, device=device) * self.space_size

                # Vérifie la distance vis-à-vis des cellules existantes.
                conflict = False
                for other_pos in existing_positions:
                    if torch.norm(pos - other_pos) < self.min_distance:
                        conflict = True
                        break

                # Vérifie la distance vis-à-vis des cellules déjà placées dans cette population.
                if not conflict:
                    for c in placed_cells:
                        if torch.norm(pos - c.position) < self.min_distance:
                            conflict = True
                            break

                # Si aucune collision n'est détectée, on crée la cellule.
                if not conflict:
                    direction = torch.nn.functional.normalize(
                        torch.rand(2, device=device) * 2 - 1, dim=0
                    )
                    speed = torch.normal(
                        mean=self.velocity_magnitude,
                        std=self.ecart_type,
                        size=(1,),
                        device=device
                    ).item()
                    velocity = direction * speed
                    basal_value = (self.cell_params['a0']
                                if random.random() < self.basal_fraction
                                else 0)

                    # Création de la cellule
                    new_cell = CellAgent(
                        id=cell_id_counter,
                        pop=self.pop_tag,
                        position=pos,
                        velocity=velocity,
                        velocity_magnitude=speed,
                        persistence=self.persistence,
                        space_size=self.space_size,
                        tau=self.tau,
                        noise=self.noise,
                        cell_params=self.cell_params,
                        sensitivity_cAMP_threshold=self.sensitivity_cAMP_threshold,
                        basal_value=basal_value,
                        A_init=self.A_init,
                        R_init=self.R_init
                    )
                    cell_id_counter += 1
                    placed_cells.append(new_cell)
                    placed = True

            if not placed:
                print(f"Impossible de placer la cellule {i} de la population {self.pop_tag} après {max_attempts} essais.")

        # On stocke les cellules placées dans self.cells.
        self.cells = placed_cells

        # Affichage du nombre total de cellules placées.
        print(f"Population {self.pop_tag} : {len(self.cells)} cellules placées sur {self.num_cells} demandées.")

class Surface:
    """
    Classe pour représenter une surface avec friction variable.
    """
    def get_friction(self, position):
        """
        Retourne une valeur de friction aléatoire entre 0 et 0.2.
        """
        return torch.empty(1, device=device).uniform_(0, 0.2).item()

class cAMP:
    """
    Représente le champ de cAMP, évoluant selon :
         ∂S/∂t = D_cAMP ∇²S - aPDE * S + Production
    La production locale est répartie sur un patch via un noyau gaussien.
    """
    def __init__(self, space_size: float, cell_params: dict, initial_condition=None):
        """
        Initialise le champ de cAMP.

        Paramètres:
            space_size (float): Taille du domaine (μm).
            cell_params (dict): Paramètres du cAMP.
            initial_condition: Condition initiale (non utilisée ici).
        """
        self.space_size = space_size
        self.grid_resolution = cell_params['grid_resolution']  # μm
        self.grid_size = int(space_size / self.grid_resolution)
        self.D_cAMP = cell_params['D_cAMP']    # μm²/min
        self.aPDE = cell_params['aPDE']          # min⁻¹
        self.a0 = cell_params['a0']              # Production basale (a.u.)
        self.dx = self.grid_resolution         # μm
        self.dt = DELTA_T                      # min
        x = torch.linspace(0, space_size, self.grid_size, device=device)
        y = torch.linspace(0, space_size, self.grid_size, device=device)
        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
        self.signal = torch.zeros((self.grid_size, self.grid_size), device=device)
        
        # Noyau gaussien pour la production locale
        self.prod_radius = 3
        kernel_size = 2 * self.prod_radius + 1
        sigma = self.prod_radius / 2.0
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        for i in range(kernel_size):
            for j in range(kernel_size):
                dx_val = i - self.prod_radius
                dy_val = j - self.prod_radius
                kernel[i, j] = math.exp(-(dx_val**2 + dy_val**2) / (2 * sigma**2))
        kernel = kernel / np.sum(kernel)
        self.kernel = torch.tensor(kernel, device=device)

    def compute_laplacian(self, S: torch.Tensor) -> torch.Tensor:
        """
        Calcule le Laplacien de S via un schéma 4-points avec conditions périodiques.
        """
        laplacian_S = (torch.roll(S, shifts=1, dims=0) + torch.roll(S, shifts=-1, dims=0) +
                       torch.roll(S, shifts=1, dims=1) + torch.roll(S, shifts=-1, dims=1) -
                       4 * S) / (self.dx ** 2)
        return laplacian_S

    def compute_laplacian_9point(self, S: torch.Tensor) -> torch.Tensor:
        """
        Calcule le Laplacien de S via un schéma 9-points (meilleure isotropie).
        """
        dx2 = self.dx ** 2
        
        S_up    = torch.roll(S, shifts=+1, dims=0)
        S_down  = torch.roll(S, shifts=-1, dims=0)
        S_left  = torch.roll(S, shifts=+1, dims=1)
        S_right = torch.roll(S, shifts=-1, dims=1)
        
        S_upleft    = torch.roll(S_up,    shifts=+1, dims=1)
        S_upright   = torch.roll(S_up,    shifts=-1, dims=1)
        S_downleft  = torch.roll(S_down,  shifts=+1, dims=1)
        S_downright = torch.roll(S_down,  shifts=-1, dims=1)
        
        laplacian_S = (-20.0 * S + 4.0 * (S_up + S_down + S_left + S_right) +
                       2.0 * (S_upleft + S_upright + S_downleft + S_downright)) / (6.0 * dx2)
        return laplacian_S
    
    def update(self, cells: list):
        """
        Met à jour le champ de cAMP par diffusion, dégradation et production locale.

        Équation discrétisée :
            S(t+dt) = S(t) + dt * (D_cAMP * Laplacien - aPDE * S + Production)
        """
        A_grid = torch.zeros_like(self.signal)
        production_threshold = 15
        if cells:
            for cell in cells:
                x_idx = int(cell.position[0].item() / self.grid_resolution) % self.grid_size
                y_idx = int(cell.position[1].item() / self.grid_resolution) % self.grid_size
                local_signal = self.get_signal_at_position(cell.position)
                if not cell.is_latent and local_signal > 0:
                    cell.camp_production += cell.a0
                    for dx in range(-self.prod_radius, self.prod_radius + 1):
                        for dy in range(-self.prod_radius, self.prod_radius + 1):
                            xx = (x_idx + dx) % self.grid_size
                            yy = (y_idx + dy) % self.grid_size
                            weight = self.kernel[dx + self.prod_radius, dy + self.prod_radius]
                            A_grid[xx, yy] += cell.a0 * weight
                if not cell.is_latent and local_signal < production_threshold:
                    if cell.A > cell.af:
                        cell.camp_production += cell.D
                        for dx in range(-self.prod_radius, self.prod_radius + 1):
                            for dy in range(-self.prod_radius, self.prod_radius + 1):
                                xx = (x_idx + dx) % self.grid_size
                                yy = (y_idx + dy) % self.grid_size
                                weight = self.kernel[dx + self.prod_radius, dy + self.prod_radius]
                                A_grid[xx, yy] += cell.D * weight #* cell.L.item()
        laplacian_S = self.compute_laplacian_9point(self.signal)
        degradation_term = self.aPDE * self.signal if cells else 0.0
        self.signal += self.dt * (self.D_cAMP * laplacian_S - degradation_term + A_grid)
        self.signal = torch.clamp(self.signal, min=0)
        if torch.isnan(self.signal).any() or torch.isinf(self.signal).any():
            print("NaN or Inf detected in cAMP signal.")
            sys.exit(1)

    def get_signal_at_position(self, position: torch.Tensor) -> float:
        """
        Retourne la concentration de cAMP à une position donnée.
        """
        x_idx = int(position[0].item() / self.grid_resolution) % self.grid_size
        y_idx = int(position[1].item() / self.grid_resolution) % self.grid_size
        return self.signal[x_idx, y_idx]

    def compute_gradient_at(self, position: torch.Tensor) -> torch.Tensor:
        """
        Calcule le gradient du champ de cAMP en un point par différences centrales.
        """
        grad_x = (torch.roll(self.signal, shifts=-1, dims=0) - torch.roll(self.signal, shifts=1, dims=0)) / (2 * self.dx)
        grad_y = (torch.roll(self.signal, shifts=-1, dims=1) - torch.roll(self.signal, shifts=1, dims=1)) / (2 * self.dx)
        x_idx = int(position[0].item() / self.grid_resolution) % self.grid_size
        y_idx = int(position[1].item() / self.grid_resolution) % self.grid_size
        return torch.tensor([grad_x[x_idx, y_idx], grad_y[x_idx, y_idx]], device=device)

def create_uniform_kernel(radius: float, grid_resolution: float):
    """
    Crée un noyau uniforme pour distribuer le cAMP de manière homogène
    dans un cercle de rayon donné (en μm) sur une grille de résolution grid_resolution (en μm).

    Retourne:
        kernel (torch.Tensor): Noyau uniforme de dimension (n, n) dont la somme vaut 1.
    """
    # Calcul du rayon en nombre de cases
    r_cells = int(math.ceil(radius / grid_resolution))
    size = 2 * r_cells + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    
    count = 0
    for i in range(-r_cells, r_cells + 1):
        for j in range(-r_cells, r_cells + 1):
            # Vérifier si le point (i, j) est à l'intérieur du cercle
            if i**2 + j**2 <= (radius / grid_resolution)**2:
                kernel[i + r_cells, j + r_cells] = 1
                count += 1
    # Normalisation pour que la somme du noyau soit 1
    kernel /= count
    return torch.tensor(kernel, device=device)

# Dans la classe cAMP, par exemple dans __init__, vous pouvez créer ce noyau uniforme :
# class cAMP:
#     def __init__(self, space_size: float, cell_params: dict, initial_condition=None):
#         self.space_size = space_size
#         self.grid_resolution = cell_params['grid_resolution']  # μm
#         self.grid_size = int(space_size / self.grid_resolution)
#         self.D_cAMP = cell_params['D_cAMP']    # μm²/min
#         self.aPDE = cell_params['aPDE']          # min⁻¹
#         self.a0 = cell_params['a0']              # Production basale (a.u.)
#         self.dx = self.grid_resolution         # μm
#         self.dt = DELTA_T                      # min
#         x = torch.linspace(0, space_size, self.grid_size, device=device)
#         y = torch.linspace(0, space_size, self.grid_size, device=device)
#         self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
#         self.signal = torch.zeros((self.grid_size, self.grid_size), device=device)
        
#         # Créer un noyau uniforme pour une production sur le cercle de la cellule
#         self.cell_radius = R_EQ  # Assurez-vous que R_EQ est défini et correspond au rayon de la cellule (en μm)
#         self.kernel = create_uniform_kernel(self.cell_radius, self.grid_resolution)

#     def compute_laplacian(self, S: torch.Tensor) -> torch.Tensor:
#         """
#         Calcule le Laplacien de S via un schéma 4-points avec conditions périodiques.
#         """
#         laplacian_S = (torch.roll(S, shifts=1, dims=0) + torch.roll(S, shifts=-1, dims=0) +
#                        torch.roll(S, shifts=1, dims=1) + torch.roll(S, shifts=-1, dims=1) -
#                        4 * S) / (self.dx ** 2)
#         return laplacian_S

#     def compute_laplacian_9point(self, S: torch.Tensor) -> torch.Tensor:
#         """
#         Calcule le Laplacien de S via un schéma 9-points (meilleure isotropie).
#         """
#         dx2 = self.dx ** 2
        
#         S_up    = torch.roll(S, shifts=+1, dims=0)
#         S_down  = torch.roll(S, shifts=-1, dims=0)
#         S_left  = torch.roll(S, shifts=+1, dims=1)
#         S_right = torch.roll(S, shifts=-1, dims=1)
        
#         S_upleft    = torch.roll(S_up,    shifts=+1, dims=1)
#         S_upright   = torch.roll(S_up,    shifts=-1, dims=1)
#         S_downleft  = torch.roll(S_down,  shifts=+1, dims=1)
#         S_downright = torch.roll(S_down,  shifts=-1, dims=1)
        
#         laplacian_S = (-20.0 * S + 4.0 * (S_up + S_down + S_left + S_right) +
#                        2.0 * (S_upleft + S_upright + S_downleft + S_downright)) / (6.0 * dx2)
#         return laplacian_S

#     def update(self, cells: list):
#         """
#         Met à jour le champ de cAMP par diffusion, dégradation et production locale.

#         Équation discrétisée :
#             S(t+dt) = S(t) + dt * (D_cAMP * Laplacien - aPDE * S + Production)
#         """
#         A_grid = torch.zeros_like(self.signal)
#         production_threshold = 15.0  # Par exemple, ne produire du cAMP que si la concentration locale < 5.0
#         if cells:
#             for cell in cells:
#                 x_idx = int(cell.position[0].item() / self.grid_resolution) % self.grid_size
#                 y_idx = int(cell.position[1].item() / self.grid_resolution) % self.grid_size
#                 local_signal = self.get_signal_at_position(cell.position)

#                 if not cell.is_latent and local_signal < production_threshold:
#                     # Production basale uniformément dans le cercle de la cellule
#                     for dx in range(-self.kernel.shape[0]//2, self.kernel.shape[0]//2 + 1):
#                         for dy in range(-self.kernel.shape[1]//2, self.kernel.shape[1]//2 + 1):
#                             xx = (x_idx + dx) % self.grid_size
#                             yy = (y_idx + dy) % self.grid_size
#                             weight = self.kernel[dx + self.kernel.shape[0]//2, dy + self.kernel.shape[1]//2]
#                             A_grid[xx, yy] += cell.a0 * weight
#                     # Production additionnelle si A dépasse un certain seuil (af)
                    
#                     if cell.A > cell.af:
#                         for dx in range(-self.kernel.shape[0]//2, self.kernel.shape[0]//2 + 1):
#                             for dy in range(-self.kernel.shape[1]//2, self.kernel.shape[1]//2 + 1):
#                                 xx = (x_idx + dx) % self.grid_size
#                                 yy = (y_idx + dy) % self.grid_size
#                                 weight = self.kernel[dx + self.kernel.shape[0]//2, dy + self.kernel.shape[1]//2]
#                                 A_grid[xx, yy] += cell.D #* weight
#         laplacian_S = self.compute_laplacian_9point(self.signal)
#         degradation_term = self.aPDE * self.signal if cells else 0.0
#         self.signal += self.dt * (self.D_cAMP * laplacian_S - degradation_term + A_grid)
#         self.signal = torch.clamp(self.signal, min=0)
#         if torch.isnan(self.signal).any() or torch.isinf(self.signal).any():
#             print("NaN or Inf detected in cAMP signal.")
#             sys.exit(1)

#     def get_signal_at_position(self, position: torch.Tensor) -> float:
#         """
#         Retourne la concentration de cAMP à une position donnée.
#         """
#         x_idx = int(position[0].item() / self.grid_resolution) % self.grid_size
#         y_idx = int(position[1].item() / self.grid_resolution) % self.grid_size
#         return self.signal[x_idx, y_idx]

#     def compute_gradient_at(self, position: torch.Tensor) -> torch.Tensor:
        # """
        # Calcule le gradient du champ de cAMP en un point par différences centrales.
        # """
        # grad_x = (torch.roll(self.signal, shifts=-1, dims=0) - torch.roll(self.signal, shifts=1, dims=0)) / (2 * self.dx)
        # grad_y = (torch.roll(self.signal, shifts=-1, dims=1) - torch.roll(self.signal, shifts=1, dims=1)) / (2 * self.dx)
        # x_idx = int(position[0].item() / self.grid_resolution) % self.grid_size
        # y_idx = int(position[1].item() / self.grid_resolution) % self.grid_size
        # return torch.tensor([grad_x[x_idx, y_idx], grad_y[x_idx, y_idx]], device=device)
# =============================================================================
# Paramètres de simulation
# =============================================================================

# Contrôles généraux
INCLUDE_CELLS = True       # Simulation avec cellules
INITIAL_AMPc = True        # Injection initiale de cAMP
PLOT = True                # Activation de l'affichage

# Domaine et temps
SPACE_SIZE = 200           # μm, taille du domaine (carré)
TIME_SIMU = 100           # min, durée totale de la simulation

# Paramètre de détection de gradient
R_SENSING_GRAD = 5.0       # μm

# =============================================================================
# Paramètres du modèle (FitzHugh–Nagumo, diffusion du cAMP, etc.)
# =============================================================================
cell_params = {
    'c0': 0.4,  # (a.u.) Paramètre non utilisé dans la version actuelle. Initialement prévu pour 
                # introduire un terme constant dans l'équation de R. Peut être conservé pour des 
                # expérimentations futures. [Plage: 0 à 1] --> Pourrait remplacer local_signal mis actuellement

    'a': 0.4,  # (a.u.) Intensité du terme de stimulation dans l'équation de A. Influence 
               # l'excitabilité du système. [Plage: 0.1 à 2.0]

    'gamma': 0.1,  # (min⁻¹) Coefficient de couplage dans l'équation de R, contrôlant 
                   # la vitesse de relaxation de R. [Plage: 0.1 à 1.0]

    'Kd': 0.5,  # (a.u.) Constante de dissociation du cAMP pour la liaison aux récepteurs. 
              # Définit la sensibilité des récepteurs au cAMP. [Plage: 1 à 10]

    'sigma': 0.01,  # (a.u.) Amplitude du bruit ajouté dans la dynamique de A pour 
                    # simuler des fluctuations stochastiques. [Plage: 0 à 0.1]

    'epsilon': 0.1,  # (min⁻¹) Facteur d'échelle influençant l'évolution de R. 
                       # Généralement choisi petit pour ralentir R par rapport à A. [Plage: 0.01 à 0.1]

    'D': 4e4,  # (a.u.) Quantité de cAMP produite lors d’un "spike" d’activation. 
                # Définit l'intensité de la production ponctuelle. [Plage: 10 à 50]

    'a0': 0,  # (a.u.) Production basale de cAMP, même en l'absence de stimulation forte. 
              # [Plage: 0 à 1]

    'af': -1,  # (a.u.) Seuil d'activation de la production additionnelle de cAMP lorsque A dépasse cette valeur. 
              # [Plage: -1 à 1]

    'noise': False,  # (bool) Activation ou non du bruit stochastique dans la dynamique de A. 
                     # [Valeurs possibles: True ou False]

    'D_cAMP': 1.0,  # (μm²/min) Coefficient de diffusion du cAMP dans le milieu. 
                    # Contrôle la propagation du cAMP. [Plage: 0.1 à 1.0]

    'aPDE': 0.7, #0.7,  # (min⁻¹) Taux de dégradation du cAMP, simulant l’action de la phosphodiestérase (PDE). 
                  # [Plage: 0.1 à 1.0]

    'grid_resolution': 1, # 0.5,  # (μm) Taille d'une case de la grille spatiale pour la diffusion du cAMP. 
                             # [Plage: 0.1 à 1.0]

    'chemotaxis_sensitivity': 0.0,  # (sans unité) Sensibilité des cellules au gradient de cAMP. 
                                    # 0 = pas de réponse, 1 = réponse maximale. [Plage: 0 à 1]

    'activation_threshold_cAMP': 0.1,  #0.1 (sans unité) Seuil sur la fraction de récepteurs liés au cAMP (L).
                                   # Tant que L < activation_threshold_cAMP, la cellule reste latente.
                                   # Lorsque L dépasse ce seuil, la cellule devient active et suit FitzHugh-Nagumo.
                                   # [Plage: 0 à 1]
                                   
    'kon': 0.5,  # (min⁻¹) Constante de liaison du cAMP aux récepteurs, définissant la rapidité d’association. 
                 # [Plage: 0.1 à 5.0]

    'koff': 1,  # (min⁻¹) Constante de dissociation du cAMP, définissant la rapidité de libération des récepteurs. 
                  # [Plage: 0.1 à 5.0]
}

# Critère CFL pour le pas de temps
FACTEUR_SECURITE = 0.9
if cell_params['D_cAMP'] == 0:
    DELTA_T = 0.001
else:
    DELTA_T = FACTEUR_SECURITE * (cell_params['grid_resolution'] ** 2) / (4 * cell_params['D_cAMP'])
    print("delta t serai", DELTA_T)
DELTA_T = 0.01
print("Intervalle de temps (min):", DELTA_T)
PLOT_INTERVAL = int(1 / DELTA_T)

cell_params['D'] = 5*cell_params['D']*DELTA_T

# Paramètres d'interaction cellulaire
MU = 0                   # μm/(a.u.×min), désactivation du déplacement par force
F_REP = 40               # a.u., force répulsive
F_ADH = 7                # a.u., force adhésive
R_EQ = 4               # μm, rayon d'équilibre
R_0 = 5.8                # μm, rayon maximal d'interaction
MIN_DISTANCE_INIT = 2*R_EQ # μm, distance minimale initiale
COEFF_CARRE = 50         # Coefficient pour force quadratique (optionnel)
COEFF_REP = 0.5          # Coefficient pour force répulsive
FLUCTUATION_FACTOR = 0   # Fluctuation aléatoire


initial_state = False
if initial_state == False:
    saving_initial_state = True
else:
    saving_initial_state = False

if initial_state:
    cells, PACKING_FRACTION= load_initial_state("/Users/souchaud/Desktop/initial_state.csv" )
    N_CELLS = len(cells)
    print(N_CELLS, "cells")
else:   
    # Nombre de cellules
    PACKING_FRACTION = 0.9
    # Apprioximation du nombre max de cercle dans un carré de coté SPACE_SIZE
    # N_max = SPACE_SIZE**2 / (2*np.sqrt(3)*R_EQ**2)
    N_CELLS = int((PACKING_FRACTION * SPACE_SIZE ** 2) / (math.pi * ((R_EQ) ** 2))) # Req/2???
    N_CELLS = int((SPACE_SIZE**2)/(4*np.sqrt(3)*R_EQ**2))
    # N_CELLS = 1
    print(N_CELLS, "cells")

    # Paramètres pour deux populations
    velocity_magnitude_pop1 = 0
    ECART_TYPE_POP1 = 0.3
    NOISE_POP_1 = 0
    TAU_POP_1 = 5 
    PERSISTENCE_POP1 = 0
    SENSITIVITY_cAMP_THRESHOLD_POP1 = 2

    velocity_magnitude_pop2 = 0
    ECART_TYPE_POP2 = 0.5
    NOISE_POP_2 = 0
    TAU_POP_2 = 5
    PERSISTENCE_POP2 = 0
    SENSITIVITY_cAMP_THRESHOLD_POP2 = 2

    pop1 = N_CELLS // 2
    pop2 = N_CELLS - pop1

    initial_A = -1
    initial_R = -1

    cell_id_counter = 0  # Identifiant unique global

    # 1) Première population (pas d'existing_cells)
    population1 = Population(
        num_cells=pop1,
        space_size=SPACE_SIZE,
        velocity_magnitude=velocity_magnitude_pop1,
        persistence=PERSISTENCE_POP1,
        ecart_type=ECART_TYPE_POP1,
        min_distance=MIN_DISTANCE_INIT,
        pop_tag="Population 1",
        tau=TAU_POP_1,
        noise=NOISE_POP_1,
        cell_params=cell_params,
        sensitivity_cAMP_threshold=SENSITIVITY_cAMP_THRESHOLD_POP1,
        basal_fraction=0.001,
        A_init=initial_A,
        R_init=initial_R
    )

    # 2) Deuxième population (on passe la liste des cellules déjà créées)
    population2 = Population(
        num_cells=pop2,
        space_size=SPACE_SIZE,
        velocity_magnitude=velocity_magnitude_pop2,
        persistence=PERSISTENCE_POP2,
        ecart_type=ECART_TYPE_POP2,
        min_distance=MIN_DISTANCE_INIT,
        pop_tag="Population 2",
        tau=TAU_POP_2,
        noise=NOISE_POP_2,
        cell_params=cell_params,
        sensitivity_cAMP_threshold=SENSITIVITY_cAMP_THRESHOLD_POP2,
        basal_fraction=0.001,
        A_init=initial_A,
        R_init=initial_R,
        existing_cells=population1.cells  # <-- On informe la Pop2 qu'il y a déjà des cellules
    )

    cells = population1.cells + population2.cells

    if saving_initial_state:
        save_initial_state(cells, PACKING_FRACTION, "/Users/souchaud/Desktop/initial_state.csv")

surface = Surface()
camp_field = cAMP(SPACE_SIZE, cell_params, initial_condition=None)

# Injection initiale de cAMP aux positions de quelques cellules pour activer certaines cellules
if INITIAL_AMPc:
    if int(N_CELLS * 0.01) == 0:
        n_cells_to_activate = 20
    else:
        n_cells_to_activate = int(N_CELLS * 0.01)
    # n_cells_to_activate = 1
    print(n_cells_to_activate, "cells activated")

    indices_a_activer = random.sample(range(len(cells)), k=n_cells_to_activate)
    for i, cell in enumerate(cells):
        x_idx = int(cell.position[0].item() / camp_field.grid_resolution) % camp_field.grid_size
        y_idx = int(cell.position[1].item() / camp_field.grid_resolution) % camp_field.grid_size
        if i in indices_a_activer:
            camp_field.signal[x_idx, y_idx] += cell_params['D']/10
        else:
            camp_field.signal[x_idx, y_idx] += 0.0
    plot_camp_field(camp_field, space_size=SPACE_SIZE, iteration=0, vmin=0, vmax=15)

# Sauvegarde initiale des figures
if PLOT:
    PATH = f'../simulations_images/latent_mode_2/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    else:
        print("WARNING : FOLDER ALREADY EXISTS! Les images vont être écrasées.")
    fig_tmp, ax_tmp = plt.subplots(figsize=(6, 6))
    plot_environment(cells, camp_field, SPACE_SIZE, axis=ax_tmp, iteration=0)
    plot_combined_state(cells, camp_field, SPACE_SIZE, 0, PATH, device)
    plt.savefig(f'{PATH}image_0.png', bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close()

# Enregistrement des paramètres de la simulation dans un fichier texte

if save_initial_state ==False:
    def save_parameters(PATH, cell_params, SPACE_SIZE, TIME_SIMU, DELTA_T, PLOT_INTERVAL, PACKING_FRACTION, N_CELLS,
                        F_REP, F_ADH, R_EQ, R_0, MIN_DISTANCE_INIT, COEFF_CARRE, COEFF_REP,
                        velocity_magnitude_pop1, ECART_TYPE_POP1, NOISE_POP_1, TAU_POP_1, PERSISTENCE_POP1, SENSITIVITY_cAMP_THRESHOLD_POP1,
                        velocity_magnitude_pop2, ECART_TYPE_POP2, NOISE_POP_2, TAU_POP_2, PERSISTENCE_POP2, SENSITIVITY_cAMP_THRESHOLD_POP2):
        parameters_file = os.path.join(PATH, "simulation_parameters.txt")
        with open(parameters_file, "w") as f:
            f.write("Paramètres de la simulation :\n")
            f.write(f"SPACE_SIZE = {SPACE_SIZE} μm\n")
            f.write(f"TIME_SIMU = {TIME_SIMU} min\n")
            f.write(f"DELTA_T = {DELTA_T} min\n")
            f.write(f"PLOT_INTERVAL = {PLOT_INTERVAL}\n")
            f.write(f"PACKING_FRACTION = {PACKING_FRACTION}\n")
            f.write(f"N_CELLS = {N_CELLS}\n\n")
            
            f.write("Paramètres d'interaction cellulaire :\n")
            f.write(f"F_REP = {F_REP}\n")
            f.write(f"F_ADH = {F_ADH}\n")
            f.write(f"R_EQ = {R_EQ}\n")
            f.write(f"R_0 = {R_0}\n")
            f.write(f"MIN_DISTANCE_INIT = {MIN_DISTANCE_INIT}\n")
            f.write(f"COEFF_CARRE = {COEFF_CARRE}\n")
            f.write(f"COEFF_REP = {COEFF_REP}\n\n")
            
            f.write("Paramètres des cellules de la population 1 :\n")
            f.write(f"velocity_magnitude_pop1 = {velocity_magnitude_pop1}\n")
            f.write(f"ECART_TYPE_POP1 = {ECART_TYPE_POP1}\n")
            f.write(f"NOISE_POP_1 = {NOISE_POP_1}\n")
            f.write(f"TAU_POP_1 = {TAU_POP_1}\n")
            f.write(f"PERSISTENCE_POP1 = {PERSISTENCE_POP1}\n")
            f.write(f"SENSITIVITY_cAMP_THRESHOLD_POP1 = {SENSITIVITY_cAMP_THRESHOLD_POP1}\n\n")
            
            
            f.write("Paramètres des cellules de la population 2 :\n")
            f.write(f"velocity_magnitude_pop1 = {velocity_magnitude_pop2}\n")
            f.write(f"ECART_TYPE_POP1 = {ECART_TYPE_POP2}\n")
            f.write(f"NOISE_POP_1 = {NOISE_POP_2}\n")
            f.write(f"TAU_POP_1 = {TAU_POP_2}\n")
            f.write(f"PERSISTENCE_POP1 = {PERSISTENCE_POP2}\n")
            f.write(f"SENSITIVITY_cAMP_THRESHOLD_POP1 = {SENSITIVITY_cAMP_THRESHOLD_POP2}\n\n")

            f.write("Paramètres du modèle FHN (cell_params) :\n")
            for key, value in cell_params.items():
                f.write(f"{key} : {value}\n")

            print("Paramètres enregistrés dans le fichier :", parameters_file)

    save_parameters(PATH, cell_params, SPACE_SIZE, TIME_SIMU, DELTA_T, PLOT_INTERVAL, PACKING_FRACTION, N_CELLS,
                F_REP, F_ADH, R_EQ, R_0, MIN_DISTANCE_INIT, COEFF_CARRE, COEFF_REP,
                velocity_magnitude_pop1, ECART_TYPE_POP1, NOISE_POP_1, TAU_POP_1, PERSISTENCE_POP1, SENSITIVITY_cAMP_THRESHOLD_POP1,
                velocity_magnitude_pop2, ECART_TYPE_POP2, NOISE_POP_2, TAU_POP_2, PERSISTENCE_POP2, SENSITIVITY_cAMP_THRESHOLD_POP2)
# =============================================================================
# Initialisation des affichages interactifs pour la cellule 0 et le champ de cAMP
# =============================================================================

plt.ion()
# Initialisation des graphiques interactifs pour la cellule et pour le champ de cAMP
fig_inter, ax1_inter = plt.subplots(figsize=(10, 6))
ax2_inter = ax1_inter.twinx()  # Axe pour cAMP
ax3_inter = ax1_inter.twinx()  # Nouvel axe pour L

ax3_inter.spines["right"].set_position(("outward", 60))

ax1_inter.set_xlabel("Temps (min)")
ax1_inter.set_ylabel("A et R", color='black')
ax2_inter.set_ylabel("cAMP (local & cumulé)", color='black')
ax3_inter.set_ylabel("L (fraction liée)", color='purple')

line_A_inter, = ax1_inter.plot([], [], 'b-', label="A (activateur)")
line_R_inter, = ax1_inter.plot([], [], 'g-', label="R (répresseur)")
line_prod_inter, = ax2_inter.plot([], [], 'r--', label="cAMP cumulé")
line_local_inter, = ax2_inter.plot([], [], 'm-.', label="cAMP local")
line_L_inter, = ax3_inter.plot([], [], color='purple', linestyle='dotted', label="L (récepteurs liés)")

lines_inter = [line_A_inter, line_R_inter, line_prod_inter, line_local_inter, line_L_inter]
labels_inter = [line.get_label() for line in lines_inter]
ax1_inter.legend(lines_inter, labels_inter, loc='upper left')

# Initialisation unique du champ de cAMP
fig_camp, ax_camp = plt.subplots(figsize=(6,6))
im_camp = ax_camp.imshow(camp_field.signal.cpu().numpy().T, origin='lower',
                         extent=[0, SPACE_SIZE, 0, SPACE_SIZE], cmap='viridis',
                         alpha=0.8, vmin=0, vmax=15)
ax_camp.set_title("Champ de cAMP en temps réel")
ax_camp.set_xlabel("Position X (μm)")
ax_camp.set_ylabel("Position Y (μm)")

def update_interactive_plot():
    """
    Met à jour le tracé interactif pour la cellule d'ID 0.
    """
    line_A_inter.set_data(cell0_time, cell0_A)
    line_R_inter.set_data(cell0_time, cell0_R)
    line_prod_inter.set_data(cell0_time, cell0_prod)
    line_local_inter.set_data(cell0_time, cell0_local)
    line_L_inter.set_data(cell0_time, cell0_L)  # Mise à jour de L

    ax1_inter.relim()
    ax1_inter.autoscale_view()
    ax2_inter.relim()
    ax2_inter.autoscale_view()
    ax3_inter.relim()  # Actualisation de l'axe pour L
    ax3_inter.autoscale_view()

    fig_inter.canvas.draw()
    fig_inter.canvas.flush_events()

def update_camp_field_map():
    """
    Met à jour la carte interactive du champ de cAMP.
    """
    global im_camp
    im_camp.set_data(camp_field.signal.cpu().numpy().T)
    ax_camp.draw_artist(ax_camp.patch)
    ax_camp.draw_artist(im_camp)
    fig_camp.canvas.flush_events()

data_list = []
cell0_time = []
cell0_A = []
cell0_R = []
cell0_local = []
cell0_prod = []
cell0_L = []  # Nouvelle liste pour stocker L au cours du temps

time = 0.0
iteration = 1
MAX_DISTANCE = np.sqrt(2 * (SPACE_SIZE / 2) ** 2)

# =============================================================================
# Boucle principale de la simulation
# =============================================================================
while time < TIME_SIMU:
    if INCLUDE_CELLS:
        for cell in cells:
            sig_val = camp_field.get_signal_at_position(cell.position)
            cell.update_state(sig_val, DELTA_T)
    
    camp_field.update(cells)
    if INITIAL_AMPc and (iteration % PLOT_INTERVAL == 0):
        plot_camp_field(camp_field, space_size=SPACE_SIZE, iteration=time)
    
    if iteration % PLOT_INTERVAL == 0:
        update_camp_field_map()
    
    if torch.isnan(camp_field.signal).any() or torch.isinf(camp_field.signal).any():
        print(f"NaN or Inf detected in cAMP signal at iteration {iteration}")
        sys.exit(1)
    
    if INCLUDE_CELLS:
        for cell in cells:
            local_camp = camp_field.get_signal_at_position(cell.position)
            if local_camp >= cell.sensitivity_threshold:
                grad_cAMP = camp_field.compute_gradient_at(cell.position)
                if torch.norm(grad_cAMP) > 0:
                    grad_cAMP = grad_cAMP / torch.norm(grad_cAMP)
                    cell.direction = (1 - cell_params['chemotaxis_sensitivity']) * cell.direction + \
                                     cell_params['chemotaxis_sensitivity'] * grad_cAMP
                    cell.direction = torch.nn.functional.normalize(cell.direction, p=2, dim=0)
        
        positions = torch.stack([cell.position for cell in cells])
        coordinates_diff = positions[:, None, :] - positions[None, :, :]
        coordinates_diff = torch.remainder(coordinates_diff - (SPACE_SIZE / 2), SPACE_SIZE) - (SPACE_SIZE / 2)
        distances = torch.stack([torch.norm(coordinates_diff[i], dim=1) for i in range(coordinates_diff.shape[0])])
        if torch.any(distances > MAX_DISTANCE):
            print("At least one distance exceeds the maximum possible.")
        
        force_field = force_field_inbox(coordinates_diff, distances, Req=R_EQ, R0=R_0,
                                        Frep=F_REP, Fadh=F_ADH, coeff_a=COEFF_CARRE, coeff_rep=COEFF_REP)
        if torch.isnan(force_field).any() or torch.isinf(force_field).any():
            print(f"NaN or Inf detected in force_field at iteration {iteration}")
            sys.exit(1)
        
        V0 = torch.tensor([cell.velocity_magnitude for cell in cells], device=device).unsqueeze(1)
        dirs = torch.stack([cell.direction for cell in cells])
        fluctuations = (torch.rand(V0.shape, device=device) - 0.5) * FLUCTUATION_FACTOR
        displacement = torch.zeros((len(cells), 2), device=device)
        if torch.isnan(displacement).any() or torch.isinf(displacement).any():
            print(f"NaN or Inf detected in displacement at iteration {iteration}")
            sys.exit(1)
        
        positions += displacement
        for idx, cell in enumerate(cells):
            cell.position = torch.remainder(cell.position + displacement[idx], SPACE_SIZE)
            new_dir = autovel(displacement[idx].unsqueeze(0), cell.direction.unsqueeze(0),
                              cell.tau, cell.noise, DELTA_T, persistence=cell.persistence)
            cell.direction = new_dir.squeeze(0)
            if torch.isnan(cell.position).any() or torch.isinf(cell.position).any():
                print(f"NaN or Inf in position of cell {cell.id} at iteration {iteration}")
                sys.exit(1)
            if torch.isnan(cell.direction).any() or torch.isinf(cell.direction).any():
                print(f"NaN or Inf in direction of cell {cell.id} at iteration {iteration}")
                sys.exit(1)
            if iteration % PLOT_INTERVAL == 0:
                data_list.append({
                    'frame': time,
                    'particle': cell.id,
                    'pop_tag': cell.pop,
                    'x': cell.position[0].item(),
                    'y': cell.position[1].item(),
                    'dir_x': cell.direction[0].item(),
                    'dir_y': cell.direction[1].item()
                })
    
    if INCLUDE_CELLS and len(cells) > 0:
        first_cell = cells[0]
        cell0_time.append(time)
        cell0_A.append(first_cell.A.item())
        cell0_R.append(first_cell.R.item())
        cell0_local.append(camp_field.get_signal_at_position(first_cell.position).item())
        cell0_prod.append(first_cell.camp_production)
        cell0_L.append(first_cell.L.item())  # Enregistrement de L
    
    if iteration % PLOT_INTERVAL == 0:
        update_interactive_plot()
        plt.pause(0.001)
    
    if PLOT and (iteration % PLOT_INTERVAL == 0):
        plot_combined_state(cells, camp_field, SPACE_SIZE, iteration, PATH, device)
    
    time += DELTA_T
    iteration += 1

# Sauvegarde des données de simulation dans un CSV
df = pd.DataFrame(data_list)
df.to_csv(os.path.join(PATH, "simulation_data.csv"), index=False)
print("Simulation terminée. Données sauvegardées.")

# Tracé final des oscillations et de la production cumulative pour la cellule 0
plt.ioff()
fig_final, ax_final = plt.subplots(figsize=(10,6))
ax_final.plot(cell0_time, cell0_A, label="A (activateur)", color='blue')
ax_final.plot(cell0_time, cell0_R, label="R (répresseur)", color='green')
ax2_final = ax_final.twinx()
ax2_final.plot(cell0_time, cell0_local, label="cAMP local", color='red', linestyle='--')
ax2_final.plot(cell0_time, cell0_prod, label="cAMP cumulé", color='black', linestyle=':')
ax_final.set_xlabel("Temps (min)")
ax_final.set_ylabel("A et R")
ax2_final.set_ylabel("cAMP (local & cumulé)")
ax_final.legend(loc='upper left')
ax2_final.legend(loc='upper right')
plt.title("Oscillations d'une cellule FHN et production cumulative de cAMP (dynamique récepteur)")
plt.tight_layout()
plt.savefig("single_cell_oscillation.png", dpi=200)
plt.show()