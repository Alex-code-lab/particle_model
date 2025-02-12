#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation du modèle de particules avec diffusion de cAMP et mise à jour des états cellulaires.
Auteur : Souchaud Alexandre
Date   : 2025-02-11

Ce script simule l'interaction entre des cellules qui évoluent dans un espace 2D.
Les cellules interagissent par des forces de répulsion et d'adhésion et communiquent
via la production et la diffusion d'une molécule de signalisation, le cAMP.
La dynamique cellulaire est décrite par un modèle type FitzHugh-Nagumo.
"""

# =============================================================================
# Importation des modules nécessaires
# =============================================================================
import math
import os
import sys
import torch  # Pour les calculs sur GPU/CPU via des tenseurs
import matplotlib.pyplot as plt  # Pour tracer les graphiques et visualiser les résultats
import pandas as pd  # Pour sauvegarder les données sous forme de DataFrame (CSV)
import numpy as np  # Pour des opérations mathématiques et la gestion des tableaux
import trackpy as tp  # Pour le suivi des particules (bien que non utilisé dans ce script)
import functions_analyze as lib  # Module personnalisé pour l'analyse (non exploité dans ce code)
from scipy.signal import find_peaks  # Pour détecter des pics dans des signaux (non utilisé directement)

# =============================================================================
# Choix du device pour les opérations torch (GPU si disponible, sinon CPU)
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device for torch operations:", device)

# =============================================================================
# Fonctions de calcul des forces entre cellules
# =============================================================================
def force_field_inbox(coordinates_diff, distances, Req, R0, Frep, Fadh, coeff_a=None, coeff_rep=None):
    """
    Calcule le champ de forces résultant de l'interaction entre chaque paire de cellules,
    en prenant en compte une force d'adhésion (attractive) pour des distances comprises
    entre Req et R0 et une force de répulsion (repulsive) pour des distances inférieures ou égales à Req.

    Paramètres:
      - coordinates_diff : Tenseur des différences de positions entre chaque paire de cellules (shape: [N, N, 2])
      - distances        : Tenseur des distances entre chaque paire (utilisé pour la logique, ici non exploité)
      - Req              : Rayon d'équilibre (μm) (au-delà duquel la répulsion diminue)
      - R0               : Rayon d'interaction maximale (μm)
      - Frep             : Intensité de la force de répulsion
      - Fadh             : Intensité de la force d'adhésion
      - coeff_a          : (Optionnel) Coefficient pour une force quadratique (non utilisé ici)
      - coeff_rep        : Coefficient pour ajuster la force de répulsion

    Retourne:
      - force_field : Tenseur de forme [N, 2] contenant la force nette appliquée à chaque cellule.
    """
    Rlim = 1e-6  # Valeur minimale pour la distance afin d'éviter la division par zéro
    # Calcul de la norme (distance) de chaque vecteur différence
    R = torch.norm(coordinates_diff, dim=2)
    # Pour éviter une division par zéro, remplacer les petites valeurs par Rlim
    R = torch.where(R > Rlim, R, torch.full_like(R, Rlim))
    
    # Calcul de la force d'adhésion :
    # Si la distance R est comprise entre Req et R0, on applique une force linéairement attractive.
    force = torch.where((R < R0) & (R > Req),
                        -((Fadh / (R0 - Req)) * R - Fadh * Req / (R0 - Req)),
                        torch.zeros_like(R))
    # Calcul de la force de répulsion :
    # Pour R <= Req, la force de répulsion est calculée en fonction de l'inverse de la distance.
    force = torch.where(R <= Req,
                        -Frep * coeff_rep * (1 / Req - 1 / R),
                        force)
    # Normalisation du vecteur de différences pour obtenir uniquement la direction
    norm_diff = torch.nn.functional.normalize(coordinates_diff, dim=2)
    # Somme des contributions de force sur toutes les interactions pour chaque cellule
    force_field = torch.sum(force[:, :, None] * norm_diff, dim=1)
    return force_field

# =============================================================================
# Fonctions de tracé (visualisation)
# =============================================================================
def plot_environment(cells, camp_field, space_size, axis, iteration=None):
    """
    Trace l'environnement de simulation :
      - Le champ de cAMP en arrière-plan.
      - La position de chaque cellule, colorée en fonction de leur population.

    Paramètres:
      - cells      : Liste des objets CellAgent (les cellules).
      - camp_field : Objet cAMP contenant le champ de signal.
      - space_size : Taille de l'espace de simulation (en μm).
      - axis       : Axe matplotlib où tracer l'image.
      - iteration  : (Optionnel) Indice de l'itération pour afficher le temps.

    Retourne:
      - im         : L'objet image issu de l'affichage du champ de cAMP (pour pouvoir ajouter une barre d'échelle).
    """
    axis.set_xlim(0, space_size)
    axis.set_ylim(0, space_size)
    extent = [0, space_size, 0, space_size]
    # Affichage du champ de cAMP en arrière-plan
    im = axis.imshow(camp_field.signal.T.cpu().numpy(), origin='lower', extent=extent,
                     cmap=plt.cm.viridis, alpha=0.5, vmin=0, vmax=15)
    # Extraction des positions des cellules
    x = [cell.position[0].item() for cell in cells]
    y = [cell.position[1].item() for cell in cells]
    # Choix de la couleur en fonction de la population
    colors = ['blue' if cell.pop == 'Population 1' else 'red' for cell in cells]
    axis.scatter(x, y, s=5, color=colors, alpha=0.5, edgecolors='k')
    axis.set_xlabel('Position X (μm)')
    axis.set_ylabel('Position Y (μm)')
    if iteration is not None:
        axis.set_title(f'Temps : {iteration * DELTA_T:.2f} min')
    return im

def plot_camp_field(camp_field, space_size, iteration, vmin=0, vmax=10000.0):
    """
    Affiche le champ de cAMP avec une échelle de couleurs fixe.

    Paramètres:
      - camp_field : Objet cAMP contenant le champ de signal.
      - space_size : Taille de l'espace de simulation (en μm).
      - iteration  : Indice de l'itération pour le titre.
      - vmin, vmax : Limites de l'échelle de couleur.
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
    plt.show()
    plt.close()

def plot_function(pas, Req, R0, Frep, Fadh, a, coeff_rep):
    """
    Trace les courbes des forces de répulsion et d'adhésion en fonction de la distance.

    Paramètres:
      - pas      : Pas de discrétisation pour la distance.
      - Req      : Rayon d'équilibre (μm).
      - R0       : Rayon d'interaction maximale (μm).
      - Frep     : Intensité de la force de répulsion.
      - Fadh     : Intensité de la force d'adhésion.
      - a        : (Non utilisé ici, mais éventuellement un coefficient supplémentaire).
      - coeff_rep: Coefficient ajustant la force de répulsion.
    """
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.set_xlim(0, R0)
    axis.set_ylim(-Frep, Fadh)
    # Calcul des valeurs pour la force de répulsion pour des distances entre "pas" et Req
    R_rep = np.arange(pas, Req, pas)
    force_rep = [Frep * coeff_rep * (1 / Req - 1 / R) for R in R_rep]
    axis.plot(R_rep, force_rep, label='Répulsion')
    # Calcul des valeurs pour la force d'adhésion pour des distances entre Req et R0
    R_adh = np.arange(Req, R0, pas)
    force_adh = [(Fadh / (R0 - Req)) * (R - Req) for R in R_adh]
    axis.plot(R_adh, force_adh, label='Adhésion')
    axis.set_xlabel('Distance (μm)')
    axis.set_ylabel('Force')
    axis.legend()
    plt.show()

def autovel(dX, n, tau, noise, dt, persistence):
    """
    Met à jour la direction des cellules en fonction du déplacement effectué,
    de la direction précédente et d'un bruit aléatoire.

    Paramètres:
      - dX          : Déplacement de la cellule (tenseur 2D).
      - n           : Direction précédente (tenseur 2D).
      - tau         : Constante de temps liée à la persistance.
      - noise       : Intensité du bruit ajouté à l'angle.
      - dt          : Pas de temps.
      - persistence : Facteur de persistance (pour moduler l'effet de l'historique directionnel).

    Retourne:
      - new_direction : Nouveau vecteur directionnel (tenseur 2D) normalisé.
    """
    # Normalisation du déplacement (pour éviter les valeurs trop faibles)
    dX_norm = torch.nn.functional.normalize(dX, dim=1) * 0.9999999
    # Ajustement de la persistance pour éviter une division par zéro
    if persistence == 1:
        persistence = 0.9999999
    # Calcul de l'angle initial du déplacement
    theta = torch.atan2(dX_norm[:, 1], dX_norm[:, 0])
    # Calcul de la variation d'angle basée sur la différence entre directions
    dtheta = torch.arcsin((n[:, 0] * dX_norm[:, 1] - n[:, 1] * dX_norm[:, 0])) * dt / tau
    # Ajout d'un terme de bruit (aléatoire) à l'angle
    rnd = (2 * math.pi * (torch.rand(len(dX), 1, device=device) - 0.5)) * noise * np.sqrt(dt)
    theta_update = theta + dtheta + rnd.squeeze(1)
    # Conversion de l'angle mis à jour en vecteur unitaire (direction)
    new_direction = torch.stack((torch.cos(theta_update), torch.sin(theta_update)), dim=1)
    return new_direction

# =============================================================================
# Définition des classes pour la simulation
# =============================================================================
class CellAgent:
    def __init__(self, id, pop, position, velocity, velocity_magnitude, persistence,
                 space_size, tau, noise, cell_params, sensitivity_cAMP_threshold):
        """
        Représente une cellule individuelle dans la simulation.

        Paramètres:
          - id                         : Identifiant unique de la cellule.
          - pop                        : Étiquette de la population (ex: "Population 1").
          - position                   : Position initiale (tenseur 2D).
          - velocity                   : Vecteur vitesse initiale.
          - velocity_magnitude         : Norme (vitesse) de la cellule.
          - persistence                : Facteur de persistance directionnelle.
          - space_size                 : Taille de l'espace de simulation.
          - tau                        : Constante de temps influençant la dynamique directionnelle.
          - noise                      : Intensité du bruit dans la mise à jour de la direction.
          - cell_params                : Dictionnaire contenant divers paramètres cellulaires.
          - sensitivity_cAMP_threshold : Seuil pour la détection du cAMP.
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
        # La direction initiale est obtenue en normalisant la vitesse
        self.direction = torch.nn.functional.normalize(velocity, p=2, dim=0)
        # États internes du modèle de FitzHugh-Nagumo
        self.A = torch.tensor(0.5, device=device)
        self.R = torch.tensor(0.5, device=device)
        self.cell_params = cell_params
        # Paramètres liés à la production de cAMP
        self.D = cell_params['D']
        self.a0 = cell_params['a0']
        self.af = cell_params['af']
        self.sensitivity_threshold = sensitivity_cAMP_threshold

    def update_state(self, signal_value, dt):
        """
        Met à jour les états internes A et R de la cellule en fonction du signal local de cAMP.
        
        Paramètres:
          - signal_value : Valeur du signal de cAMP à la position de la cellule.
          - dt           : Pas de temps pour la mise à jour.
        """
        # Extraction des paramètres du modèle FitzHugh-Nagumo
        a = self.cell_params['a']
        Kd = self.cell_params['Kd']
        gamma = self.cell_params['gamma']
        c0 = self.cell_params['c0']
        epsilon = self.cell_params['epsilon']
        sigma = self.cell_params['sigma']
        noise_flag = self.cell_params.get('noise', True)
        # Calcul du terme d'excitation induit par le cAMP
        I_S = a * torch.log1p(signal_value / Kd)
        # Mise à jour de l'état A (potentiellement lié au potentiel ou à l'activation)
        dA = (self.A - (self.A ** 3) / 3 - self.R + I_S) * dt
        if noise_flag:
            # Ajout d'un bruit aléatoire, pondéré par sigma et √(dt)
            dA += sigma * math.sqrt(dt) * torch.randn((), device=device)
        self.A += dA
        # Mise à jour de l'état R (peut jouer le rôle de variable de récupération)
        dR = (self.A - gamma * self.R + c0) * epsilon * dt
        self.R += dR

class Population:
    def __init__(self, num_cells, space_size, velocity_magnitude, persistence, min_distance,
                 pop_tag, ecart_type, tau, noise, cell_params, sensitivity_cAMP_threshold):
        """
        Représente une population de cellules dans la simulation.

        Paramètres:
          - num_cells                : Nombre de cellules dans la population.
          - space_size               : Taille de l'espace de simulation (en μm).
          - velocity_magnitude       : Vitesse moyenne des cellules.
          - persistence              : Facteur de persistance directionnelle.
          - min_distance             : Distance minimale entre cellules pour éviter le chevauchement.
          - pop_tag                  : Étiquette de la population (ex: "Population 1").
          - ecart_type               : Écart-type pour la distribution des vitesses.
          - tau                      : Constante de temps pour la dynamique directionnelle.
          - noise                    : Intensité du bruit dans le mouvement.
          - cell_params              : Paramètres cellulaires.
          - sensitivity_cAMP_threshold : Seuil de détection du cAMP.
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
        self.cells = []
        # Initialisation immédiate des cellules de la population
        self.initialize_cells()

    def initialize_cells(self):
        """
        Initialise les cellules en leur assignant des positions et des vitesses aléatoires,
        tout en respectant une distance minimale pour éviter le chevauchement.
        """
        global cell_id_counter
        # Génère des positions aléatoires dans l'espace de simulation
        positions = torch.rand((self.num_cells, 2), device=device) * self.space_size
        # Génère des directions aléatoires et normalisées
        directions = torch.nn.functional.normalize(torch.empty_like(positions).uniform_(-1, 1), dim=1)
        # Génère des vitesses selon une distribution normale centrée sur velocity_magnitude
        speeds = torch.normal(mean=self.velocity_magnitude, std=self.ecart_type, size=(self.num_cells,), device=device)
        if self.min_distance != 0:
            # On construit une grille pour contrôler les positions et éviter les chevauchements
            grid_size = int(np.ceil(self.space_size / self.min_distance))
            grid = [[[] for _ in range(grid_size)] for _ in range(grid_size)]
            for i, position in enumerate(positions):
                placed = False
                # Répéter jusqu'à ce qu'une position valide soit trouvée
                while not placed:
                    grid_x = int(position[0] / self.min_distance)
                    grid_y = int(position[1] / self.min_distance)
                    # Vérifier dans la cellule de la grille
                    conflicts = any(torch.norm(position - other) < self.min_distance for other in grid[grid_x][grid_y])
                    if not conflicts:
                        # Vérifier dans les cellules voisines de la grille
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                nx, ny = grid_x + dx, grid_y + dy
                                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                                    if any(torch.norm(position - other) < self.min_distance for other in grid[nx][ny]):
                                        conflicts = True
                                        break
                            if conflicts:
                                break
                    if not conflicts:
                        # Position acceptée, on l'ajoute à la grille
                        grid[grid_x][grid_y].append(position)
                        velocity = directions[i] * speeds[i]
                        self.cells.append(CellAgent(cell_id_counter, self.pop_tag, position, velocity,
                                                     speeds[i], self.persistence, self.space_size,
                                                     self.tau, self.noise, self.cell_params, self.sensitivity_cAMP_threshold))
                        cell_id_counter += 1
                        placed = True
                    else:
                        # Si conflit, générer une nouvelle position aléatoire
                        position = torch.rand(2, device=device) * self.space_size
        else:
            # Si aucune contrainte de distance minimale, initialiser directement
            for i, position in enumerate(positions):
                velocity = directions[i] * speeds[i]
                self.cells.append(CellAgent(cell_id_counter, self.pop_tag, position, velocity,
                                             speeds[i], self.persistence, self.space_size,
                                             self.tau, self.noise, self.cell_params, self.sensitivity_cAMP_threshold))
                cell_id_counter += 1

class Surface:
    def get_friction(self, position):
        """
        Retourne une valeur de friction aléatoire entre 0 et 0.2 à une position donnée.
        Cette valeur peut être utilisée pour moduler le mouvement des cellules.
        """
        return torch.empty(1, device=device).uniform_(0, 0.2).item()

class cAMP:
    def __init__(self, space_size, cell_params, initial_condition=None):
        """
        Représente le champ de cAMP, mis à jour par diffusion, dégradation et production par les cellules.

        Paramètres:
          - space_size      : Taille de l'espace de simulation (en μm).
          - cell_params     : Dictionnaire des paramètres liés au cAMP.
          - initial_condition : Condition initiale pour le champ (peut être une fonction, un tenseur ou une constante).
        """
        self.space_size = space_size
        self.grid_resolution = cell_params['grid_resolution']
        # Détermination du nombre de points de la grille en fonction de la résolution
        self.grid_size = int(space_size / self.grid_resolution)
        self.D_cAMP = cell_params['D_cAMP']
        self.aPDE = cell_params['aPDE']
        self.a0 = cell_params['a0']
        self.dx = self.grid_resolution
        self.dt = DELTA_T  # Pas de temps calculé selon le critère CFL
        # Création des axes de la grille
        x = torch.linspace(0, space_size, self.grid_size, device=device)
        y = torch.linspace(0, space_size, self.grid_size, device=device)
        self.X, self.Y = torch.meshgrid(x, y, indexing='ij')
        # Initialisation du champ de cAMP
        if initial_condition is None:
            self.signal = torch.zeros((self.grid_size, self.grid_size), device=device)
        else:
            if callable(initial_condition):
                self.signal = initial_condition(self.X, self.Y)
            elif isinstance(initial_condition, torch.Tensor):
                if initial_condition.shape == (self.grid_size, self.grid_size):
                    self.signal = initial_condition.to(device)
                else:
                    raise ValueError("La forme de initial_condition ne correspond pas à la grille.")
            else:
                self.signal = torch.full((self.grid_size, self.grid_size), initial_condition, device=device)

    def compute_laplacian(self, S):
        """
        Calcule le Laplacien de la matrice S à l'aide de différences finies
        avec conditions périodiques.

        Paramètres:
          - S : Tenseur 2D (ex: champ de cAMP).

        Retourne:
          - laplacian_S : Tenseur de même forme que S contenant le Laplacien.
        """
        laplacian_S = (torch.roll(S, shifts=1, dims=0) + torch.roll(S, shifts=-1, dims=0) +
                       torch.roll(S, shifts=1, dims=1) + torch.roll(S, shifts=-1, dims=1) -
                       4 * S) / (self.dx ** 2)
        return laplacian_S

    def update(self, cells):
        """
        Met à jour le champ de cAMP en intégrant :
          - La production locale de cAMP par les cellules.
          - La diffusion (via le Laplacien).
          - La dégradation du cAMP.

        Paramètres:
          - cells : Liste des cellules présentes dans la simulation.
        """
        # Initialisation d'une grille pour accumuler la production de cAMP
        A_grid = torch.zeros_like(self.signal)
        if cells:
            for cell in cells:
                # Conversion de la position de la cellule en indices de la grille
                x_idx = int(cell.position[0].item() / self.grid_resolution) % self.grid_size
                y_idx = int(cell.position[1].item() / self.grid_resolution) % self.grid_size
                # Production de cAMP de base
                A_grid[x_idx, y_idx] += cell.a0
                # Production additionnelle si l'état A dépasse le seuil af
                if cell.A > cell.af:
                    A_grid[x_idx, y_idx] += cell.D
        # Calcul du terme de diffusion par le Laplacien
        laplacian_S = self.compute_laplacian(self.signal)
        # Terme de dégradation proportionnel à la concentration
        degradation_term = self.aPDE * self.signal if cells else 0.0
        # Mise à jour du signal de cAMP selon l'équation de diffusion-dégradation-production
        self.signal += self.dt * (self.D_cAMP * laplacian_S - degradation_term + A_grid)
        # S'assurer que le signal reste positif
        self.signal = torch.clamp(self.signal, min=0)
        # Vérification de l'absence d'erreurs numériques
        if torch.isnan(self.signal).any() or torch.isinf(self.signal).any():
            print(f"NaN or Inf detected in cAMP signal at iteration corresponding to time {self.dt * iteration:.2f} min")
            sys.exit(1)

    def get_signal_at_position(self, position):
        """
        Retourne la valeur du signal de cAMP à la position donnée.

        Paramètres:
          - position : Position de la cellule (tenseur 2D).

        Retourne:
          - Valeur du signal de cAMP extrait de la grille.
        """
        x_idx = int(position[0].item() / self.grid_resolution) % self.grid_size
        y_idx = int(position[1].item() / self.grid_resolution) % self.grid_size
        return self.signal[x_idx, y_idx]
    
    def compute_gradient_at(self, position):
        """
        Calcule le gradient du champ de cAMP à la position donnée en utilisant une différence centrale.

        Paramètres:
          - position : Position de la cellule (tenseur 2D).

        Retourne:
          - Tenseur 2D [grad_x, grad_y] correspondant au gradient à la position donnée.
        """
        grad_x = (torch.roll(self.signal, shifts=-1, dims=0) - torch.roll(self.signal, shifts=1, dims=0)) / (2 * self.dx)
        grad_y = (torch.roll(self.signal, shifts=-1, dims=1) - torch.roll(self.signal, shifts=1, dims=1)) / (2 * self.dx)
        x_idx = int(position[0].item() / self.grid_resolution) % self.grid_size
        y_idx = int(position[1].item() / self.grid_resolution) % self.grid_size
        return torch.tensor([grad_x[x_idx, y_idx], grad_y[x_idx, y_idx]], device=device)

# =============================================================================
# Paramètres de simulation
# =============================================================================

# Contrôles et options générales
INCLUDE_CELLS = True      # Indique si les cellules doivent être prises en compte dans la simulation
INITIAL_AMPc = False      # Condition initiale pour le champ de cAMP (True pour un champ pré-défini)
PLOT = True               # Active l'affichage et la sauvegarde des images

# Paramètres de l'espace et du temps
SPACE_SIZE = 50  # μm  # Taille de l'espace de simulation (longueur d'un côté du carré)
TIME_SIMU = 1000  # min  # Durée totale de la simulation

# Paramètre pour la perception du gradient par les cellules
R_SENSING_GRAD = 4.0  # μm  # Distance sur laquelle une cellule peut détecter le gradient de cAMP

# =============================================================================
# Paramètres du modèle de FitzHugh-Nagumo et de la diffusion du cAMP
# =============================================================================

cell_params = {
    # Paramètres du modèle FitzHugh-Nagumo pour la dynamique d'excitation et de récupération des cellules
    'c0': 0.1,  # a.u.  # Terme constant influençant l'évolution de R (peut stabiliser les oscillations)
    'a': 2.0,  # a.u.  # Intensité du terme de stimulation dans l'équation de A (impacte l'excitabilité)
    'gamma': 1,  # min⁻¹  # Facteur de couplage entre A et R (contrôle la relaxation de R)
    'Kd': 0.1,  # a.u.  # Constante de dissociation pour le signal cAMP (module la sensibilité au cAMP)
    'sigma': 0.1,  # a.u.  # Amplitude du bruit aléatoire ajouté à A (simule des fluctuations)
    'epsilon': 0.2,  # min⁻¹  # Facteur d'échelle pour la mise à jour de R (contrôle la rapidité de réponse)

    # Paramètres de production de cAMP
    'D': 2500.0,  # a.u.  # Quantité de cAMP produite par une cellule lorsque A dépasse le seuil af
    'a0': 50,  # a.u.  # Production de cAMP de base, même sans excitation (maintient un niveau minimal)
    'af': 1,  # a.u.  # Seuil d'activation : production de cAMP si A dépasse cette valeur
    'noise': True,  # Active ou désactive l'ajout d'un bruit aléatoire dans la mise à jour de A

    # Paramètres de diffusion et de dégradation du cAMP
    'D_cAMP': 240.0,  # μm²/min  # Coefficient de diffusion du cAMP (propagation du signal)
    'aPDE': 15,  # min⁻¹  # Taux de dégradation du cAMP (plus il est élevé, plus le cAMP disparaît vite)

    # Paramètres liés à la discrétisation de l'espace et à la chimiotaxie
    'grid_resolution': 0.5,  # μm  # Taille d'une case de la grille de simulation
    'chemotaxis_sensitivity': 0.3  # Adimensionnel  # Sensibilité des cellules au gradient de cAMP
}

# =============================================================================
# Calcul du pas de temps (DELTA_T) pour assurer la stabilité numérique
# =============================================================================
FACTEUR_SECURITE = 0.9  # Facteur de sécurité pour garantir la stabilité numérique
DELTA_T = FACTEUR_SECURITE * (cell_params['grid_resolution'] ** 2) / (4 * cell_params['D_cAMP'])  # min
print("Intervalle de temps en min : ", DELTA_T)
PLOT_INTERVAL = int(1 / DELTA_T)  # Nombre d'itérations entre deux tracés

# =============================================================================
# Paramètres pour les interactions cellulaires
# =============================================================================

MU = 1  # μm/(a.u. × min)  # Facteur de conversion de la force en déplacement
F_REP = 40  # a.u.  # Intensité de la force répulsive entre cellules
F_ADH = 7  # a.u.  # Intensité de la force adhésive entre cellules
R_EQ = 1.1  # μm  # Rayon d'équilibre des interactions cellulaires (seuil répulsion/adhésion)
R_0 = 1.6  # μm  # Rayon maximal d'interaction entre cellules
MIN_DISTANCE_INIT = R_EQ  # μm  # Distance minimale initiale entre cellules
COEFF_CARRE = 50  # Adimensionnel  # Coefficient ajustant la force quadratique (si utilisée)
COEFF_REP = 0.5  # Adimensionnel  # Coefficient ajustant la force de répulsion
FLUCTUATION_FACTOR = 3  # Adimensionnel  # Facteur de fluctuation aléatoire du déplacement

# =============================================================================
# Détermination du nombre de cellules
# =============================================================================

PACKING_FRACTION = 0.004  # Adimensionnel  # Fraction d'empaquetage des cellules dans l'espace
N_CELLS = int((PACKING_FRACTION * SPACE_SIZE ** 2) / (math.pi * ((R_EQ / 2) ** 2)))  # Nombre total de cellules
print(N_CELLS, "cells")  # Affichage du nombre de cellules générées

# =============================================================================
# Paramètres spécifiques pour deux populations de cellules
# =============================================================================
# Ces paramètres permettent d'étudier les différences de comportement entre deux groupes cellulaires.

# Population 1 (ex: cellules moins mobiles)
velocity_magnitude_pop1 = 0 * 3  # μm/min  # Vitesse moyenne des cellules de la population 1
ECART_TYPE_POP1 = 0.3  # μm/min  # Écart-type de la vitesse de déplacement
NOISE_POP_1 = 0 * 8  # Adimensionnel  # Intensité du bruit ajouté au mouvement
TAU_POP_1 = 5  # min  # Temps caractéristique de la persistance directionnelle
PERSISTENCE_POP1 = 0  # Adimensionnel  # Niveau de persistance du mouvement
SENSITIVITY_cAMP_THRESHOLD_POP1 = 5  # a.u.  # Seuil de sensibilité au cAMP

# Population 2 (ex: cellules plus mobiles)
velocity_magnitude_pop2 = 0 * 8  # μm/min  # Vitesse moyenne des cellules de la population 2
ECART_TYPE_POP2 = 0.5  # μm/min  # Écart-type de la vitesse de déplacement
NOISE_POP_2 = 0 * 5  # Adimensionnel  # Intensité du bruit ajouté au mouvement
TAU_POP_2 = 5  # min  # Temps caractéristique de la persistance directionnelle
PERSISTENCE_POP2 = 0  # Adimensionnel  # Niveau de persistance du mouvement
SENSITIVITY_cAMP_THRESHOLD_POP2 = 5  # a.u.  # Seuil de sensibilité au cAMP


# Initialisation d'un compteur global pour l'identifiant des cellules
cell_id_counter = 0
# Création de deux populations distinctes
population1 = Population(num_cells=int(N_CELLS / 2), space_size=SPACE_SIZE,
                         velocity_magnitude=velocity_magnitude_pop1,
                         persistence=PERSISTENCE_POP1, ecart_type=ECART_TYPE_POP1,
                         min_distance=MIN_DISTANCE_INIT, pop_tag="Population 1",
                         tau=TAU_POP_1, noise=NOISE_POP_1, cell_params=cell_params,
                         sensitivity_cAMP_threshold=SENSITIVITY_cAMP_THRESHOLD_POP1)

population2 = Population(num_cells=int(N_CELLS / 2), space_size=SPACE_SIZE,
                         velocity_magnitude=velocity_magnitude_pop2,
                         persistence=PERSISTENCE_POP2, ecart_type=ECART_TYPE_POP2,
                         min_distance=MIN_DISTANCE_INIT, pop_tag="Population 2",
                         tau=TAU_POP_2, noise=NOISE_POP_2, cell_params=cell_params,
                         sensitivity_cAMP_threshold=SENSITIVITY_cAMP_THRESHOLD_POP2)

# Fusion des cellules des deux populations
cells = population1.cells + population2.cells

# Création d'une instance de Surface (actuellement non utilisée dans la simulation)
surface = Surface()

# =============================================================================
# Fonctions de conditions initiales pour le champ de cAMP
# =============================================================================
def no_gradient_initial_condition(X, Y):
    """Champ de cAMP constant partout (0.5)."""
    return torch.full_like(X, 0.5, device=device)

def gradient_initial_condition(X, Y, center_value=10000.0, radius=50.0):
    """
    Définit un champ de cAMP où une zone circulaire au centre a une concentration élevée,
    et zéro ailleurs.
    """
    X_c = X.mean().item()
    Y_c = Y.mean().item()
    distance = torch.sqrt((X - X_c) ** 2 + (Y - Y_c) ** 2)
    mask = distance <= radius
    signal = torch.zeros_like(X)
    signal[mask] = center_value
    return signal

# =============================================================================
# Initialisation du champ de cAMP
# =============================================================================
if INITIAL_AMPc:
    # Si une condition initiale pour le cAMP est souhaitée, par exemple avec un gradient défini
    camp_field = cAMP(SPACE_SIZE, cell_params, initial_condition=gradient_initial_condition)
    plot_camp_field(camp_field, space_size=SPACE_SIZE, iteration=0, vmin=0, vmax=15)
else:
    # Sinon, initialisation d'un champ nul (zéro partout)
    camp_field = cAMP(SPACE_SIZE, cell_params, initial_condition=None)

# Sauvegarde de l'état initial si l'option PLOT est activée
if PLOT:
    PATH = f'../simulations_images/v1{velocity_magnitude_pop1}v2{velocity_magnitude_pop2}a{COEFF_CARRE}coefrep{COEFF_REP}fadh{F_ADH}frep{F_REP}/'
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    else:
        print("WARNING : FOLDER ALREADY EXISTS!")
        sys.exit(0)
    # Tracé de l'état initial (positions des cellules et champ de cAMP)
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_environment(cells, camp_field, SPACE_SIZE, axis=ax, iteration=0)
    plt.savefig(f'{PATH}image_0.png', bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close()

# =============================================================================
# Boucle principale de la simulation
# =============================================================================
time = 0.0       # Temps initial de la simulation
iteration = 1    # Compteur d'itérations
# Calcul de la distance maximale possible dans l'espace (pour vérification)
MAX_DISTANCE = np.sqrt(2 * (SPACE_SIZE / 2) ** 2)
data_list = []   # Liste pour stocker les données (pour export CSV)

while time < TIME_SIMU:
    if INCLUDE_CELLS:
        # Mise à jour des états internes A et R pour chaque cellule
        for cell in cells:
            sig_val = camp_field.get_signal_at_position(cell.position)
            cell.update_state(sig_val, DELTA_T)
    
    # Mise à jour du champ de cAMP (diffusion, dégradation, production)
    camp_field.update(cells)
    if INITIAL_AMPc and (iteration % PLOT_INTERVAL == 0):
        plot_camp_field(camp_field, space_size=SPACE_SIZE, iteration=time)
    
    # Vérification que le champ de cAMP ne contient pas d'erreurs numériques
    if torch.isnan(camp_field.signal).any() or torch.isinf(camp_field.signal).any():
        print(f"NaN or Inf detected in cAMP signal at iteration {iteration}")
        sys.exit(1)
    
    if INCLUDE_CELLS:
        # Mise à jour de la direction de chaque cellule par chimiotaxie en fonction du gradient de cAMP
        for cell in cells:
            local_camp = camp_field.get_signal_at_position(cell.position)
            if local_camp >= cell.sensitivity_threshold:
                # Utilisation de la méthode compute_gradient_at de la classe cAMP
                grad_cAMP = camp_field.compute_gradient_at(cell.position)
                if torch.norm(grad_cAMP) > 0:
                    # Normalisation et combinaison de la direction précédente et du gradient détecté
                    grad_cAMP = grad_cAMP / torch.norm(grad_cAMP)
                    cell.direction = (1 - cell_params['chemotaxis_sensitivity']) * cell.direction + \
                                     cell_params['chemotaxis_sensitivity'] * grad_cAMP
                    cell.direction = torch.nn.functional.normalize(cell.direction, p=2, dim=0)
        # Calcul des forces intercellulaires à partir des positions actuelles des cellules
        positions = torch.stack([cell.position for cell in cells])
        # Calcul des différences de position entre chaque paire de cellules
        coordinates_diff = positions[:, None, :] - positions[None, :, :]
        # Correction pour conditions périodiques de l'espace
        coordinates_diff = torch.remainder(coordinates_diff - (SPACE_SIZE / 2), SPACE_SIZE) - (SPACE_SIZE / 2)
        # Calcul des distances entre les cellules
        distances = torch.stack([torch.norm(coordinates_diff[i], dim=1) for i in range(coordinates_diff.shape[0])])
        if torch.any(distances > MAX_DISTANCE):
            print("At least one distance exceeds the maximum possible.")
        # Calcul du champ de force global agissant sur chaque cellule
        force_field = force_field_inbox(coordinates_diff, distances, Req=R_EQ, R0=R_0,
                                         Frep=F_REP, Fadh=F_ADH, coeff_a=COEFF_CARRE, coeff_rep=COEFF_REP)
        if torch.isnan(force_field).any() or torch.isinf(force_field).any():
            print(f"NaN or Inf detected in force_field at iteration {iteration}")
            sys.exit(1)
        
        # Calcul du déplacement des cellules :
        # - Première composante : déplacement dû aux forces intercellulaires
        # - Deuxième composante : mouvement propulsé par la cellule avec fluctuation aléatoire
        V0 = torch.tensor([cell.velocity_magnitude for cell in cells], device=device).unsqueeze(1)
        dirs = torch.stack([cell.direction for cell in cells])
        fluctuations = (torch.rand(V0.shape, device=device) - 0.5) * FLUCTUATION_FACTOR
        displacement = MU * force_field * DELTA_T + (V0 + fluctuations) * dirs * DELTA_T
        if torch.isnan(displacement).any() or torch.isinf(displacement).any():
            print(f"NaN or Inf detected in displacement at iteration {iteration}")
            sys.exit(1)
        
        # Mise à jour des positions et directions de chaque cellule
        positions += displacement
        for idx, cell in enumerate(cells):
            # Mise à jour de la position avec conditions périodiques
            cell.position = torch.remainder(cell.position + displacement[idx], SPACE_SIZE)
            # Mise à jour de la direction via la fonction autovel
            new_dir = autovel(displacement[idx].unsqueeze(0), cell.direction.unsqueeze(0),
                              cell.tau, cell.noise, DELTA_T, persistence=cell.persistence)
            cell.direction = new_dir.squeeze(0)
            if torch.isnan(cell.position).any() or torch.isinf(cell.position).any():
                print(f"NaN or Inf in position of cell {cell.id} at iteration {iteration}")
                sys.exit(1)
            if torch.isnan(cell.direction).any() or torch.isinf(cell.direction).any():
                print(f"NaN or Inf in direction of cell {cell.id} at iteration {iteration}")
                sys.exit(1)
            # Sauvegarde des données pour l'analyse (toutes les PLOT_INTERVAL itérations)
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
    
    # Tracé périodique de l'état de la simulation en 4 sous-graphes
    if PLOT and (iteration % PLOT_INTERVAL == 0):
        # Création de la figure avec 4 sous-graphes
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)

        # =============================================================================
        # Calcul des grilles pour l'affichage des moyennes de A et R
        # =============================================================================
        grid_size = camp_field.grid_size
        A_grid = torch.zeros((grid_size, grid_size), device=device)
        R_grid = torch.zeros((grid_size, grid_size), device=device)
        cell_counts = torch.zeros((grid_size, grid_size), device=device)

        # Remplissage des grilles avec les valeurs des cellules
        for cell in cells:
            x_idx = int(cell.position[0].item() / camp_field.grid_resolution) % grid_size
            y_idx = int(cell.position[1].item() / camp_field.grid_resolution) % grid_size
            A_grid[x_idx, y_idx] += cell.A
            R_grid[x_idx, y_idx] += cell.R
            cell_counts[x_idx, y_idx] += 1

        # Éviter la division par zéro en remplaçant les cases vides par 1
        cell_counts = torch.where(cell_counts == 0, torch.ones_like(cell_counts), cell_counts)
        A_avg = A_grid / cell_counts
        R_avg = R_grid / cell_counts

        # =============================================================================
        # 1) Environnement : positions des cellules et champ de cAMP
        # =============================================================================
        im0 = plot_environment(cells, camp_field, SPACE_SIZE, axis=axes[0], iteration=iteration)
        cbar0 = fig.colorbar(im0, ax=axes[0], shrink=0.6, aspect=20, label='Concentration de cAMP')

        # =============================================================================
        # 2) Champ de cAMP complet
        # =============================================================================
        extent = [0, SPACE_SIZE, 0, SPACE_SIZE]
        im1 = axes[1].imshow(camp_field.signal.cpu().numpy().T, origin='lower', extent=extent,
                            cmap='viridis', alpha=0.8, vmin=0, vmax=15)
        axes[1].set_title(f'Champ de cAMP à l\'itération {iteration}')
        axes[1].set_xlabel('X (μm)')
        axes[1].set_ylabel('Y (μm)')
        cbar1 = fig.colorbar(im1, ax=axes[1], shrink=0.6, aspect=20, label='cAMP')

        # =============================================================================
        # 3) Moyenne locale de l'état A
        # =============================================================================
        im2 = axes[2].imshow(A_avg.cpu().numpy().T, origin='lower', extent=extent,
                            cmap='GnBu', alpha=0.8, vmin=0, vmax=A_avg.max().item() if A_avg.max() > 0 else 1)
        axes[2].set_title(f'Concentration de A à l\'itération {iteration}')
        axes[2].set_xlabel('X (μm)')
        axes[2].set_ylabel('Y (μm)')
        cbar2 = fig.colorbar(im2, ax=axes[2], shrink=0.6, aspect=20, label='A')

        # =============================================================================
        # 4) Moyenne locale de l'état R
        # =============================================================================
        im3 = axes[3].imshow(R_avg.cpu().numpy().T, origin='lower', extent=extent,
                            cmap='BuGn', alpha=0.8, vmin=0, vmax=R_avg.max().item() if R_avg.max() > 0 else 1)
        axes[3].set_title(f'Concentration de R à l\'itération {iteration}')
        axes[3].set_xlabel('X (μm)')
        axes[3].set_ylabel('Y (μm)')
        cbar3 = fig.colorbar(im3, ax=axes[3], shrink=0.6, aspect=20, label='R')

        # =============================================================================
        # Sauvegarde et affichage
        # =============================================================================
        plt.savefig(f'{PATH}combined_{iteration}.png', bbox_inches='tight', dpi=300, pad_inches=0)
        plt.close()
    
    # Mise à jour du temps et du compteur d'itération
    time += DELTA_T
    iteration += 1

# =============================================================================
# Sauvegarde finale des données de simulation dans un fichier CSV
# =============================================================================
df = pd.DataFrame(data_list)
df.to_csv(os.path.join(PATH, "simulation_data.csv"), index=False)
print("Simulation terminée. Données sauvegardées.")