#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation de la propagation d'AMPc dans une population de cellules Dictyostelium.

Ce script simule l'évolution d'un champ de signal (AMPc) diffusif couplé à une dynamique 
cellulaire. Des images du champ de signal sont sauvegardées périodiquement dans le dossier 
'output_files'. De plus, le suivi de la cellule 0 est enregistré et tracé : 
les valeurs de A, R et du signal local (AMPc produit) sont affichées en fonction du temps.

Les fonctions sont optimisées avec Numba pour le CPU et les matrices creuses de SciPy sont 
utilisées pour la diffusion.
"""

from __future__ import division
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Permet de sauvegarder les figures sans affichage interactif
import matplotlib.pyplot as plt
import os
import scipy.sparse as sparse
from numba import njit

# =============================================================================
# Fonctions de signalisation et utilitaires
# =============================================================================

@njit
def addcD(dt, lap, Dsig, J, rho, a0, D, signal, A_grid_hsided):
    """
    Calcule l'incrément du signal en combinant diffusion et réaction.

    Parameters
    ----------
    dt : float
        Pas de temps.
    lap : ndarray
        Laplacien du signal.
    Dsig : float
        Coefficient de diffusion du signal.
    J : float
        Coefficient de réaction (aPDE).
    rho : float
        Terme de production.
    a0 : float
        Constante de production.
    D : float
        Coefficient supplémentaire dans le terme source.
    signal : ndarray
        Champ de signal actuel.
    A_grid_hsided : ndarray
        Grille issue de l'état des agents (après application d'une fonction seuil).

    Returns
    -------
    ndarray
        Incrément du signal.
    """
    return dt * (Dsig * lap - J * signal + rho * a0) + dt * (rho * D * A_grid_hsided)


def accumulate_arr(coord, arr, shape):
    """
    Accumule les valeurs d'un tableau d'agents dans une grille 2D.

    Cette fonction utilise les coordonnées des agents pour sommer leurs valeurs
    dans une grille de dimensions 'shape'.

    Parameters
    ----------
    coord : ndarray
        Tableau de dimensions (2, nombre_d_agents) contenant les coordonnées.
    arr : ndarray
        Tableau des valeurs associées aux agents.
    shape : tuple
        Dimensions (n_lignes, n_colonnes) de la grille de sortie.

    Returns
    -------
    ndarray
        Grille 2D avec les sommes accumulées.
    """
    lidx = np.ravel_multi_index(coord, shape)
    return np.bincount(lidx, arr, minlength=shape[0] * shape[1]).reshape(shape)


def scale(A, B, k):
    """
    Remplit la matrice A par la matrice B en créant des blocs de taille k x k.

    Parameters
    ----------
    A : ndarray
        Matrice de destination.
    B : ndarray
        Matrice source.
    k : int
        Facteur d'échelle (taille du bloc).

    Returns
    -------
    ndarray
        Matrice A modifiée.
    """
    Y, X = A.shape
    for y in range(k):
        for x in range(k):
            A[y:Y:k, x:X:k] = B
    return A


def downscale_array(arr, scale_factor):
    """
    Réduit la taille d'un tableau 'arr' par un facteur 'scale_factor'
    en faisant la moyenne sur des blocs.

    Parameters
    ----------
    arr : ndarray
        Tableau 2D à réduire.
    scale_factor : int
        Facteur de réduction.

    Returns
    -------
    ndarray
        Tableau réduit par moyenne sur blocs.
    """
    new_shape = (arr.shape[0] // scale_factor, scale_factor,
                 arr.shape[1] // scale_factor, scale_factor)
    return arr.reshape(new_shape).mean(axis=(1, 3))


def calc_square_laplacian_noflux_matrix(size):
    """
    Construit la matrice laplacienne 2D avec conditions de bord « no flux »
    pour une grille de dimensions 'size'.

    Parameters
    ----------
    size : tuple
        Dimensions (n_lignes, n_colonnes) de la grille.

    Returns
    -------
    csr_matrix
        Matrice laplacienne sous format compressé.
    """
    nrows, ncols = size
    # Matrice identité pour la direction x
    Ix = sparse.eye(ncols)
    # Vecteur de 1 pour construire la matrice de différences en x
    e_x = np.ones(ncols)
    Ax = sparse.diags([-e_x, 2 * e_x, -e_x], [-1, 0, 1], shape=(ncols, ncols))
    # Matrice identité pour la direction y
    Iy = sparse.eye(nrows)
    e_y = np.ones(nrows)
    Ay = sparse.diags([-e_y, 2 * e_y, -e_y], [-1, 0, 1], shape=(nrows, nrows))
    # Produit de Kronecker pour obtenir le laplacien 2D
    lap = sparse.kron(Ay, Ix) + sparse.kron(Iy, Ax)
    return lap.tocsr()


# =============================================================================
# Définition des paramètres par défaut
# =============================================================================

# Paramètres de la grille
grid_params_default = {
    'dx': 0.5,               # Pas spatial : distance entre deux points consécutifs de la grille (ex. en unités arbitraires ou microns)
    'D_sig': 5,              # Coefficient de diffusion du signal : contrôle la vitesse à laquelle le signal se propage sur la grille
    'box_size_x': 50,        # Taille totale de la grille en x : longueur horizontale de la zone de simulation
    'box_size_y': 50,        # Taille totale de la grille en y : longueur verticale de la zone de simulation
    'agent_dim': 1,          # Dimension caractéristique d'un agent : utilisée pour le downscaling ou pour mapper les agents sur la grille
    'num_agents': 15000      # Nombre total d'agents (cellules) présents dans la simulation
}

# Paramètres cellulaires
cell_params_default = {
    'c0': 0.1,                 # Terme constant c0 : peut représenter une valeur de base ou un seuil dans la dynamique cellulaire
    'a': 0.11,               # Coefficient d'activation : module la réponse des cellules au signal (ex. influence la sensibilité ou l'amplification)
    'gamma': 0.5,            # Coefficient d'inhibition ou de régulation négative : contrôle la rétroaction inhibitrice sur l'activation cellulaire
    'Kd': 1e-5,              # Constante de dissociation (Kd) : détermine la sensibilité de la réponse cellulaire au signal (plus Kd est faible, plus la cellule est sensible)
    'sigma': 0.5,            # Amplitude du bruit dans la dynamique cellulaire : permet d'introduire des fluctuations stochastiques pour modéliser l'incertitude
    'epsilon': 0.2,          # Paramètre d'échelle temporelle pour la variable R : ajuste la rapidité de la réponse ou la dynamique de l'inhibition
    'cStim': 10,            # Niveau de stimulation externe : valeur de concentration ou d'intensité du stimulus exogène appliqué aux cellules
    'aPDE': 100,             # Coefficient dans l'équation de réaction-diffusion (aPDE) : module la dégradation ou l'interaction entre le signal et la cellule
    'rho': 0.1,              # Taux de production du signal : détermine la quantité de signal produit par les cellules (source de signal)
    'D': 10,                 # Coefficient additionnel lié à la production du signal par les agents : différent du D_sig, il module la contribution de l'état des agents à la production du signal
    'a0': 1,                 # Terme constant dans la production du signal : représente la valeur de base de production, indépendamment des interactions
    'af': 0                  # Paramètre additionnel éventuel pour ajuster une influence supplémentaire (souvent laissé à zéro si non utilisé)
}

# =============================================================================
# Classe de simulation de la population
# =============================================================================

class DictyPop:
    """
    Classe pour simuler la propagation d'AMPc et la dynamique cellulaire
    dans une population de cellules Dictyostelium.

    Attributes
    ----------
    g_params : dict
        Paramètres de la grille.
    c_params : dict
        Paramètres cellulaires.
    T : float
        Temps total de simulation.
    dt : float
        Pas de temps.
    Tsteps : int
        Nombre de pas de simulation.
    signal_size : tuple
        Dimensions de la grille de signal.
    fluxes : ndarray
        Termes de flux aux bords.
    A : ndarray
        État A de chaque agent.
    R : ndarray
        État R de chaque agent.
    signal : ndarray
        Champ de signal (AMPc).
    """
    
    def __init__(self, T, save_every, g_params=grid_params_default,
                 c_params=cell_params_default, noise=True, progress_report=True,
                 save_data=True):
        """
        Initialise la simulation.

        Parameters
        ----------
        T : float
            Temps total de simulation (unités de temps).
        save_every : float
            Intervalle de sauvegarde en temps.
        g_params : dict, optional
            Paramètres de la grille (par défaut grid_params_default).
        c_params : dict, optional
            Paramètres cellulaires (par défaut cell_params_default).
        noise : bool, optional
            Si True, ajoute du bruit à la dynamique (par défaut True).
        progress_report : bool, optional
            Si True, affiche l'avancement de la simulation (par défaut True).
        save_data : bool, optional
            Si True, sauvegarde l'évolution des états (par défaut True).
        """
        self.g_params = g_params
        self.c_params = c_params
        self.T = T            # Temps total de simulation
        self.ts = 0           # Temps courant
        # Choix du pas de temps (dépend de dx et de D_sig)
        self.dt = self.g_params['dx']**2 / (8 * self.g_params['D_sig'])
        self.Tsteps = int(self.T / self.dt)
        # Taille de la grille de signal
        self.signal_size = (int(self.g_params['box_size_x'] / self.g_params['dx']),
                            int(self.g_params['box_size_y'] / self.g_params['dx']))
        self.fluxes = np.zeros(4)  # Pour d'éventuels flux sur les bords
        self.progress_report = progress_report
        self.save_every = save_every
        self.save_data = save_data

        # Rapport entre la taille d'agent et le pas spatial (pour downscaling)
        self.agent_ratio = int(self.g_params['agent_dim'] / self.g_params['dx'])
        if self.agent_ratio < 1:
            self.agent_ratio = 1
        # Grille pour les agents et grille « agrandie » pour interpolation
        self.A_grid = np.zeros((self.signal_size[0] // self.agent_ratio,
                                self.signal_size[1] // self.agent_ratio))
        self.A_grid_big = np.zeros(self.signal_size)
        # Initialisation du champ de signal (AMPc)
        self.signal = np.zeros(self.signal_size)
        self.noise_flag = noise

        # Initialisation de l'état cellulaire (deux composantes : A et R)
        num_agents = self.g_params['num_agents']
        cell_state = np.random.normal(loc=0.0, scale=2, size=(2, num_agents))
        self.A = cell_state[0, :]
        self.R = cell_state[1, :]

        self.D = float(self.c_params['D'])
        self.dsignal = np.zeros(self.signal.shape)
        self.dA = np.zeros(self.A.shape)
        self.dR = np.zeros(self.R.shape)

        # Initialisation des coordonnées (sur la grille réduite) de chaque agent
        grid_shape = self.A_grid.shape
        self.coord = np.zeros((2, num_agents), dtype=int)
        for agent in range(num_agents):
            self.coord[0, agent] = np.random.randint(0, grid_shape[0])
            self.coord[1, agent] = np.random.randint(0, grid_shape[1])

        # Construction de la matrice laplacienne 2D avec conditions de bord « no flux »
        if min(self.signal_size) > 1:
            self.lap_mat = calc_square_laplacian_noflux_matrix(self.signal_size)
        else:
            # Cas 1D (rare)
            n = max(self.signal_size)
            self.lap_mat = sparse.diags([np.ones(n-1), -2 * np.ones(n), np.ones(n-1)],
                                        [-1, 0, 1], format="csr")
            self.lap_mat += sparse.diags(np.array([1] + [0]*(n-2) + [1]), format="csr")

        # Sauvegarde de l'évolution si demandé
        if self.save_data:
            num_save = int(self.T // self.save_every) + 1
            self.A_saved = np.zeros((self.A_grid.shape[0], self.A_grid.shape[1], num_save))
            self.R_saved = np.zeros((self.A_grid.shape[0], self.A_grid.shape[1], num_save))
            self.S_saved = np.zeros((self.signal.shape[0], self.signal.shape[1], num_save))
            self.A_saved[:, :, 0] = self.getAGrid()
            self.R_saved[:, :, 0] = self.getRGrid()
            self.S_saved[:, :, 0] = self.signal

    def getAGrid(self):
        """
        Accumule l'état A des agents dans la grille réduite.

        Returns
        -------
        ndarray
            Grille 2D des valeurs de A.
        """
        # S'assure que A ne contient pas de NaN avant accumulation
        safe_A = np.nan_to_num(self.A, nan=0.0)
        return accumulate_arr(self.coord, safe_A, self.A_grid.shape)

    def getRGrid(self):
        """
        Accumule l'état R des agents dans la grille réduite.

        Returns
        -------
        ndarray
            Grille 2D des valeurs de R.
        """
        safe_R = np.nan_to_num(self.R, nan=0.0)
        return accumulate_arr(self.coord, safe_R, self.A_grid.shape)

    def setSignal(self, signal):
        """
        Définit le champ de signal.

        Parameters
        ----------
        signal : ndarray
            Nouvelle grille de signal.
        """
        assert signal.shape == self.signal.shape and signal.dtype == self.signal.dtype, \
            "Input signal incorrect shape or type"
        self.signal = np.array(signal)

    def setCellState(self, state):
        """
        Définit l'état cellulaire pour tous les agents.

        Parameters
        ----------
        state : ndarray
            Tableau de forme (2, nombre_d_agents) contenant les états.
        """
        assert state.shape[0] == 2 and state.shape[1] == self.A.shape[0], \
            "Input cell state incorrect shape or type"
        self.A = np.array(state[0, :])
        self.R = np.array(state[1, :])

    def setFluxes(self, fluxes):
        """
        Définit les flux appliqués aux bords du domaine.

        Parameters
        ----------
        fluxes : ndarray
            Tableau de forme (4,) contenant les flux sur chaque bord.
        """
        assert fluxes.shape == self.fluxes.shape and fluxes.dtype == self.fluxes.dtype, \
            "Input fluxes incorrect shape or type"
        self.fluxes = np.array(fluxes)

    def getdA(self):
        """
        Calcule la dérivée de A pour chaque agent.

        La dérivée de A est calculée en fonction de la dynamique interne 
        (effet auto-activant avec saturation et inhibition par R) et d'une 
        contribution dépendant du signal local.

        Pour éviter des erreurs dans np.log1p, le signal local est d'abord clipé 
        afin que le ratio (signal/Kd) soit toujours ≥ -0.999 et ≤ 1e6.

        Returns
        -------
        ndarray
            Incrément de A pour chaque agent.
        """
        # Réduction du champ de signal vers la grille des agents (moyenne sur blocs)
        agent_signal = downscale_array(self.signal, self.agent_ratio)
        # Récupération de la valeur locale du signal pour chaque agent
        local_signal = agent_signal[self.coord[0, :], self.coord[1, :]]
        # On clippe le signal local pour éviter que la division ne produise des valeurs énormes
        max_signal = 1e6 * self.c_params['Kd']
        min_signal = -0.999 * self.c_params['Kd']
        local_signal = np.clip(local_signal, min_signal, max_signal)
        # Calcul du ratio signal/Kd
        ratio = local_signal / self.c_params['Kd']
        # Clamp de sécurité sur le ratio
        ratio = np.clip(ratio, -0.999, 1e6)
        
        dA = ((self.A - (self.A**3) / 3 - self.R) +
              (self.c_params['a'] * np.log1p(ratio))) * self.dt
        # Assurer qu'aucune valeur NaN ne soit renvoyée
        return np.nan_to_num(dA, nan=0.0)

    def getdR(self):
        """
        Calcule la dérivée de R pour chaque agent.

        Returns
        -------
        ndarray
            Incrément de R pour chaque agent.
        """
        dR = ((self.A - self.c_params['gamma'] * self.R + self.c_params['c0'])
              * self.c_params['epsilon']) * self.dt
        return np.nan_to_num(dR, nan=0.0)

    def getLapC(self):
        """
        Calcule la laplacienne du champ de signal.

        Returns
        -------
        ndarray
            Laplacien du signal sous forme d'une grille 2D.
        """
        flat_signal = self.signal.flatten()
        laplacian = (1 / (self.g_params['dx']**2)) * self.lap_mat.dot(flat_signal)
        return laplacian.reshape(self.signal.shape)

    def update(self):
        """
        Met à jour l'état des agents et le champ de signal pour un pas de temps.

        Cette méthode :
          - Met à jour les états A et R des agents.
          - Met à jour le champ de signal via une équation diffusion-réaction.
          - Applique des conditions aux bords via les flux spécifiés.

        Returns
        -------
        ndarray
            Nouveau champ de signal.
        """
        self.ts += self.dt
        self.dA = self.getdA()
        if self.noise_flag:
            self.dA += self.c_params['sigma'] * np.random.normal(
                loc=0.0, scale=np.sqrt(self.dt), size=self.A.shape)
        self.dR = self.getdR()

        # Mise à jour de la grille associée aux agents
        self.A_grid = accumulate_arr(self.coord, self.A, self.A_grid.shape)
        self.A_grid_big = scale(self.A_grid_big, self.A_grid, self.agent_ratio)
        self.dsignal = addcD(self.dt, self.getLapC(), self.g_params['D_sig'],
                             self.c_params['aPDE'], self.c_params['rho'],
                             self.c_params['a0'], self.c_params['D'],
                             self.signal, np.heaviside(self.A_grid_big, 0.5))

        # Mise à jour des états et du champ de signal
        self.A += self.dA
        self.R += self.dR
        self.signal += self.dsignal

        # Optionnel : clamp des variables pour éviter la propagation de NaN ou d'infinis
        self.A = np.clip(np.nan_to_num(self.A, nan=0.0, posinf=10, neginf=-10), -10, 10)
        self.R = np.clip(np.nan_to_num(self.R, nan=0.0, posinf=10, neginf=-10), -10, 10)
        # Lignes de clamp sur self.signal sont commentées pour étudier la divergence naturelle

        # Application des conditions aux bords (flux)
        dx = self.g_params['dx']
        self.signal[0, 1:-1] -= self.fluxes[0] / dx
        self.signal[-1, 1:-1] -= self.fluxes[1] / dx
        self.signal[1:-1, 0] -= self.fluxes[2] / dx
        self.signal[1:-1, -1] -= self.fluxes[3] / dx
        self.signal[0, 0] -= (self.fluxes[0] + self.fluxes[2]) / (2 * dx)
        self.signal[-1, 0] -= (self.fluxes[1] + self.fluxes[2]) / (2 * dx)
        self.signal[0, -1] -= (self.fluxes[0] + self.fluxes[3]) / (2 * dx)
        self.signal[-1, -1] -= (self.fluxes[1] + self.fluxes[3]) / (2 * dx)
        return self.signal


# =============================================================================
# Fonction principale de simulation et tracé du suivi d'une cellule
# =============================================================================

def run_simulation():
    """
    Exécute la simulation de propagation d'AMPc et sauvegarde périodiquement 
    les images du champ de signal.

    La simulation s'exécute sur un temps total T et les images sont sauvegardées 
    dans le dossier 'output_files'. De plus, le suivi de la cellule 0 est enregistré
    (valeurs de A, R et du signal local) et tracé en fonction du temps.
    """
    # Paramètres de la simulation
    T = 10              # Temps total de simulation (unités arbitraires)
    save_every = 1      # Intervalle de sauvegarde (en temps)
    sim = DictyPop(T, save_every)
    steps = sim.Tsteps

    output_dir = "output_files"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Listes pour enregistrer le temps, A, R et le signal local pour la cellule 0
    time_record = []
    A_record = []
    R_record = []
    signal_record = []  # Signal local (AMPc) associé à la cellule 0

    print(f"Début de la simulation sur {steps} étapes (dt = {sim.dt:.4f}) ...")
    for step in range(steps):
        S = sim.update()  # Mise à jour de la simulation

        # Enregistrement du temps et des valeurs de A et R pour la cellule 0
        time_record.append(sim.ts)
        A_record.append(sim.A[0])
        R_record.append(sim.R[0])
        # Calcul du signal local pour la cellule 0 à partir du champ global :
        # On réduit le champ de signal vers la grille des agents et on récupère la valeur
        local_signal_field = downscale_array(sim.signal, sim.agent_ratio)
        # Utilisation des coordonnées de la cellule 0 (déjà dans la grille réduite)
        signal_record.append(local_signal_field[sim.coord[0, 0], sim.coord[1, 0]])

        # Sauvegarde d'une image du champ de signal tous les 100 pas
        if step % 100 == 0:
            plt.figure()
            plt.imshow(S, cmap='viridis', origin='lower')
            plt.title(f"t = {sim.ts:.2f}")
            plt.colorbar(label="Signal")
            filename = os.path.join(output_dir, f"signal_{step:05d}.png")
            plt.savefig(filename)
            plt.close()
            print(f"Étape {step}/{steps} – image sauvegardée : {filename}")

    print("Simulation terminée.")

    # Tracé du suivi de la cellule 0 pour A, R et le signal local (AMPc produit)
    plt.figure(figsize=(10, 6))
    plt.plot(time_record, A_record, label="A (cellule 0)", color='blue')
    plt.plot(time_record, R_record, label="R (cellule 0)", color='red')
    plt.plot(time_record, signal_record, label="Signal local (AMPc)", color='green')
    plt.xlabel("Temps")
    plt.ylabel("Valeur d'état / Signal")
    plt.title("Suivi de la cellule 0 : A, R et Signal local (AMPc)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    trace_filename = os.path.join(output_dir, "cellule_0_suivi.png")
    plt.savefig(trace_filename)
    plt.close()
    print(f"Tracé du suivi de la cellule 0 sauvegardé dans {trace_filename}")


# =============================================================================
# Lancement de la simulation
# =============================================================================

if __name__ == "__main__":
    run_simulation()