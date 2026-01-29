# Used packages
import numpy as np
from enum import Enum
from sklearn.neighbors import KDTree
import json
from matplotlib import patches, pyplot as plt
from scipy.stats import norm
from scipy import stats

# Used modules from other python files
from main import Branch, Individual, Field, Extinction
from plots import rcCustom, rcCustom_wide

# Seed for the random number generator to ensure reproducibility
np.random.seed(3)

# A number of simulation parameters
n_alpha = 7
n_species_alpha = 10
n_species = n_alpha * n_species_alpha
t_min = -0.3
t_max = -0.01

m = 14
l = 2 ** m
l = 10
delta0 = 0.1
delta_diff = 8
d = 5
L_av = 20

# determine alpha values
t = [-0.05, -0.15, -0.25, -0.35, -0.45, -0.55, -0.65]
alpha = list(map(lambda o: 2**o, t))
species_alpha = [o for o in alpha for i in range(n_species_alpha)]

# Get simulation results from Field
grid = Field(
    species_alpha=species_alpha,
    m=m,
    l=l,
    delta0=delta0,
    delta_diff=delta_diff,
    d=d,
    L_av=L_av
)

params_extinction = Extinction(
    a=0,
    b=1.01
)

# Set evaluation radii
r_min = 0.5
r_max = 15
num_bins = 30 
R_values = np.logspace(np.log10(r_min), np.log10(r_max), num_bins + 1)

# get area from radius
A_values = np.pi * (R_values ** 2)

# Print number of individuals and alpha values per species
species = grid.points.values()
number_of_individuals = [len(species) for species in grid.points.values()]
for i in range(len(species)):
    print(f"Species {i} with alpha={round(species_alpha[i], 3)} has {number_of_individuals[i]} individuals.")

# Number of individuals before habitat loss and original area
n_individuals = 15000
area_original = A_values[-1]

# Precision threshold for bisection method
epsilon = 1e-6

# Critical abundance values
critical_abundance = np.array([0, 50, 250, 500, 1000, 5000])

# Arrays to store q values and extinction probabilities
q_array = np.empty(len(critical_abundance), dtype=object)
extinction_prob_array = np.empty(len(critical_abundance), dtype=object)

# Calculate q values and extinction probabilities for each critical abundance
for j in range(len(critical_abundance)):
    q = np.empty(len(A_values) - 1)
    extinction_probabilities = np.empty(len(A_values) - 1)

    for i in range(0, len(A_values) - 1):
        area = A_values[i]
        fractional_area = area / area_original
        q_value = params_extinction.q_numeric(fractional_area, n_individuals)
        q[i] = q_value
        extinction_probabilities[i] = params_extinction.extinction_probability(q = q_value,
                                                             n_c = critical_abundance[j],
                                                             n_0 = n_individuals)
    q_array[j] = q
    extinction_prob_array[j] = extinction_probabilities

with plt.rc_context(rc=rcCustom):
    plt.figure()
    for i in range(len(critical_abundance)):
        plt.plot(A_values[:-1], extinction_prob_array[i], 'o-', label=fr'$n_c$ = {critical_abundance[i]}', markersize=3)
    plt.legend()
    plt.xlabel("Area [a.u.]")
    plt.ylabel("Extinction Probability [a.u.]")
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')
    plt.title(fr"Extinction probability for $n_0 = {n_individuals}$")
    plt.show()

# Used critical abundance value
critical_abundance = 250

# Empty lists to store q values and extinction probabilities
q_array = []
extinction_prob_array = []

# Calculate q values and extinction probabilities for each species
for j in range(n_species):
    n_individuals = len(list(grid.points.values())[j])
    if n_individuals < critical_abundance:
        print(f"Species {j} has only {n_individuals} individuals, skipping.")
        continue

    q = np.empty(len(A_values) - 1)
    extinction_probabilities = np.empty(len(A_values) - 1)

    for i in range(0, len(A_values) - 1):
        area = A_values[i]
        fractional_area = area / area_original
        q_value = params_extinction.q_numeric(fractional_area, n_individuals)
        q[i] = q_value
        # print(f"Area: {area}, Fractional Area: {fractional_area}, q: {q_value}")
        extinction_probabilities[i] = params_extinction.extinction_probability(q = q_value,
                                                             n_c = critical_abundance,
                                                             n_0 = n_individuals)
    
    q_array.append(q)
    extinction_prob_array.append(extinction_probabilities)

# Plot extinction probabilities per species
with plt.rc_context(rc=rcCustom):
    plt.figure()
    for i in range(len(extinction_prob_array)):
        plt.plot(A_values[:-1], extinction_prob_array[i], 'o-', label=fr'$\alpha$ = {round(species_alpha[i], 2)}, $n_0$ = {number_of_individuals[i]}', markersize=3)
    plt.xlabel("Area [a.u.]")
    plt.ylabel("Extinction Probability [a.u.]")
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')
    plt.title(fr"Extinction probability, for $n_c = {critical_abundance}$")
    plt.show()

# Alpha for which to plot extinction probabilities and number of individuals
alpha = species_alpha[1]
mask = [i for i in range(len(species_alpha)) if species_alpha[i] == alpha]
n_indiv_alpha = sorted([number_of_individuals[i] for i in mask])

# Used critical abundance value, median MVP from Traill et al. (2007)
critical_abundance = 4169

# Empty lists to store q values and extinction probabilities
q_array_alpha = []
extinction_prob_alpha = []

# Calculate q values and extinction probabilities for each species with given alpha
for j in range(len(mask)):
    n_individuals = n_indiv_alpha[j]
    if n_individuals < critical_abundance:
        print(f"Species {j} has only {n_individuals} individuals, skipping.")
        continue
    q = np.empty(len(A_values) - 1)
    extinction_probabilities = np.empty(len(A_values) - 1)

    for i in range(0, len(A_values) - 1):
        area = A_values[i]
        fractional_area = area / area_original
        q_value = params_extinction.q_numeric(fractional_area, n_individuals)
        q[i] = q_value
        extinction_probabilities[i] = params_extinction.extinction_probability(q = q_value,
                                                             n_c = critical_abundance,
                                                             n_0 = n_individuals)
    q_array_alpha.append(q)
    extinction_prob_alpha.append(extinction_probabilities)

# Plot extinction probabilities per species
with plt.rc_context(rc=rcCustom):
    plt.figure()
    for i in range(len(extinction_prob_alpha)):
        plt.plot(A_values[:-1], extinction_prob_alpha[i], 'o-', label=fr'$n_0$ = {n_indiv_alpha[i]}', markersize=3)
    plt.legend()
    plt.xlabel("Area [a.u.]")
    plt.ylabel("Extinction Probability [a.u.]")
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(10 ** 1, 0.7*10 ** 3)
    plt.title(fr"Extinction probability, for $n_c = {critical_abundance}$ (median MVP) and $\alpha = {round(alpha, 2)}$")
    plt.show()