# Used packages
import numpy as np
from enum import Enum
from sklearn.neighbors import KDTree
import json
from matplotlib import patches, pyplot as plt
from scipy.stats import norm
from scipy import stats

from main import Branch, Individual, Field

n_alpha = 10
n_species_alpha = 10
n_species = n_alpha * n_species_alpha
t_min = -0.3
t_max = -0.01
for n in range(n_alpha):
    t = np.linspace(t_min, t_max, n_alpha)

alpha = list(map(lambda o: 2**o, t))
print(alpha)
species_alpha = [o for o in alpha for i in range(n_species_alpha)]

m = 14
l = 2 ** m
l = 10
delta0 = 0.1
delta_diff = 8
d = 5
L_av = 20

grid = Field(
    species_alpha=species_alpha,
    m=m,
    l=l,
    delta0=delta0,
    delta_diff=delta_diff,
    d=d,
    L_av=20
)

with open("test.json", "w+") as f:
        d = dict()
        for key in grid.points.keys():
            d[key] = list(map(lambda x: [x.x, x.y, x.theta], grid.points[key]))
        json.dump(d, f)
    
r_min = 0.5
r_max = 15
num_bins = 30 
R_values = np.logspace(np.log10(r_min), np.log10(r_max), num_bins + 1)

species = grid.points.values()
number_of_individuals = [len(species) for species in grid.points.values()]

for i in range(len(species)):
    print(f"Species {i} with alpha={round(species_alpha[i], 3)} has {number_of_individuals[i]} individuals.")
plt.figure(figsize=(10, 10))
colors = plt.cm.jet(np.linspace(0, 1, 5))

unique_alphas = sorted(list(set([key.split("-")[0] for key in grid.points.keys()])))
alpha_index= -1 #len(unique_alphas) // 2 # here middle alpha
alpha = unique_alphas[alpha_index]

for i in range(5):
    species_key = f"{alpha}-{i}"
    if species_key in grid.points:
        species = grid.points[species_key]
        xs = [p.x for p in species]
        ys = [p.y for p in species]
        plt.scatter(xs, ys, s=2, alpha=0.6, label=f'Species {i}', color=colors[i])

r_min = 0.5
r_max = 15
num_bins = 30 
R_values = np.logspace(np.log10(r_min), np.log10(r_max), num_bins + 1)

# get area from radius
A_values = np.pi * (R_values ** 2)

def function(q, fractional_area, initial_species_count):
    lhs = fractional_area * initial_species_count
    rhs = (q / (1 - q)) - ((initial_species_count + 1) * q ** (initial_species_count + 1)) / (1 - q ** (initial_species_count + 1))
    return lhs - rhs

# Root finding function for q, bisection method
def q_bisection(a, b, epsilon, fractional_area, initial_species_count):
    f_a = function(a, fractional_area, initial_species_count)
    f_b = function(b, fractional_area, initial_species_count)

    # Check condition for bisection method
    if f_a * f_b > 0:
        print(f"Bisection method fails for initial species count {initial_species_count}, using other method.")
        q_try = np.linspace(0, 1.01, 100000)
        q = q_alternative(q_try, fractional_area, initial_species_count)
        return q
    
    # Middle point
    c = (a + b) / 2.0
    f_c = function(c, fractional_area, initial_species_count)

    while abs(f_c) > epsilon:
        c = (a + b) / 2.0
        f_c = function(c, fractional_area, initial_species_count)
        f_a = function(a, fractional_area, initial_species_count)

        if f_c * f_a < 0:
            b = c
        else:
            a = c
            
    return c

def q_alternative(q_try, fractional_area, initial_species_count):
    lhs = fractional_area * initial_species_count
    # lhs = np.full_like(q_try, lhs)
    rhs = (q_try / (1 - q_try)) - ((initial_species_count + 1) * q_try ** (initial_species_count + 1)) / (1 - q_try ** (initial_species_count + 1))
    root_find = lhs - rhs

    y_closest = np.min(np.abs(root_find))
    q_closest = q_try[np.argmin(np.abs(root_find))]
    
    return q_closest

def extinction_probability(q, n_c, n_0):
    # print(a, q, n_c, n_0)
    return (q ** (n_c + 1) - 1) / (q ** (n_0 + 1) - 1)

# Main computation
q_try = np.linspace(0, 1.01, 100000)
n_individuals = 15569
area_original = A_values[-1]

a = 0
b = 1.01
epsilon = 1e-6

# critical_abundance = np.linspace(0, 1000, 11)
critical_abundance = np.array([0, 25, 50, 100, 500, 1000])

q_array = np.empty(len(critical_abundance), dtype=object)
extinction_prob_array = np.empty(len(critical_abundance), dtype=object)

for j in range(len(critical_abundance)):
    q = np.empty(len(A_values) - 1)
    extinction_probabilities = np.empty(len(A_values) - 1)

    for i in range(0, len(A_values) - 1):
        area = A_values[i]
        fractional_area = area / area_original
        q_value = q_bisection(a, b, epsilon, fractional_area, n_individuals)
        q[i] = q_value
        # print(f"Area: {area}, Fractional Area: {fractional_area}, q: {q_value}")
        extinction_probabilities[i] = extinction_probability(q = q_value,
                                                             n_c = critical_abundance[j],
                                                             n_0 = n_individuals)
    q_array[j] = q
    extinction_prob_array[j] = extinction_probabilities

plt.figure()

for i in range(len(critical_abundance)):
    plt.plot(A_values[:-1], extinction_prob_array[i], 'o-', label=fr'$n_c$ = {critical_abundance[i]}', markersize=3)

plt.legend()
plt.xlabel("Area")
plt.ylabel("Extinction Probability")
plt.grid(True)
plt.yscale('log')
plt.xscale('log')
plt.title(fr"Extinction probability for $n_0 = {n_individuals}$")
plt.show()

critical_abundance = 100
q_array = []
extinction_prob_array = []

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
        q_value = q_bisection(a, b, epsilon, fractional_area, n_individuals)
        q[i] = q_value
        extinction_probabilities[i] = extinction_probability(q = q_value,
                                                             n_c = critical_abundance,
                                                             n_0 = n_individuals)
    q_array.append(q)
    extinction_prob_array.append(extinction_probabilities)

plt.figure()

for i in range(len(extinction_prob_array)):
    plt.plot(A_values[:-1], extinction_prob_array[i], 'o-', label=fr'$\alpha$ = {round(species_alpha[i], 2)}, $n_0$ = {number_of_individuals[i]}', markersize=3)

plt.xlabel("Area")
plt.ylabel("Extinction Probability")
plt.grid(True)
plt.yscale('log')
plt.xscale('log')
plt.title(fr"Extinction probability, for $n_c = {critical_abundance}$")
plt.show()

alpha = species_alpha[50]
mask = [i for i in range(len(species_alpha)) if species_alpha[i] == alpha]
extinction_prob_array_alpha = [extinction_prob_array[i] for i in mask]

plt.figure()

for i in range(len(extinction_prob_array_alpha)):
    plt.plot(A_values[:-1], extinction_prob_array_alpha[i], 'o-', label=fr'$n_0$ = {number_of_individuals[mask[i]]}', markersize=3)

plt.legend()
plt.xlabel("Area")
plt.ylabel("Extinction Probability")
plt.grid(True)
plt.yscale('log')
plt.xscale('log')
plt.title(fr"Extinction probability, for $n_c = {critical_abundance}$ and $\alpha = {round(alpha, 2)}$")
plt.show()