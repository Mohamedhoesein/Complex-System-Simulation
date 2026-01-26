from enum import Enum
import json

from matplotlib.pylab import norm
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from scipy import stats
from sklearn.neighbors import KDTree

class Branch(Enum):
    """
    How to branch for the population.
    """
    LEFT = 1
    RIGHT = 2
    BOTH = 4

class Individual:
    """
    X Y coordinates of an individual.
    """
    def __init__(self, x: float, y: float, theta: float):
        """
        Initialise a position for an individual.
        
        :param x: X coordinate of the individual
        :type x: float
        :param y: Y coordinate of the individual
        :type y: float
        """
        self.x = x
        self.y = y
        self.theta = theta

class Field:
    """
    Container for different individuals.
    """
    def __init__(
            self,
            species_alpha: list[float],
            m: int,
            l: float,
            delta0: float,
            delta_diff: float,
            d: float,
            L_av: int
        ):
        """
        Initialise an area.
        
        :param x: The maximum x coordinate, assuming the range [0,x] is valid.
        :type x: float
        :param y: The maximum y coordinate, assuming the range [0,x] is valid.
        :type y: float
        :param species_alpha: The alpha for each species.
        :type species_alpha: list[float]
        :param m: The maximum number of iterations.
        :type m: int
        :param l: The starting length of an branch.
        :type l: float
        :param delta0: The initial delta.
        :type delta0: float
        :param delta_diff: The change in delta from which to draw a delta for a new brench.
        :type delta_diff: float
        :param d: The maximum diameter of the initial point.
        :type d: float
        :param L_av: averaging area side length
        :type L_av: int
        """
        assert m > 0
        assert l > 0
        assert d > 0
        self.species_alpha = species_alpha
        self.m = m
        self.l = l
        self.delta0 = delta0
        self.delta_diff = delta_diff
        self.d = d
        self.points = dict()
        previous_alpha = 0
        count = 0
        for alpha in self.species_alpha:
            if previous_alpha == alpha:
                count += 1
            else:
                previous_alpha = alpha
                count = 0
            self.points[f"{alpha}-{count}"] = self.generate_species(alpha) # (alpha, species index): [list of individuals]
        
        self.L_av = L_av
        self.omega_range = L_av / 2

    def generate_species(self, alpha: float) -> list[Individual]:
        """
        Generate a species with a given alpha.
        
        :param alpha: The alpha parameter for the current species.
        :type alpha: float
        :return: The set of individuals for the current species.
        :rtype: list[Individual]
        """
        branch_options = [Branch.LEFT, Branch.RIGHT, Branch.BOTH]
        p = [1-alpha, 1-alpha, 2*alpha-1]

        # initial point
        theta0 = np.random.uniform(0, 2*np.pi)

        r0 = self.get_initial_point()
        x = np.array([r0.x])
        y = np.array([r0.y])
        thetas = np.array([theta0])

        l = self.l

        for n in range(1, self.m+1):
            l /= 2

            delta_max = self.delta0 * (self.delta_diff)**(2*(n//2)/self.m)

            point_count = len(x)

            deltas = np.random.uniform(-delta_max, delta_max, point_count)
            new_thetas = thetas + deltas + np.pi/2
            branches = np.random.choice(branch_options, p=p, size=point_count)
            left = (branches == Branch.LEFT) | (branches == Branch.BOTH)
            right = (branches == Branch.RIGHT) | (branches == Branch.BOTH)
            left_x = x[left] + l*np.cos(new_thetas[left])
            left_y = y[left] + l*np.sin(new_thetas[left])
            right_x = x[right] + l*np.cos(new_thetas[right] + np.pi)
            right_y = y[right] + l*np.sin(new_thetas[right] + np.pi)
            x = np.concatenate([left_x, right_x])
            y = np.concatenate([left_y, right_y])
            thetas = np.concatenate([new_thetas[left], new_thetas[right]])

        return list(map(lambda x: Individual(x[0], x[1], x[2]), np.stack([x, y, thetas], axis=-1)))

    def get_initial_point(self) -> Individual:
        """
        Create the initial point for a tree.
        
        :return: The initial point.
        :rtype: Individual
        """
        middle = Individual(0, 0, 0)
        # x+1 and y+1 are needed because we take [0,x] as a valid range which has x+1 integers,
        # and the same applies to y.
        offset = np.random.rand()*self.d
        radian = np.random.rand()*2*np.pi
        return Individual(
            middle.x + offset*np.cos(radian),
            middle.y + offset*np.sin(radian),
            0
        )
    
    def species_area_curve(self, R_values: np.ndarray, n_samples: int = 1000) -> np.ndarray:    # optimized   
        """
        Compute Species Area Relationship S_C(R).
        
        :param R_values: array of radii.
        :type R_values: np.ndarray
        :param n_samples: number of sample points to use for estimation.
        :type n_samples: int
        :return: S_C(R) values for each radius in R_values.
        :rtype: np.ndarray
        """
        # initialize the total sum array
        S_C = np.zeros_like(R_values, dtype=float)
        
        # generate n sample points in omega
        sample_x = np.random.uniform(-self.omega_range, self.omega_range, n_samples)
        sample_y = np.random.uniform(-self.omega_range, self.omega_range, n_samples)
        sample_points = np.column_stack([sample_x, sample_y])
        
        for species in self.points.values(): # dictionary
            if len(species) == 0:
                continue

            # find nearest neighbor distances using KDTree
            points = np.array([(p.x, p.y) for p in species])
            tree = KDTree(points)
            
            # for each sample point, find distance to nearest individual of this species
            dists, _ = tree.query(sample_points)
            dists = dists.ravel()  # flatten to 1D array
            # count fraction where minimum distance <= R: proximity function (vectorized)
            proximity_matrix = dists[:, None] <= R_values[None, :]
            
            # average over samples to get Fs(R) for this species
            Fs_R = np.mean(proximity_matrix, axis=0)
            
            # add this species' contribution to the total
            S_C += Fs_R
            
        return S_C
    
    def compute_correlation_function(self, species: dict, r_bins: np.ndarray) -> np.ndarray:
        # consider points within omega
        points_in_omega = [p for p in species
                           if -self.omega_range <= p.x <= self.omega_range 
                           and -self.omega_range <= p.y <= self.omega_range]
        
        if len(points_in_omega) < 2:
            return np.zeros(len(r_bins) - 1)
        
        # calculate distances between all pairs
        points_array = np.array([(p.x, p.y) for p in points_in_omega])
        n = len(points_array)
        dx = points_array[:, None, 0] - points_array[None, :, 0]
        dy = points_array[:, None, 1] - points_array[None, :, 1]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        
        # consider unique pairs
        d_flat = dist_matrix[np.triu_indices(n, k=1)]
        
        # build historgram
        counts, _ = np.histogram(d_flat, bins=r_bins)
        
        # divide by area of annulus for density estimation
        # where area of annulus = pi * (r_outer^2 - r_inner^2)
        r_inner = r_bins[:-1]
        r_outer = r_bins[1:]
        bin_areas = np.pi * (r_outer**2 - r_inner**2)
        
        # avoid division by zero if a bin has 0 area 
        rho_raw = counts / bin_areas
        rho_raw = np.nan_to_num(rho_raw)

        # normalize by rho_s(1) (first bin)
        if rho_raw[0] > 0:
            rho_s = rho_raw / rho_raw[0]
        else:
            # if the first bin is empty, we can't normalize relative to it.
            rho_s = np.zeros_like(rho_raw)

        return rho_s
    # group list of list by alpha
    def get_correlations_grouped_by_alpha(self, r_bins: np.ndarray) -> dict:
        """
        :return: A dictionary mapping each unique alpha to its average correlation function.
        :rtype: dict
        """
        grouped_results = {}
        unique_alphas = list(set(map(lambda key: key.split("-")[0], self.points.keys()))) #sorted(list(set(self.species_alpha)))
        # for each alpha
        for target_alpha in unique_alphas:
            
            current_alpha_correlations = []
            
            # associate alpha with its species points
            for alpha_val in self.points.keys():
                if alpha_val.split("-")[0] == target_alpha:
                    rho = self.compute_correlation_function(self.points[alpha_val], r_bins)
                    # add valid results
                    if np.sum(rho) > 0: 
                        current_alpha_correlations.append(rho)
            
            # compute average correlation for this group
            if len(current_alpha_correlations) > 0:
                avg_rho = np.mean(current_alpha_correlations, axis=0)
                grouped_results[float(target_alpha)] = avg_rho
            else:
                grouped_results[float(target_alpha)] = None
                
        return grouped_results

def main():
    t = [-0.01, -0.03, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.40, -0.50, -0.60, -0.65]
    alpha_values = list(map(lambda o: 2**o, t))
    # alpha_values = np.linspace(0.55, 0.90, 6)[::-1]
    species_alpha = [o for o in alpha_values for i in range(10)]
    
    grid = Field(
        species_alpha=species_alpha,
        m=14,
        l=10,     # force fractal structure inside the window 
        delta0=0.1,        
        delta_diff=8,      
        d=5,               
        L_av=20            
    )
    with open("test.json", "w+") as f:
        d = dict()
        for key in grid.points.keys():
            d[key] = list(map(lambda x: [x.x, x.y, x.theta], grid.points[key]))
        json.dump(d, f)
    r_min = 0.1
    r_max = grid.L_av / 2   # limit to half of L_av 
    num_bins = 30 
    R_values = np.logspace(np.log10(r_min), np.log10(r_max), num_bins + 1)

    # plot correlation functions
    results_corr_dict = grid.get_correlations_grouped_by_alpha(R_values)

    plt.figure(figsize=(8, 6))
    for alpha, rho_avg in results_corr_dict.items():
        if rho_avg is not None:
            r_centers = (R_values[:-1] + R_values[1:]) / 2
            plt.plot(r_centers, rho_avg, marker='o', label=f'Alpha: {alpha:.3f}')
    
    plt.xscale('log')
    plt.yscale('log') 
    plt.xlabel(r'Distance $r$')
    plt.ylabel(r'Correlation $\rho_s(r) / \rho_s(1)$')
    plt.title('Spatial Correlation (Log-Log)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig("spatial_correlation.png")
    plt.show()

    # plot species distribution (first 5 species)
    plt.figure(figsize=(10, 10))
    colors = plt.cm.jet(np.linspace(0, 1, 5))
    for key in grid.points.keys():
        k = key.split("-")[0]
        for i in range(5):
            species = grid.points[f"{k}-{i}"]
            xs = [p.x for p in species]
            ys = [p.y for p in species]
            plt.scatter(xs, ys, s=2, alpha=0.6, label=f'Species {i}', color=colors[i])
        break

    plt.title(f"Species Distribution (First 10 Species)\nWindow Size $L_{{av}}={grid.L_av}$")
    plt.xlabel("X")
    plt.ylabel("Y ")
    plt.axis('equal') 
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.savefig("species_distribution.png")
    plt.show()

    # plot Species-Area Relationship
    S_values = grid.species_area_curve(R_values, n_samples=2000)
    # get area from radius
    A_values = np.pi * (R_values ** 2)
    
    log_A = np.log10(A_values)
    log_S = np.log10(S_values)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_A, log_S)
    # obtained power law 
    plt.figure(figsize=(8, 6))
    plt.loglog(A_values, S_values, 'o', 
               color='black', markersize=5, label='Simulated data')
    # fit line for slope
    fit_line = 10**intercept * A_values**slope
    plt.loglog(A_values, fit_line, 'r--', linewidth=2,
               label=f'Power-law fit: z={slope:.3f}')
    
    plt.xlabel(r"Sampling Area $A$ ($A = \pi R^2$)")
    plt.ylabel(r"Number of Species $S_C(A)$")
    plt.title("Species-Area Relationship (SAR)")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.savefig("sar.png")
    plt.show()

    # plot lognormal distribution of abundance
    plt.figure(figsize=(8, 6))
    # get abundances
    abundances = np.array([len(species) for species in grid.points.values()])
    abundances = abundances[abundances > 0]
    # octave bins
    max_abundance = max(abundances)
    num_octaves = int(np.ceil(np.log2(max_abundance))) + 1
    bins = [2**i for i in range(num_octaves + 1)]
    
    counts, bin_edges = np.histogram(abundances, bins=bins)
    fractions = counts / counts.sum()
    octave_centers = np.log2(np.sqrt(bin_edges[:-1] * bin_edges[1:]))
    
    # fit lognormal distribution
    log2_abundances = np.log2(abundances)
    mu_log2 = log2_abundances.mean()
    sigma_log2 = log2_abundances.std()
    x_log2 = np.linspace(log2_abundances.min() - 1, log2_abundances.max() + 1, 200)
    normal_pdf = norm.pdf(x_log2, mu_log2, sigma_log2)
    bin_width = np.mean(np.diff(octave_centers)) if len(octave_centers) > 1 else 1.0
    normal_pdf_scaled = normal_pdf * bin_width

    plt.bar(octave_centers, fractions, width=1,
            alpha=0.7, color='skyblue', edgecolor='black', 
            label='Simulated SAD')
    plt.plot(x_log2, normal_pdf_scaled, 'r-', linewidth=2,
             label=f'Lognormal fit')
    plt.xlabel('log₂(Abundance) [octaves]', fontsize=12)
    plt.ylabel('Fraction of species', fontsize=12)
    plt.title('Species Abundance Distribution', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("lognormal_distribution.png")

if __name__ == "__main__":
    main()

# %%
