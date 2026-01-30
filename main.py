from enum import Enum
import json
import numpy as np
from matplotlib import patches, pyplot as plt
from scipy.stats import norm, kstest
from scipy import stats
from sklearn.neighbors import KDTree
from plots import rcCustom, rcCustom_wide



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

        #ls = np.linspace(self.l, 1, self.m)
        l0 = self.l
        for n in range(1, self.m+1):
            #l = ls[n-1]
            l = l0*(1.5**(-(n-1)))
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
        max_R = R_values.max()
        sample_x = np.random.uniform(-self.omega_range + max_R,
                                    self.omega_range - max_R, n_samples)
        sample_y = np.random.uniform(-self.omega_range + max_R,
                                    self.omega_range - max_R, n_samples)

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
#plotting code   
def plot_correlation_functions(results_corr_dict: dict, R_values: np.ndarray):
    """
    Plot spatial correlation functions with power-law fits.
    """
    sorted_alphas = sorted(results_corr_dict.keys())
    with plt.rc_context(rc=rcCustom):
        plt.figure()  
        for alpha in sorted_alphas:
            rho_avg = results_corr_dict[alpha]
            if rho_avg is not None and np.sum(rho_avg) > 0:
                r_centers = (R_values[:-1] + R_values[1:]) / 2
                valid_indices = rho_avg > 0
                if np.sum(valid_indices) > 2:  
                    log_r = np.log10(r_centers[valid_indices])
                    log_rho = np.log10(rho_avg[valid_indices])
                    slope, intercept, r_value, p_value, std_err = stats.linregress(log_r, log_rho)
                    r_squared = r_value**2
                    plt.plot(r_centers, rho_avg, marker='o', linewidth=1.3,
                            label=f'$\\alpha$={alpha:.2f}, z={slope:.2f}, $R^2$={r_squared:.2f}')
                else:
                    plt.plot(r_centers, rho_avg, marker='o', linewidth=1.3,
                            label=f'$\\alpha$={alpha:.3f}')

        plt.xscale('log')
        plt.yscale('log') 
        plt.xlabel(r'Distance $r$ [a.u.]')
        plt.ylabel(r'Correlation $\rho_s(r) / \rho_s(1)$')
        plt.title('Spatial Correlation Functions')
        plt.legend()
        plt.tight_layout()
        plt.savefig("spatial_correlation.png")
        plt.show()
    
class Extinction:
    """Determine extinction probabilities after habitat loss.
    """    
    def __init__(self, a:float, b:float):
        """initialize parameters.

        Args:
            a (float): lower bound of the interval in which to search for the root
            b (float): upper bound of the interval in which to search for the root
        """        
        self.a = a
        self.b = b

    def q_numeric(self, area_loss, n_0):
        """function to determine q by finding the root numerically, used when bisection method fails

        Args:
            area_loss (float): fractional area loss, given by dividing the area after loss by the initial area
            n_0 (int): initial number of individuals for a given species

        Returns:
            float: root of the function f(q)
        """ 
        q_try = np.linspace(self.a, self.b, 1000000)
        lhs = area_loss * n_0
        rhs = (q_try / (1 - q_try)) - ((n_0 + 1) * q_try ** (n_0 + 1)) / (1 - q_try ** (n_0 + 1))
        root_find = lhs - rhs

        y_closest = np.min(np.abs(root_find))
        q_closest = q_try[np.argmin(np.abs(root_find))]

        return q_closest

    def function(self, q, area_loss, n_0):
        """function that gives f(q) for a given value of q. 

        Args:
            q (float): constant between 0 and 1, used to determine the extinction probability
            area_loss (float): fractional area loss, given by dividing the area after loss by the initial area
            n_0 (int): initial number of individuals for a given species

        Returns:
            float: value of the function f(q), evaluated at a given q
        """    
        lhs = area_loss * n_0
        rhs = (q / (1 - q)) - ((n_0 + 1) * q ** (n_0 + 1)) / (1 - q ** (n_0 + 1))
        return lhs - rhs

    def q_bisection(self, epsilon, area_loss, n_0):
        """Root finding using the bisection method.

        Args:
            epsilon (float): tolerance for the root-finding algorithm
            area_loss (float): fractional area loss, given by dividing the area after loss by the initial area
            n_0 (int): initial number of individuals for a given species

        Returns:
            float: root of the function f(q) within the interval [a, b]
        """    
        a = self.a
        b = self.b

        f_a = self.function(a, area_loss, n_0)
        f_b = self.function(b, area_loss, n_0)

        # Check condition for bisection method
        if f_a * f_b > 0:
            print(f"Bisection method fails for initial species count {n_0}, using other method.")
            q = self.q_numeric(area_loss, n_0)
            return q
        
        # Middle point
        c = (a + b) / 2.0
        f_c = self.function(c, area_loss, n_0)

        while abs(f_c) > epsilon:
            c = (a + b) / 2.0
            f_c = self.function(c, area_loss, n_0)
            f_a = self.function(a, area_loss, n_0)

            if f_c * f_a < 0:
                b = c
            else:
                a = c
                
        return c

    def extinction_probability(self, q, n_c, n_0):
        """Determine extinction probability.

        Args:
            q (float): probability parameter
            n_c (int): critical abundance, the number of individuals below which a species is considered ecologically extinct
            n_0 (int): initial number of individuals

        Returns:
            float: extinction probability
        """    
        return (q ** (n_c + 1) - 1) / (q ** (n_0 + 1) - 1)
    

def plot_species_distribution(grid: Field, num_species: int = 5):
    """
    Plot spatial distribution of species with omega region.
    """
    colors = plt.cm.jet(np.linspace(0, 1, num_species))

    unique_alphas = sorted(list(set([key.split("-")[0] for key in grid.points.keys()])))
    alpha_index = 4  # choose alpha index to plot
    alpha = float(unique_alphas[alpha_index])
    
    with plt.rc_context(rc=rcCustom):
        plt.figure()
        
        for i in range(num_species):
            species_key = f"{alpha}-{i}"
            if species_key in grid.points:
                species = grid.points[species_key]
                xs = [p.x for p in species]
                ys = [p.y for p in species]
                plt.scatter(xs, ys, s=2, alpha=0.6, label=f'Species {i}', color=colors[i])

        # omega box
        omega_range = grid.omega_range
        rect = patches.Rectangle(
            (-omega_range, -omega_range),  
            2 * omega_range,
            2 * omega_range,
            linewidth=1.5,
            edgecolor='red',
            facecolor='none',
            linestyle='--',
            label=f'$\\Omega$ region ($L_{{av}}={grid.L_av}$)'
        )
        ax = plt.gca()
        ax.add_patch(rect)
        
        plt.title(f"Species Distribution ({num_species} Species, $\\alpha$={alpha:.2f})")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.savefig("species_distribution.png")
        plt.show()

def plot_sar(grid: Field, R_values: np.ndarray, n_samples: int = 2000):
    """
    Plot Species-Area Relationship with bootstrapped confidence intervals.
    """
    S_values = grid.species_area_curve(R_values, n_samples=n_samples)
    A_values = np.pi * (R_values ** 2)
    
    log_A = np.log10(A_values)
    log_S = np.log10(S_values)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_A, log_S)
    r_squared = r_value**2
    
    with plt.rc_context(rc=rcCustom):
        plt.figure()
        
        plt.loglog(A_values, S_values, 'o', 
                   color='black', markersize=5, label='Simulated data')
        # fit line
        fit_line = 10**intercept * A_values**slope
        plt.loglog(A_values, fit_line, 'r--', linewidth=2,
                   label=f'Power-law: $z={slope:.3f}$, $R^2={r_squared:.2f}$')
        plt.xlabel(r"Area $A$ [a.u.]")
        plt.ylabel(r"Number of Species $S_C(A)$")
        plt.title("Species-Area Relationship (SAR)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("sar.png")

        plt.show()

def plot_lognormal_distribution(grid: Field):
    """
    Plot species abundance distribution with lognormal fit.
    """
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
    
    # goodness of fit (K-S test)
    statistic, p_value = kstest(log2_abundances, 'norm', 
                                args=(mu_log2, sigma_log2))
    
    with plt.rc_context(rc=rcCustom):
        fig, ax = plt.subplots()  
        ax.bar(octave_centers, fractions, width=1,
                alpha=0.7, color="#5BD75B", 
                label='Simulated SAD')
        ax.plot(x_log2, normal_pdf_scaled, 'r-', linewidth=2,
                 label=f'Lognormal fit ($\\mu$={mu_log2:.2f}, $\\sigma$={sigma_log2:.2f})')
        
        ax.set_xlabel('Abundance (octaves scaled)')
        ax.set_ylabel('Fraction of species')
        fig.suptitle("Species Abundance Distribution")
        ax.set_title(f"(K-S test: p={p_value:.4f})", pad = 20)
        ax.legend()
        fig.tight_layout()
        fig.savefig("lognormal_distribution.png")
        plt.show()


def main():
    """Main simulation and analysis pipeline."""
    np.random.seed(42)
    #TODO: justify parameters
    t = [-0.05, -0.15, -0.25, -0.35, -0.45, -0.55, -0.65]
    alpha_values = list(map(lambda o: 2**o, t))
    species_alpha = [o for o in alpha_values for i in range(50)]
    grid = Field(
        species_alpha=species_alpha,   
        m=14,
        l=20,
        delta0=0.1,
        delta_diff=8,
        d=15,
        L_av=20
    )

    # Save species data
    with open("test.json", "w+") as f:
        d = dict()
        for key in grid.points.keys():
            d[key] = list(map(lambda x: [x.x, x.y, x.theta], grid.points[key]))
        json.dump(d, f)

    # radius range
    r_min = 0.5
    r_max = 15
    num_bins = 30 
    R_values = np.logspace(np.log10(r_min), np.log10(r_max), num_bins + 1)
    
    results_corr_dict = grid.get_correlations_grouped_by_alpha(R_values)
    plot_correlation_functions(results_corr_dict, R_values)
    
    plot_species_distribution(grid, num_species=5)
    
    plot_sar(grid, R_values, n_samples=2000)
    
    plot_lognormal_distribution(grid)
    

if __name__ == "__main__":
    main()

