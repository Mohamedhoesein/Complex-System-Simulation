from enum import Enum

import numpy as np
from matplotlib import pyplot as plt

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
            d: float
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
        self.points = []
        for alpha in self.species_alpha:
            self.points.append(self.generate_species(alpha))

    def generate_species(self, alpha: float) -> list[Individual]:
        """
        Generate a species with a given alpha.
        
        :param alpha: The alpha parameter for the current species.
        :type alpha: float
        :return: The set of individuals for the current species.
        :rtype: list[Individual]
        """
        p = [1-alpha, 1-alpha, 2*alpha-1]

        # initial point
        theta0 = np.random.uniform(0, 2*np.pi)

        r0 = self.get_initial_point()
        points = [Individual(r0.x, r0.y, theta0)]

        l = self.l

        for n in range(1, self.m+1):
            l /= 2
            new_points = []

            delta_max = self.delta0 * (self.delta_diff)**(2*(n//2)/self.m)

            for p0 in points:
                delta = np.random.uniform(-delta_max, delta_max)
                theta = p0.theta + np.pi/2 + delta

                branch = np.random.choice([Branch.LEFT, Branch.RIGHT, Branch.BOTH], p=p)

                if branch in (Branch.LEFT, Branch.BOTH):
                    new_points.append(
                        Individual(p0.x + l*np.cos(theta),
                                p0.y + l*np.sin(theta),
                                theta)
                    )
                if branch in (Branch.RIGHT, Branch.BOTH):
                    new_points.append(
                        Individual(p0.x + l*np.cos(theta + np.pi),
                                p0.y + l*np.sin(theta + np.pi),
                                theta + np.pi)
                    )

            points = new_points

        return points

    def get_initial_point(self) -> Individual:
        """
        Create the initial point for a tree.
        
        :return: The initial point.
        :rtype: Individual
        """
        middle = Individual(0, 0, 0)
        # x+1 and y+1 are needed because we take [0,x] as a valid range which has x+1 integers,
        # and the same applies to y.
        x_offset = np.random.rand()*self.d
        y_offset = np.random.rand()*self.d
        x_radian = np.random.rand()*2*np.pi
        y_radian = np.random.rand()*2*np.pi
        return Individual(
            middle.x + x_offset*np.cos(x_radian),
            middle.y + y_offset*np.sin(y_radian),
            0
        )

    # def f(self, species: list[Individual], r: float) -> float:
    #     """
    #     Implement the proximity function Fs(r)
        
    #     :param species: The species for which to implement the proximity function.
    #     :type species: list[Individual]
    #     :param R: The radius for which to check.
    #     :type R: float
    #     :return: The proximity values.
    #     :rtype: float
    #     """
    #     a = np.pi*r**2
    #     inverse_a = 1/a
    #     cummulator = 0
    #     for point in species:
    #         distances = map(
    #             lambda other:
    #                 np.sqrt(
    #                     (other.x-point.x)**2 + (other.y-point.y)**2
    #                 ),
    #             species
    #         )
    #         size = len(list(filter(
    #             lambda x: abs(x) <= r and x != 0,
    #             distances
    #         )))
    #         cummulator += size
    #     return cummulator * inverse_a

    def f(self, species: list[Individual], r: float) -> float:
        if len(species) < 2:
            return 0.0
        coords = np.array([[p.x, p.y] for p in species])
        dx = coords[:, None, 0] - coords[None, :, 0]
        dy = coords[:, None, 1] - coords[None, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        # exclude self-distances
        dists = dists[dists > 0]
        cummulator = np.sum(dists <= r)
        area = np.pi * r**2
        return cummulator / area


    # def s(self, r: float) -> float:
    #     """
    #     Imeplementation of the SAR satistic.
        
    #     :param r: The radius to use for the statistic.
    #     :type r: float
    #     :return: The SAR statistic of all species.
    #     :rtype: float
    #     """
    #     cummulator = 0
    #     for species in self.points:
    #         cummulator += self.f(species, r)
    #     return cummulator

    # def restrict(self, species: list[Individual], L: float, origin) -> list[Individual]:
    #     x_0, y_0 = origin
    #     half = L / 2
    #     restricted_species = []

    #     for individual in species:
    #         if (x_0 - half <= individual.x <= x_0 + half) and (y_0 - half <= individual.y <= y_0 + half):
    #             restricted_species.append(individual)

    #     return restricted_species

    def restriction_box(self, species: list[Individual], L: float, origin) -> list[Individual]:
        x_0, y_0 = origin
        half = L / 2
        coords = np.array([[p.x, p.y] for p in species])
        
        mask =  (coords[:,0] >= x_0-half) & (coords[:,0] <= x_0+half) & \
                (coords[:,1] >= y_0-half) & (coords[:,1] <= y_0+half)
        return [species[i] for i in range(len(species)) if mask[i]]

    def s(self, r: float, L: float, origin=(0.0, 0.0)) -> float:
        """
        Imeplementation of the SAR satistic.
        
        :param r: The radius to use for the statistic.
        :type r: float
        :return: The SAR statistic of all species.
        :rtype: float
        """
        cummulator = 0
        for species in self.points:
            restricted_area = self.restriction_box(species, L, origin)
            if len(restricted_area) > 1:
                cummulator += self.f(restricted_area, r)
            elif len(restricted_area) == 1:
                cummulator += 1
        return cummulator    

    def pair_correlation(self, species: list[float], r_bins: np.ndarray[np.floating], area: float):
        """
        Compute radial pair correlation function g(r) for one species.

        :param species: The species for which to calculate the pair correlation function.
        :type species: list[Individual]
        :param r_bins: Array of bin edges.
        :type r_bins: np.ndarray[np.floating]
        :param area: area of observation window
        :type area: float
        :return: g(r) values for each bin
        """
        points = np.array([(p.x, p.y) for p in species])
        n = len(points)
        if n < 2:
            return np.zeros(len(r_bins) - 1)

        # Pairwise distance matrix
        dx = points[:, None, 0] - points[None, :, 0]
        dy = points[:, None, 1] - points[None, :, 1]
        d = np.sqrt(dx**2 + dy**2)

        # Keep only upper triangle (exclude self-pairs and double counting)
        d = d[np.triu_indices(n, k=1)]

        # Histogram distances
        counts, edges = np.histogram(d, bins=r_bins)

        # Shell areas
        r_inner = edges[:-1]
        r_outer = edges[1:]
        shell_areas = np.pi * (r_outer**2 - r_inner**2)

        # Density
        rho = n / area

        # g(r), this scales how many individuals are seen at a radius x,
        # by how many you expect on average (rho),
        # how many there are in total (n),
        # and size of the shell.
        g = counts / (rho * shell_areas * n)

        return g

    def all_pair_correlation(self, r_bins: np.ndarray[np.floating], area: float):
        """
        Calculate the pair correlation for all species.
        
        :param r_bins: Array of bin edges.
        :type r_bins: np.ndarray[np.floating]
        :param area: Area of observation window.
        :type area: float
        """
        gs = [self.pair_correlation(species, r_bins, area)
            for species in self.points if len(species) > 1]
        return np.mean(gs, axis=0)

def main():
    """
    Entry point for the calculations.
    """
    np.random.seed(2)
    t = [-0.01, -0.03, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.40, -0.50, -0.60, -0.65]
    alpha = list(map(lambda o: 2**o, t))
    #TODO: change the range so that the proper amount of species for each alpha is used.
    species_alpha = [o for o in alpha for i in range(1)]
    grid = Field(
        species_alpha=species_alpha,
        m=14,
        l=80,
        delta0=0.1,
        delta_diff=8,
        d=5
    )

    for processed_points in grid.points:
        x_scatter, y_scatter = zip(*map(lambda point: (point.x, point.y), processed_points))
        plt.scatter(x_scatter, y_scatter)
    plt.savefig("./test.png")
    plt.clf()

    r_min = 0.5
    r_max = 50.0
    num_bins = 25
    l = 80

    r_bins = np.logspace(np.log10(r_min), np.log10(r_max), num_bins + 1)
    r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    # #TODO: Do the correlation within an alpha, and not across.
    gs = grid.all_pair_correlation(r_bins, l * l)
    plt.clf()
    plt.loglog(r_centers, gs, 'o-')
    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.savefig("rho.png")

if __name__ == "__main__":
    main()
