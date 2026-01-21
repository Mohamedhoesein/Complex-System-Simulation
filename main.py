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
            x: float,
            y: float,
            species_omega: list[float],
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
        :param species_omega: The omega for each species.
        :type species_omega: list[float]
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
        assert x > 0
        assert y > 0
        for omega in species_omega:
            assert omega >= 2
        assert m > 0
        assert l > 0
        assert d > 0
        self.x = x
        self.y = y
        self.species_omega = species_omega
        self.m = m
        self.l = l
        self.delta0 = delta0
        self.delta_diff = delta_diff
        self.d = d
        self.points = self.place()

    def place(self) -> list[list[Individual]]:
        """
        Place individuals of a species in an area.
        
        :return: The coordinates for each spiecies.
        :rtype: list[list[Individual]]
        """
        all_points = []
        for s in self.species_omega:
            # For the current species s calculate the probability to have a certain amount of branches.
            # Given that for a single branch 1-alpha=2**(-omega_s/2) we get alpha=1-2**(-omega_s/2).
            # From this we can get the probabilities for each branch
            alpha = 1-2**(-s/2)
            single_branch = 1-alpha
            both_branch = 2*alpha-1
            # Select the way to branch.
            branch = np.random.choice(
                [Branch.LEFT, Branch.RIGHT, Branch.BOTH],
                p=[single_branch, single_branch, both_branch]
            )
            # The initial offset is just delta0 which is a parameter.
            delta = self.delta0
            # We randomly generate a starting point.
            start = self.get_initial_point()

            l = self.l
            new_points = self.new_points(branch, delta, start, l)
            # Keep track of the delta so later this can be used as the extra needed
            # offset for later branches.
            for n in range(1, self.m):
                # Reduce the branch length as specified in the algorithms
                l = l/(2**n)
                points = new_points.copy()
                new_points = []
                for point in points:
                    # Calculate the new range from which to draw delta delta0 is \delta_0 in latex,
                    # delta_diff is \Delta\delta in latex
                    delta = self.delta0*(self.delta_diff)**(2*int(n/2)/self.m)
                    theta = (np.random.random() * 2 * delta) - delta
                    # Determine how many branches
                    branch = np.random.choice(
                        [Branch.LEFT, Branch.RIGHT, Branch.BOTH],
                        p=[single_branch, single_branch, both_branch]
                    )
                    # We use previous_delta + np.pi/2 because the new delta is a delta
                    # from the line which has a right angle with the previous line.
                    # The previous line has a delta of previous_delta, and np.pi/2 gives
                    # us the needed radians for the 90 degrees.
                    new_points.extend(
                        self.new_points(
                            branch,
                            point.theta + theta + np.pi/2,
                            point,
                            l
                        )
                    )
                all_points.append(new_points)
        return all_points

    def get_initial_point(self) -> Individual:
        """
        Create the initial point for a tree.
        
        :return: The initial point.
        :rtype: Individual
        """
        middle = Individual((self.x+1)/2, (self.y+1)/2, 0)
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

    def new_points(
            self,
            branch: Branch,
            theta: float,
            point: Individual,
            distance: float
        ) -> list[Individual]:
        """
        Calculate the new branching points.
        
        :param branch: What type of branch to use, left, right or both.
        :type branch: Branch
        :param theta: The degree at which to place the line on which we have the new points.
        :type theta: float
        :param point: The origin of the line.
        :type point: Individual
        :param distance: The distance of the origin from which to place the new individuals.
        :type distance: float
        :return: The new individuals.
        :rtype: list[Individual]
        """
        # To calculate the new points we use the calculation from
        # https://stackoverflow.com/questions/2912779/how-to-calculate-a-point-with-an-given-center-angle-and-radius
        left_branch = Individual(
            point.x + distance*np.cos(theta),
            point.y + distance*np.sin(theta),
            theta
        )
        right_branch = Individual(
            point.x + distance*np.cos(np.pi + theta),
            point.y + distance*np.sin(np.pi + theta),
            theta
        )
        if branch == Branch.LEFT:
            return [left_branch]
        if branch == Branch.RIGHT:
            return [right_branch]
        return [left_branch, right_branch]

    def f(self, species: list[Individual], r: float) -> float:
        """
        Implement the proximity function Fs(r)
        
        :param species: The species for which to implement the proximity function.
        :type species: list[Individual]
        :param R: The radius for which to check.
        :type R: float
        :return: The proximity values.
        :rtype: float
        """
        a = np.pi*r**2
        inverse_a = 1/a
        for point in species:
            distances = map(lambda other: np.sqrt((other.x-point.x)**2+(other.y-point.y)**2), species)
            size = len(list(filter(
                lambda x: abs(x) <= r and x != 0,
                distances
            )))
            cummulator += size
        return cummulator * inverse_a

    def s(self, r: float) -> float:
        """
        Imeplementation of the SAR satistic.
        
        :param r: The radius to use for the statistic.
        :type r: float
        :return: The SAR statistic of all species.
        :rtype: float
        """
        cummulator = 0
        for species in self.points:
            cummulator += self.f(species, r)
        return cummulator

if __name__ == "__main__":
    omega = list(range(2, 14))
    species_omega = [o for o in omega for i in range(100)]
    grid = Field(
        x=200,
        y=200,
        species_omega=species_omega,
        m=14,
        l=80,
        delta0=0.1,
        delta_diff=8,
        d=5
    )

    PLOT_COUNT = len(grid.points)
    colors = range(0, PLOT_COUNT)
    for processed_points, color in zip(grid.points, colors):
        x_scatter, y_scatter = zip(*map(lambda point: (point.x, point.y), processed_points))
        plt.scatter(x_scatter, y_scatter)
    plt.savefig("./test.png")
