import numpy as np
from enum import Enum

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
    def __init__(self, x: float, y: float):
        """
        Initialise a position for an individual.
        
        :param x: X coordinate of the individual
        :type x: float
        :param y: Y coordinate of the individual
        :type y: float
        """
        self.x = x
        self.y = y

class Area:
    def __init__(self, x: float, y: float, species_omega: list[float], m: int, l: float, delta0: float, delta_diff: float):
        """
        Initialise an area.
        
        :param x: The maximum x coordinate.
        :type x: float
        :param y: The maximum y coordinate.
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
        """
        self.x = x
        self.y = y
        self.species_omega = species_omega
        self.m = m
        self.l = l
        self.delta0 = delta0
        self.delta_diff = delta_diff
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
            branch = np.random.choice([Branch.LEFT, Branch.RIGHT, Branch.BOTH], p=[single_branch, single_branch, both_branch])
            # The initial offset is just delta0 which is a parameter.
            delta = self.delta0
            # We randomly generate a starting point.
            start = [np.random.rand() * self.x, np.random.rand() * self.y]

            l = self.l
            new_points = self.new_points(branch, delta, start, l)
            all_points.append(new_points)
            # Keep track of the delta so later this can be used as the extra needed offset for later branches.
            previous_theta = delta
            for n in range(1, self.m):
                # Reduce the branch length as specified in the algorithms
                l = l/(2**n)
                # Calculate the new range from which to draw delta delta0 is \delta_0 in latex, delta_diff is \Delta\delta in latex
                delta = self.delta0*(self.delta_diff)**(2*int(n/2)/self.m)
                delta = (np.random.random() * 2 * delta) - delta
                p = new_points.copy()
                new_points = []
                for point in p:
                    # Determine how many branches
                    branch = np.random.choice([Branch.LEFT, Branch.RIGHT, Branch.BOTH], p=[single_branch, single_branch, both_branch])
                    # We use previous_delta + np.pi/2 because the new delta is a delta from the line which has a right angle with the previous line.
                    # The previous line has a delta of previous_delta, and np.pi/2 gives us the needed radians for the 90 degrees.
                    new_points.extend(self.new_points(branch, previous_theta + delta + np.pi/2, point, l))
                # Add the extra degree
                previous_theta += delta + np.pi/2
            all_points.append(new_points)
        return all_points

    def new_points(self, branch: Branch, theta: float, point: Individual, distance: float) -> list[Individual]:
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
        # To calculate the new points we use the calculation from https://stackoverflow.com/questions/2912779/how-to-calculate-a-point-with-an-given-center-angle-and-radius
        left_branch = [point.x+distance*np.cos(theta), point.y+distance*np.sin(theta)]
        right_branch = [point.x+distance*np.cos(np.pi - theta), point.y+distance*np.sin(np.pi - theta)]
        if branch == Branch.LEFT:
            return [left_branch]
        if branch == Branch.RIGHT:
            return [right_branch]
        return [left_branch, right_branch]

if __name__ == "__main__":
    grid = Area(100, 100, [0.1], 10, 14, 0.1, 8)