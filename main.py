import numpy as np

class Species:
    def __init__(self, count: int, omega: float):
        self.count = count
        self.omega = omega

class Grid:
    def __init__(self, x: float, y: float, species: list[Species], m: int, l: float, delta0: float, delta_diff: float):
        self.x = x
        self.y = y
        self.species = species
        self.m = m
        self.l = l
        self.delta0 = delta0
        self.delta_diff = delta_diff
        self.points = self.place()

    def place(self):
        all_points = []
        for s in self.species:
            alpha = 1-s**(-s.omega/2)
            single_branch = 1-alpha
            both_branch = 2*alpha-1
            branch = np.random.choice(['l', 'r', 'b'], p=[single_branch, single_branch, both_branch])
            delta = self.delta0
            start = [np.random.rand() * self.x, np.random.rand() * self.y]
            new_points = self.new_points(branch, delta, start)
            l = self.l
            previous_delta = delta + np.pi/2
            for n in range(1, self.m):
                l = l/(2**n)
                delta = self.delta0*(self.delta_diff)**(2*int(n/2)/self.m)
                delta = (np.random.random() * 2 * delta) - delta
                p = new_points.copy()
                new_points = []
                for point in p:
                    branch = np.random.choice(['l', 'r', 'b'], p=[single_branch, single_branch, both_branch])
                    new_points.extend(self.new_points(branch, delta + previous_delta, point))
                previous_delta += delta + np.pi/2
            all_points.append(new_points)
        return all_points

    def new_points(self, branch, delta, point):
        left_branch = [point[0]*np.cos(delta), point[1]*np.sin(delta)]
        right_branch = [point[0]*np.cos(np.pi - delta), point[1]*np.sin(np.pi - delta)]
        if branch == 'l':
            return [left_branch]
        if branch == 'r':
            return [right_branch]
        return [left_branch, right_branch]