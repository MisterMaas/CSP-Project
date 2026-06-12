from dis import Positions

import numpy.random as random
import numpy as np
from FyeldGenerator import generate_field
from Cell import Cell
from Organism import Organism

class Model:
    # Population parameters
    FitnessPower : int
    MutationFactor : int
    NumberOfGenes : int

    # The enviroment parameters
    xSize : int
    ySize : int
    Grid : np.array
    CarryingCapacity : np.array

    # We keep track off all the cells that are
    # alive in the gird
    Cells : set
    Occupied = {}
    Organisms : set
    RegenaRate : float
    DivisionThreshold : int
    DivisionTimeSteps : int

    # Variables for the data analysis
    Timestep = 0
    TotalPopulation = -1
    MinimalDistance = 20
    MeanDistance = -1
    STDDistance = -1
    TypeOffCells : int

    # Add these tracking variables to your Model class top variables
    MeanOrgSize = 0.0
    MaxOrgSize = 0
    STDOrgSize = 0.0


    # The targets for the different
    # enviroments
    TargetA     : np.array
    TargetB     : np.array
    Target      : np.array
    TargetID= "A"

    # For now we propose that cells
    # can move in the Moore neighborhood
    Directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def __init__(self, initial_pop_density = 0.025, mean_resource = 5,
                 sd_recourse = 3, x_size = 100, y_size = 100,
                 fitness_power= 1, mutation_factor=15, number_of_genes = 20,
                 regen_rate = 0.05, division_thres = 15,
                 division_timesteps = 5):
        def FindTargets(expression_pattern: np.array):
            # Find the two targets
            def flip_expression(indices, original):
                flipped = original.copy()
                flipped[indices] ^= 1
                return flipped

            print(f"Initial Exp: {expression_pattern}")
            # We first iniitialize the target as zeros
            self.TargetA = np.zeros(self.NumberOfGenes)
            self.TargetB = np.zeros(self.NumberOfGenes)

            while not self.TargetA.any():
                # Generate target A
                a_number = random.randint(5, 21)
                a_indices = random.choice(self.NumberOfGenes, size=a_number, replace=False)
                self.TargetA = flip_expression(a_indices, expression_pattern)

            print(f"Target A: {self.TargetA}")
            # Generate target B, retrying until hamming distance from A is at least 7
            hamming_A_to_B = 0
            while hamming_A_to_B < 7 or not self.TargetB.any():
                b_number = random.randint(5, 21)
                b_indices = random.choice(self.NumberOfGenes, size=b_number, replace=False)
                self.TargetB= flip_expression(b_indices, expression_pattern)
                # Then we calculate the hamming distance
                hamming_A_to_B = np.sum(self.TargetA != self.TargetB)

            print(f"Target B: {self.TargetB}")
            # Set target to A by default
            self.Target = self.TargetA

        def InitializeGrid():
            # Field Generation copied form:
            # https://stackoverflow.com/questions/59712991/creating-a-2d-gaussian-random-field-from-a-given-2d-variance

            # Helper that generates power-law power spectrum
            def Pkgen(n):
                def Pk(k):
                    return np.power(k, -n)

                return Pk

            # Draw samples from a normal distribution
            def distrib(shape):
                # Always use standard normal — the field library expects this
                a = np.random.normal(loc=0, scale=1, size=shape)
                b = np.random.normal(loc=0, scale=1, size=shape)
                return a + 1j * b

            shape = (self.xSize, self.ySize)

            raw_field = generate_field(distrib, Pkgen(2), shape)
            field_real = np.real(raw_field)

            field_norm = (field_real - np.mean(field_real)) / (np.std(field_real) + 1e-8)
            field_scaled = field_norm * sd_recourse + mean_resource

            floored_field = np.floor(field_scaled)
            # We create the numpy array that represents the grid
            self.Grid = np.maximum(floored_field, 0)
            # Also initialize the carrying capasity, that keeps
            # of the distribution such that it can slowely regenerate.
            self.CarryingCapacity = self.Grid.copy()

            print(self.Grid)

        def InitializePopulation(parent: Cell):
            # We go over all indexces of the grid
            self.TypeOffCells = 0
            for i in range(self.xSize):
                for j in range(self.ySize):
                    # Cells get initialized and
                    # are given a initial position
                    if random.rand() < initial_pop_density:
                        cell = Cell.CopyCell(parent, i, j, self.TypeOffCells)
                        cell.CRL = 10
                        self.TypeOffCells += 1
                        self.Cells.add(cell)
                        org = Organism(self, cell, (i, j))
                        self.Organisms.add(org)
                        self.Occupied[(i, j)] = org


        # We distribute all the parameters to the class
        self.MutationFactor = mutation_factor
        self.NumberOfGenes = number_of_genes
        self.FitnessPower = fitness_power
        self.RegenaRate = regen_rate
        self.DivisionThreshold = division_thres
        self.DivisionTimeSteps = division_timesteps

        self.xSize = x_size
        self.ySize = y_size

        self.Cells = set()
        self.Organisms = set()
        self.TypeOffCells = 0

        # We create a random new cell
        cell = Cell(self,   number_of_genes)

        # Using this cell and it's exprassion pattern,
        # we determine the two target
        # exprssion patterns
        FindTargets(cell.ExpressionPattern)
        self.Target = self.TargetA

        # When we have foundt the target, we
        # execute propagation on this cell.
        cell.ExecutePropagation()

        # We initialize the grid with a
        # randomly distributed food source
        InitializeGrid()

        # We initialize the population (which lives
        # in the grid).
        InitializePopulation(cell)



    def ExecuteStep(self, priority="weight"):
        """
        Migration and division
        """""
        # We first make sure that we shuffle all the
        # organisms, otherwise there would be a priority.
        organisms_list = list(self.Organisms)
        # At first I implemented a random priority, but I think it is
        # more resonable to prioritize the heavier organisms.
        if priority == "random":
            random.shuffle(organisms_list)
        elif priority == "weight":
            # We first sort the
            organisms_list = sorted(self.Organisms, key=lambda org: org.TRL, reverse=True)
            # We first check if there are cells that have a

        # We make sure that the occupied array is empty
        for org in organisms_list:
            if org.CRL > self.DivisionThreshold:
                # All cells that are deviding have a counter
                # that checks how long they hae been in the
                # devision process. This counter is increased by one.
                org.Division_Steps += 1
                # If this counter exeeds some threshold, the cell divids.
                # The CRL will be shared among the two new cells, ensuring
                # that CRL > 0 and CRL < self.DivisionThreshold.
                if org.Division_Steps >= self.DivisionTimeSteps:
                    # all organisms then migrate one by one.
                    org.Divide()
            else:
                # We otherwise perform migration
                if org.CellAmount == 1:
                    org.MigrateSingleCell()
                else:
                    org.MigrateMultiCell()

        """
        Resource Consumption
        """""
        # We deterine the position of all the cells
        cells_list = list(self.Cells)
        rows = np.array([cell.iPos for cell in cells_list])
        cols = np.array([cell.jPos for cell in cells_list])

        # We determine what the cells consume,
        # with a maximum of 2 resources
        available = self.Grid[rows, cols]
        intakes = np.minimum(available, 2)

        # We remove the intake from the grid
        np.add.at(self.Grid, (rows, cols), -intakes)
        self.Grid = np.maximum(self.Grid, 0)

        # Then we add the intake to the cells
        for i, cell in enumerate(cells_list):
            cell.CRL += intakes[i]

        """"
        Cell death
        """
        # Update the organism CRL and determine which cells
        # die
        dead = []
        for org in self.Organisms:
            org.UpdateCRL()
            if org.CRL <= 0:
                dead.append(org)

        for org in dead:
            for pos in org.Positions:
                self.Occupied.pop(pos, None)

            for cell in org.Cells:
                if cell in self.Cells:
                    self.Cells.remove(cell)
            if org in self.Organisms:
                self.Organisms.remove(org)


        # We update the grid to slowely
        # regenerate to the original distribution
        deficit = self.CarryingCapacity - self.Grid
        self.Grid = np.minimum(
            self.Grid + self.RegenaRate * deficit,
            self.CarryingCapacity
        )

        # Info is updated for analysis
        self.UpdateInfo()


    def UpdateInfo(self):
        # To determine the minimum and mean hamming distance
        # We have set the following:
        if self.Cells:
            distances = np.array([cell.HammingDistance for cell in self.Cells])
            self.TotalPopulation = len(distances)
            self.MeanDistance = np.mean(distances)
            self.STDDistance = np.std(distances)
            self.MinimalDistance = np.min(distances)
        else:
            self.TotalPopulation = 0
            self.MeanDistance = 0
            self.STDDistance = 0
            self.MinimalDistance = self.NumberOfGenes

        if self.Organisms:
            # Gather sizes of all current organisms
            org_sizes = [org.CellAmount for org in self.Organisms]
            self.MeanOrgSize = np.mean(org_sizes)
            self.MaxOrgSize = np.max(org_sizes)
            self.STDOrgSize = np.std(org_sizes)
        else:
            self.MeanOrgSize = 0.0
            self.MaxOrgSize = 0
            self.STDOrgSize = 0.0

    def UpdateFitness(self):
        if not self.Cells:
            return
        cells_list = list(self.Cells)

        # Stack all expression patterns into a matrix
        patterns = np.array([cell.ExpressionPattern for cell in cells_list])

        # Compute all hamming distances at ones
        distances = np.sum(patterns != self.Target, axis=1)

        # Compute all fitnesses in one vectorised call
        max_dist = len(self.Target)
        fitnesses = (1 - (distances / max_dist)) ** self.FitnessPower

        # Write results back to each cell
        for i, cell in enumerate(cells_list):
            cell.HammingDistance = int(distances[i])
            cell.Fitness = float(fitnesses[i])

    def SwitchTarget(self):
        # We simply switch the target
        if np.array_equal(self.Target, self.TargetA):
            self.Target = self.TargetB
            self.TargetID = "B"
        else:
            self.Target = self.TargetA
            self.TargetID = "A"

        self.UpdateFitness()
        self.UpdateInfo()

    # def PriorityMigration(self, migrating_cells, stationary_cells):
    #     """"
    #     My current implementation of migration is somewhat complex
    #     and time consuming. Randomly choosing which cells has priority
    #     might seem like a way better option (discuss with Bram).
    #     """
    #     # All the migrating cells propose a new position
    #     proposals = [(cell, *cell.propose_move())
    #                  for cell in migrating_cells]
    #
    #     # Migrating cost one resource.
    #     for cell, _, _ in proposals:
    #         cell.CRL -= 1
    #
    #     # The stationary cells propose
    #     # their current position
    #     proposals += [(cell, cell.iPos, cell.jPos)
    #                   for cell in stationary_cells]
    #
    #     # We initialize a dictionary containing
    #     # the final positions, where the index of
    #     # the cell in self.Cells is the key and
    #     # the position is the value.
    #     final_pos = {idx: (ni, nj) for idx, (cell, ni, nj) in enumerate(proposals)}
    #
    #     # We then look if there are any coflicts,
    #     # meaning that more than one cell wants to
    #     # inhibit a grid position. We loop over
    #     # this process till there is no conflict present
    #     # any more
    #     conflict = True
    #     attempts = 0
    #     max_attempts = 100
    #
    #     while conflict and attempts < max_attempts:
    #         conflict = False
    #         attempts += 1
    #         # We identify a target map, which is
    #         # a dictionary where the keys are
    #         # positions and the values are
    #         # the cells that want to inhibit that
    #         # position
    #         map = {}
    #         for idx, pos in final_pos.items():
    #             map.setdefault(pos, []).append(idx)
    #
    #         # We loop over all the keys in the
    #         # map
    #         for pos, group in map.items():
    #             # We ignore all the map items
    #             # with only one candidate:
    #             # there is no conflict in this case
    #             if len(group) == 1:
    #                 continue
    #
    #             # If there is a group with more
    #             # then one item, this means that there
    #             # is a conflict
    #             conflict = True
    #
    #             # We sort the group by their CRL
    #             # value, where we assume that the
    #             # higher the CRL the more mass the
    #             # cell has
    #             group.sort(key=lambda i: proposals[i][0].CRL, reverse=True)
    #
    #             # We then check if one of the cells is
    #             # currently in the grid position. In other
    #             # words, we check if one of the stationary cells
    #             # is in the conflict group
    #             owner_idx = None
    #             for idx in group:
    #                 cell = proposals[idx][0]
    #                 if (cell.iPos, cell.jPos) == pos:
    #                     owner_idx = idx
    #                     break
    #
    #             # If this is the case, the owner stays in the
    #             # position and wins
    #             if owner_idx is not None:
    #                 winner = owner_idx
    #             else:
    #                 # Otherwise, the heaviest cell wins
    #                 winner = group[0]
    #
    #             # Everyone who isn't the winner gets bounced back home.
    #             # The loop is continued to see if the bounce back has not
    #             # caused a new conflict.
    #             for loser in group:
    #                 if loser != winner:
    #                     cell = proposals[loser][0]
    #                     final_pos[loser] = (cell.iPos, cell.jPos)
    #     """"
    #     Currently I have the max attempts maxed out, but I think the
    #     Migration will be changed to randomly prefer one or the other
    #     """
    #     if attempts == max_attempts:
    #         print("Warning: Migration conflict resolution maxed out!")
    #
    #     # we assign the cells to their final position
    #     for idx, (cell, _, _) in enumerate(proposals):
    #         cell.iPos, cell.jPos = final_pos[idx]

if __name__ == '__main__':
    model = Model()