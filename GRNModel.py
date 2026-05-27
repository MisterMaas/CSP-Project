from dataclasses import dataclass
import numpy.random as random
import numpy as np
from Cell import Cell

class Model:
    # Population parameters
    DeathRate = 0.1
    xSize = 100
    ySize = 50

    FitnessPower = 1
    MutationFactor = 1

    Grid : np.array
    NextGrid : np.array

    Timestep = 0
    TotalPopulation = -1
    MinimalDistance = 20
    MeanDistance = -1
    STDDistance = -1


    # The targets for the different
    # enviroments
    TargetA     : np.array
    TargetB     : np.array
    Target      : np.array
    TargetID= "A"

    def __init__(self, death_rate=0.1, fitness_power= 1, mutation_factor=15):
        def FindTargets(expression_pattern: np.array):
            # Find the two targets
            def flip_expression(indices, original):
                flipped = original.copy()
                flipped[indices] ^= 1
                return flipped

            print(f"Initial Exp: {expression_pattern}")
            # We first iniitialize the target as zeros
            self.TargetA = np.zeros(20)
            self.TargetB = np.zeros(20)

            while not self.TargetA.any():
                # Generate target A
                a_number = random.randint(5, 21)
                a_indices = random.choice(20, size=a_number, replace=False)
                self.TargetA = flip_expression(a_indices, expression_pattern)

            print(f"Target A: {self.TargetA}")
            # Generate target B, retrying until hamming distance from A is at least 7
            hamming_A_to_B = 0
            while hamming_A_to_B < 7 and not self.TargetB.any():
                b_number = random.randint(5, 21)
                b_indices = random.choice(20, size=b_number, replace=False)
                self.TargetB= flip_expression(b_indices, expression_pattern)
                # Then we calculate the hamming distance
                hamming_A_to_B = np.sum(self.TargetA != self.TargetB)

            print(f"Target B: {self.TargetB}")
            # Set target to A by default
            self.Target = self.TargetA

        def InitializePopulation(parent: Cell):
            # We create the numpy array that represents the grid
            self.Grid = np.full((self.xSize, self.ySize), None, dtype=object)
            # We go over all indexces of the grid
            total = 0
            populated = 0
            for i in range(self.xSize):
                for j in range(self.ySize):
                    # Only on a proportion of the grid
                    # does the cell get copied.
                    total += 1
                    if random.rand() < self.DeathRate:
                        # The rest of the grid becomes nan
                        self.Grid[i,j] = None
                    else:
                        self.Grid[i, j] = Cell.CopyCell(parent)
                        populated +=1

        self.MutationFactor = mutation_factor

        # We create a random new cell
        cell = Cell(self, 20)
        # Using this cell and it's exprassion pattern,
        # we determine the two target
        # exprssion patterns
        FindTargets(cell.ExpressionPattern)

        # When we have foundt the target, we
        # execute propagation on this cell.
        cell.ExecutePropagation()

        # We initialize the population (which lives
        # in the grid).
        InitializePopulation(cell)

        # We set the death rate and fitnesspower
        # to the same values as Crombach and Hogeweg
        # (This is done outside the model class for
        # easier experimentation).
        self.DeathRate = death_rate
        self.FitnessPower = fitness_power

        # We initialize the next grid as an empty grid
        # of the same size as the grid. The initial
        # target is by default target A.
        self.NextGrid = np.empty((self.xSize, self.ySize), dtype=object)
        self.Target = self.TargetA

    def ExecuteStep(self):
        def Reproduce(i: int,j : int):
            # We want to look for the neighbor with the best fitness
            best_fitness = -1
            best_parent = None

            # We then loop over all 8 neighbors to check their fitness.
            # Note, we also check the cell itself, but since it is
            # by definition empty, it will be ignored
            for k in range(-1, 2):
                for l in range(-1, 2):
                    # Make sure that we "warp" around the grid
                    di, dj = (i + k + self.xSize) % self.xSize, (j + l + self.ySize) % self.ySize
                    # Ignore all the empty cells
                    if self.Grid[di, dj] is None: continue
                    # We compare the fitness
                    # Since we're going to implement a different form of fitness, I simplified
                    # the reproduction function to simply look at which neighbor is fittest
                    if self.Grid[di, dj].Fitness > best_fitness:
                        best_fitness = self.Grid[di, dj].Fitness
                        best_parent = self.Grid[di, dj]

            # If there is no parent in the neighborhood,
            # there is no reproduction
            if best_parent is None:
                return
            # We copy the best parent to the next grid. The new child
            # mutates and we execute propagation.
            self.NextGrid[i, j] = Cell.CopyCell(best_parent)
            # Mutate returns false if the cell has died
            if not self.NextGrid[i, j].Mutate(mutation_factor=self.MutationFactor):
                self.NextGrid[i, j] = None
                return

            # Execute propagation returns false whe the cells has died
            if not self.NextGrid[i, j].ExecutePropagation():
                # In that case it stays dead
                self.NextGrid[i, j] = None
                return

        def CellStep(i :int, j :int):
            # Here we execute the updating of a cell.
            # We kill some cells and ignore them
            if self.Grid[i, j] is not None and random.rand() < self.DeathRate:
                self.NextGrid[i, j] = None
                return

            # We check wether the gird position is empty
            # or has a cell in it
            if self.Grid[i, j] is None:
                # When it is empty, we reproduce a new
                # cell in that grid which is a child form
                # the fittest neighbor
                Reproduce(i,j)
            else:
                # If the grid position contains a cell,
                # we simply copy the existing cell in the
                # next gird
                self.NextGrid[i, j] = self.Grid[i, j]

        def GridStep():
            # To determine the minimum and mean hamming distance
            # We have set the following:
            distances = []
            minimum = 21

            # Then we loop over the entirety of the grid.
            for i in range(self.xSize):
                for j in range(self.ySize):
                    # For each element in the grid we
                    # execute a Cell Step. This will
                    # put the cell in Next Grid
                    CellStep(i, j)
                    # If the cell step resulted in the
                    # cell being alive in the next step,
                    # we determin some properties about the grid
                    # for the analysis.
                    if self.NextGrid[i, j] is not None:
                        distances.append(self.NextGrid[i, j].HammingDistance)
                        if self.NextGrid[i, j].HammingDistance < minimum:
                            minimum = self.NextGrid[i, j].HammingDistance

            self.TotalPopulation = len(distances)
            if self.TotalPopulation > 0:
                self.MeanDistance = np.mean(distances)
                self.STDDistance = np.std(distances)
                self.MinimalDistance = minimum
            else:
                self.MeanDistance = 0
                self.STDDDistance = 0
                self.MinimalDistance = 20

        self.Timestep += 1

        # We make sure that the Next Grid is empty
        # This is the place where all the updated cells
        # will be moved to
        self.NextGrid = np.empty((self.xSize, self.ySize), dtype=object)

        # We loop over all the cells in the grid and for each cell
        # we perfrom a cell step.
        GridStep()

        # And transform everything back to the old grid
        self.Grid = self.NextGrid

    def SwitchTarget(self):
        # We simply switch the target
        if np.array_equal(self.Target, self.TargetA):
            self.Target = self.TargetB
            self.TargetID = "B"
        else:
            self.Target = self.TargetA
            self.TargetID = "A"

        # We then go over the grid to update all living cells
        # To determine the minimum and mean hamming distance
        # We have set the following:
        minimum = 21
        distances = []

        # Then we loop over the entirety of the grid.
        for i in range(self.xSize):
            for j in range(self.ySize):
                # If the cell step resulted in the
                # cell being alive in the next step,
                # we determin some properties about the grid
                # for the analysis.
                if self.Grid[i, j] is not None:
                    self.Grid[i, j].UpdateFitness()
                    distances.append(self.Grid[i, j].HammingDistance)
                    if self.Grid[i, j].HammingDistance < minimum:
                        minimum = self.Grid[i, j].HammingDistance

        if len(distances) > 0:
            self.MeanDistance = np.mean(distances)
            self.STDDistance = np.std(distances)
            self.MinimalDistance = minimum
        else:
            self.MeanDistance = 0
            self.STDDistance = 0
            self.MinimalDistance = 20


