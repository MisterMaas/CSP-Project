import numpy.random as random
import numpy as np
from FyeldGenerator import generate_field

from Cell import Cell

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
    Cells = []
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


    # The targets for the different
    # enviroments
    TargetA     : np.array
    TargetB     : np.array
    Target      : np.array
    TargetID= "A"

    # For now we propose that cells
    # can only move in the von neuman neighborhood
    Directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def __init__(self, initial_pop_density = 0.01, mean_resource = 5,
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
            while hamming_A_to_B < 7 and not self.TargetB.any():
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
            # We create the numpy array that represents the cells
            self.Cells = []
            # We go over all indexces of the grid
            self.TypeOffCells = 0
            for i in range(self.xSize):
                for j in range(self.ySize):
                    # Cells get initialized and
                    # are given a initial position
                    if random.rand() < initial_pop_density:
                        self.Cells.append(Cell.CopyCell(parent, i , j, self.TypeOffCells))
                        self.TypeOffCells += 1


        # We distribute all the parameters to the class
        self.MutationFactor = mutation_factor
        self.NumberOfGenes = number_of_genes
        self.FitnessPower = fitness_power
        self.RegenaRate = regen_rate
        self.DivisionThreshold = division_thres
        self.DivisionTimeSteps = division_timesteps

        self.xSize = x_size
        self.ySize = y_size

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

    def Migration(self, migrating_cells, stationary_cells):
        """"
        My current implementation of migration is somewhat complex
        and time consuming. Randomly choosing which cells has priority
        might seem like a way better option (discuss with Bram).
        """
        # All the migrating cells propose a new position
        proposals = [(cell, *cell.propose_move())
                     for cell in migrating_cells]

        # Migrating cost one resource.
        for cell, _, _ in proposals:
            cell.CRL -= 1

        # The stationary cells propose
        # their current position
        proposals += [(cell, cell.iPos, cell.jPos)
                      for cell in stationary_cells]

        # We initialize a dictionary containing
        # the final positions, where the index of
        # the cell in self.Cells is the key and
        # the position is the value.
        final_pos = {idx: (ni, nj) for idx, (cell, ni, nj) in enumerate(proposals)}

        # We then look if there are any coflicts,
        # meaning that more than one cell wants to
        # inhibit a grid position. We loop over
        # this process till there is no conflict present
        # any more
        conflict = True
        attempts = 0
        max_attempts = 100

        while conflict and attempts < max_attempts:
            conflict = False
            attempts += 1
            # We identify a target map, which is
            # a dictionary where the keys are
            # positions and the values are
            # the cells that want to inhibit that
            # position
            map = {}
            for idx, pos in final_pos.items():
                map.setdefault(pos, []).append(idx)

            # We loop over all the keys in the
            # map
            for pos, group in map.items():
                # We ignore all the map items
                # with only one candidate:
                # there is no conflict in this case
                if len(group) == 1:
                    continue

                # If there is a group with more
                # then one item, this means that there
                # is a conflict
                conflict = True

                # We sort the group by their CRL
                # value, where we assume that the
                # higher the CRL the more mass the
                # cell has
                group.sort(key=lambda i: proposals[i][0].CRL, reverse=True)

                # We then check if one of the cells is
                # currently in the grid position. In other
                # words, we check if one of the stationary cells
                # is in the conflict group
                owner_idx = None
                for idx in group:
                    cell = proposals[idx][0]
                    if (cell.iPos, cell.jPos) == pos:
                        owner_idx = idx
                        break

                # If this is the case, the owner stays in the
                # position and wins
                if owner_idx is not None:
                    winner = owner_idx
                else:
                    # Otherwise, the heaviest cell wins
                    winner = group[0]

                # Everyone who isn't the winner gets bounced back home.
                # The loop is continued to see if the bounce back has not
                # caused a new conflict.
                for loser in group:
                    if loser != winner:
                        cell = proposals[loser][0]
                        final_pos[loser] = (cell.iPos, cell.jPos)
        """"
        Currently I have the max attempts maxed out, but I think the 
        Migration will be changed to randomly prefer one or the other
        """
        if attempts == max_attempts:
            print("Warning: Migration conflict resolution maxed out!")

        # we assign the cells to their final position
        for idx, (cell, _, _) in enumerate(proposals):
            cell.iPos, cell.jPos = final_pos[idx]

    def Reproduce(self, cell, occupied):
        # First, we look if there is a neighboring position of the
        # Cell that is currently empty:
        random.shuffle(self.Directions)

        for i,j in self.Directions:
            # we apply warping
            di, dj = (i+ cell.iPos + self.xSize) % self.xSize, (j + cell.jPos + self.ySize) % self.ySize
            # If the grid is not occupied, we
            # reproduce the cell
            if (di,dj) not in occupied:
                # We set the division counter of the cell to
                # 0 and half it's CRL
                cell.Division_Steps = 0
                cell.CRL = np.floor(cell.CRL/2)
                # If one of the neighborhoods is empty,
                # we reproduce the current cell to this new position:
                new_cell = Cell.CopyCell(cell, di, dj, cell.ID)

                # The mutation might have caused the cell to die.
                # In that that case the cell is not reproduced and the
                # function returns True
                if not new_cell.Mutate(mutation_factor=self.MutationFactor):
                    return None

                # Otherwise the cell is reproduced
                return new_cell

        # For now we simply do notihing when there is no space free.
        # this will have to be updated when adhesion is applied
        return None

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

        # We create a array of all occupied grids:
        occupied = {(cell.iPos, cell.jPos) for cell in self.Cells}

        # We first check if there are cells that have a
        # CRL higher then the division threshold: these
        # are the dividing cells. We also want to
        # keep track of all new born cells
        dividing = []
        newborn_cells = []
        for cell in self.Cells:
            if cell.CRL > self.DivisionThreshold:
                # All cells that are deviding have a counter
                # that checks how long they hae been in the
                # devision process. This counter is increased by one.
                cell.Division_Steps += 1
                # If this counter exeeds some threshold, the cell divids.
                # The CRL will be shared among the two new cells, ensuring
                # that CRL > 0 and CRL < self.DivisionThreshold.
                if cell.Division_Steps >= self.DivisionTimeSteps:
                    child = self.Reproduce(cell, occupied)
                    if child is not None:
                        newborn_cells.append(child)
                        occupied.add((child.iPos, child.jPos))
                dividing.append(cell)

        migrating = [cell for cell in self.Cells if cell.CRL < self.DivisionThreshold]
        self.Migration(migrating,dividing)

        cells_to_keep = []
        # Cells either consume food when present
        for cell in self.Cells:
            if self.Grid[cell.iPos, cell.jPos] > 1:
                # If there are more then one resources, the cell will
                # take two resources.
                self.Grid[cell.iPos, cell.jPos] -= 2
                cell.CRL += 2
                cells_to_keep.append(cell)
            elif self.Grid[cell.iPos, cell.jPos] == 1:
                # If there are is one resource, the cell will
                # take only this resource
                self.Grid[cell.iPos, cell.jPos] -= 1
                cell.CRL += 1
                cells_to_keep.append(cell)
            elif cell.CRL > 0:
                    # If there is no food and the
                    # CRL of the cell is zero, the cell dies
                    # So if CRL is higher then zero, we keep the cell
                    cells_to_keep.append(cell)

        # Combine the surviving population and the newborns at the very end of the step
        self.Cells = cells_to_keep + newborn_cells

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
        distances = []
        minimum = 21
        for cell in self.Cells:
            # cell being alive in the next step,
            # we determine some properties about the grid
            # for the analysis.
            distances.append(cell.HammingDistance)
            if cell.HammingDistance < minimum:
                minimum = cell.HammingDistance

        # We update the analysis
        self.Timestep += 1
        self.TotalPopulation = len(distances)
        if self.TotalPopulation > 0:
            self.MeanDistance = np.mean(distances)
            self.STDDistance = np.std(distances)
            self.MinimalDistance = minimum
        else:
            self.MeanDistance = 0
            self.STDDDistance = 0
            self.MinimalDistance = self.NumberOfGenes

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
            self.MinimalDistance = self.NumberOfGenes


if __name__ == '__main__':
    model = Model()