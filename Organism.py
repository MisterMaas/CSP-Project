from __future__ import annotations
import math
from typing import TYPE_CHECKING
import numpy as np
import numpy.random as random

if TYPE_CHECKING:
    from Model import Model


class Organism:
    Model: Model
    TRL: int
    CRL: int
    Division_Steps: int
    CellAmount: int
    MaxFitness : float

    TestSame = 0

    """"
    Devision
    """
    # We keep track of the current resource level
    # this is 10 on initialisation. All cells in the
    # organism share their resources, and therefore have
    # the same CRL
    CRL = 10

    # The total resource level is the entire resource level
    # off all cells combined
    TRL : int

    # When a cell is in division, it takes some amount of
    # time. Therefore we have to check how far it
    # is in the devision process. All cells
    # are initialized with division steps equal to 0
    Division_Steps = 0


    def __init__(self, model, cell, pos):
        self.Model = model
        self.Cells = [cell]
        self.Positions = [pos]
        self.CRL = 10
        self.TRL = 0
        self.Division_Steps = 0
        self.CellAmount = 1
        self.Directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.MaxFitness = cell.Fitness

    def Warp(self, direction, position):
        di, dj = direction
        i, j = position
        return (int((i + di) % self.Model.xSize), int((j + dj) % self.Model.ySize))

    def Divide(self):
        # We take a random amount of cells
        # from the cells that make up the organism.
        if self.CellAmount>1:
            r = random.randint(1, self.CellAmount)
        else:
            r = 1
        selected_indices = np.random.choice(self.CellAmount, size=r, replace=False)

        for i in selected_indices:
            # We determine a random parent of the cell
            parent = self.Cells[i]

            # We randomly select one cell that is on the border of the
            # organism. This can only be the case if it has less then
            # 4 Von Neuman neighbors
            shuffled_dirs = list(self.Directions)
            random.shuffle(shuffled_dirs)
            shuffled_cells = list(self.Cells)
            random.shuffle(shuffled_cells)

            # We keep a boolean to check wether a cell
            # has been placed:
            placed = False
            for cell in shuffled_cells:
                if placed:
                    break
                for dir in shuffled_dirs:
                    # We look at a random position
                    di, dj = self.Warp(dir, (cell.iPos, cell.jPos))
                    # We make sure that cells cannot divide on
                    # occupied space
                    if (di, dj) in self.Model.Occupied:
                        continue
                    # If this position is not occupied by a cell of the
                    # organism, we duplicate the parent to this position
                    if (di,dj) not in self.Positions:
                        # The cell of the parent is placed at the border of the
                        # organism.
                        new_cell = parent.CopyCell(parent, di, dj, self.Cells[i].ID)
                        alive = new_cell.Mutate(mutation_factor=self.Model.MutationFactor)
                        # The cell might die after mutation
                        if not alive:
                            # We then ignore this cell
                            continue
                        # The CRL is devided between the new cell and
                        # the parent
                        new_cell.CRL = parent.CRL // 2
                        parent.CRL = parent.CRL // 2
                        # The new cell posibly mutates

                        # After the cell is formed, it can stick to the organism
                        # or can become it's own organism depending on the fitness
                        if random.random() < (cell.Fitness * self.MaxFitness):
                            # Otherwise the cell will stick to the organism
                            self.Cells.append(new_cell)
                            self.Positions.append((di, dj))
                            self.CellAmount += 1
                            self.Model.Occupied[(di, dj)] = self
                            cell.UniCellular = False
                            new_cell.UniCellular = False
                            if cell.Fitness > self.MaxFitness:
                                self.MaxFitness = cell.Fitness
                        else:
                            org = Organism(self.Model, new_cell, (di, dj))
                            self.Model.Organisms.add(org)
                            self.Model.Occupied[(di, dj)] = org
                        placed = True
                        break

        # The Resources are spread over all cells
        self.UpdateCRL()
        # At last we need to update the current resource level
        # and the division steps
        self.Division_Steps = 0
        return None

    def MigrateSingleCell(self):
        # We remember the last position
        last_pos = self.Positions[0]
        cell = self.Cells[0]

        # Moving cost one resource
        cell.CRL -= 1
        if cell.CRL == 0:
            # If the resource level is 0,
            # the cell dies and does not perform
            # any action
            return
        # We try out all possible directions
        # for a given organism
        directions = list(self.Model.Directions)
        random.shuffle(directions)
        for dir in directions:
            # The migrating cell proposes a position
            proposed = self.Warp(dir, last_pos)
            # if this position is not occupied, the organism can move to
            # this position:
            if proposed not in self.Model.Occupied:
                # If the cell is free to move, it
                # will move to this position
                self.Model.Occupied.pop(last_pos, None)

                # Move the cell
                cell.iPos, cell.jPos = proposed
                self.Positions = [proposed]

                # Add to new position in the dictionary
                self.Model.Occupied[proposed] = self
                break
            # When the grid is not free, there is a probability
            # that the cell adheres to the colliding organism. They
            # then form a new organism.
            collided_organism = self.Model.Occupied[proposed]
            if random.random() < (cell.Fitness * collided_organism.MaxFitness):

                # Delete the single celled organism
                self.Model.Occupied.pop(last_pos, None)
                self.Model.Organisms.remove(self)
                cell.UniCellular = False

                cell.iPos, cell.jPos = last_pos

                # Cell gets absorbed by neighbor
                collided_organism.Cells.append(cell)
                collided_organism.Positions.append(last_pos)

                if collided_organism.MaxFitness < cell.Fitness:
                    collided_organism.MaxFitness = cell.Fitness

                if collided_organism.CellAmount == 1:
                    collided_organism.Cells[0].UniCellular = False
                collided_organism.CellAmount += 1

                # Update the grid
                self.Model.Occupied[last_pos] = collided_organism

                # Update resource of new organism
                collided_organism.UpdateCRL()
                break



    def MigrateMultiCell(self):
        # Keeps track of the current position
        fallback = list(self.Positions)

        # We keep a dictionary that holds the
        # directions as keys and the total recourse
        # as it's value
        dictionary = {}

        # we keep track of all the positions that are occupied
        others_occupied = set(self.Model.Occupied.keys()) - set(self.Positions)

        # We say that the model possibly proposes it's
        # current position. It does this when there is
        # is no other posistion possible
        proposed = list(self.Positions)
        # We try out all possible directions
        # for a given organism
        directions = list(self.Model.Directions)
        random.shuffle(directions)
        for dir in directions:
            # The migrating cell proposes a candidate
            candidate = [self.Warp(dir, pos) for pos in self.Positions]
            candidate_set = set(candidate)

            if candidate_set.isdisjoint(others_occupied):
                # If the organism is multi-cellular, all possible
                # directions are checked and stored in the dictionary
                TRL = sum(self.Model.Grid[di, dj] for di, dj in candidate)
                dictionary[tuple(candidate)] = TRL

        # We check wether the dict is is non empty.
        if dictionary:
            # We select the best direction:
            best = max(dictionary, key=lambda k: dictionary[k])
            proposed = list(best)
        else:
            # otherwise we fall back to the current position
            proposed = fallback

        # We remove the old positions from occupied
        for pos in self.Positions:
            self.Model.Occupied.pop(pos, None)

        # Assign all the cells to the new position
        for cell, new_pos in zip(self.Cells, proposed):
            cell.iPos, cell.jPos = new_pos
            cell.CRL -= 1

        # Assign the organism to the new position
        self.Positions = proposed

        # And update occupied
        for pos in proposed:
            self.Model.Occupied[pos] = self

    def UpdateCRL(self):
        self.TRL = sum(cell.CRL for cell in self.Cells)
        self.CRL = math.floor(self.TRL / self.CellAmount)
        # Sync back so cells and organism agree
        for cell in self.Cells:
            cell.CRL = self.CRL
        return self.CRL



    # def DivideOrganism(self):
    #     ### This was pretty much vibe coded.
    #     # not sure if we want to add this
    #
    #     # After the cells are divided, it is also possible for
    #     # the organism to devide. This is simply done by
    #     # "Cutting it in half". We want the probability of this
    #     # happening to increase depending on the size of the
    #     # organism
    #
    #     # We let the probibility of dividing depend on the
    #     # average fitness of the organism
    #     avg_fitness = np.mean([cell.Fitness for cell in self.Cells])
    #     K_base = 8.0
    #     alpha = 2.0
    #     K_dynamic = K_base * (1.0 + alpha * avg_fitness)
    #
    #     # We implement a Hill Function probability distribution
    #     n = 4
    #     prob = (self.CellAmount ** n) / (K_dynamic ** n + self.CellAmount ** n)
    #
    #     # We take a random value and devide the cell
    #     if random.random() < prob and len(self.Positions) >= 2:
    #         # Directions for Von Neumann neighborhood (up, down, left, right)
    #         vn_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    #
    #         # 1. Pick a random starting cell from the organism to seed Group 1
    #         # Shuffling ensures the "cut" direction changes randomly every time
    #         all_positions = list(self.Positions)
    #         random.shuffle(all_positions)
    #         start_pos = all_positions[0]
    #
    #         group_1 = {start_pos}
    #         queue = [start_pos]
    #         target_size = len(self.Positions) // 2
    #
    #         # 2. Flood-fill using Von Neumann rules to gather exactly half the cells
    #         while queue and len(group_1) < target_size:
    #             current = queue.pop(0)
    #
    #             # Check all 4 Von Neumann directions
    #             for direction in vn_dirs:
    #                 # We use your existing Warp method to keep coordinates wrapped safely
    #                 neighbor = self.Warp(direction, current)
    #
    #                 # The neighbor MUST be a part of this organism, and not yet in group_1
    #                 if neighbor in self.Positions and neighbor not in group_1:
    #                     group_1.add(neighbor)
    #                     queue.append(neighbor)
    #
    #                     # Stop immediately if we hit our target split size
    #                     if len(group_1) == target_size:
    #                         break
    #
    #         # 3. Everything left over automatically becomes Group 2
    #         group_2 = set(self.Positions) - group_1
    #
    #         # 4. Convert back to lists for your model processing
    #         group_1 = list(group_1)
    #         group_2 = list(group_2)
    #
    #         # --- Structural Verification ---
    #         # In extremely rare cases (like a highly wound, snake-like shape),
    #         # taking half the cells can clip the "tail", leaving group_2 fragmented.
    #         # Let's add a quick check to make sure Group 2 is also validly connected.
    #         def is_contiguous(group_list):
    #             if not group_list: return False
    #             visited = {group_list[0]}
    #             check_q = [group_list[0]]
    #             while check_q:
    #                 curr = check_q.pop(0)
    #                 for d in vn_dirs:
    #                     nb = self.Warp(d, curr)
    #                     if nb in group_list and nb not in visited:
    #                         visited.add(nb)
    #                         check_q.append(nb)
    #             return len(visited) == len(group_list)
    #
    #         # If the split broke group_2's contiguity, fall back or skip this step
    #         # to preserve perfect structural integrity.
    #         if not is_contiguous(group_2):
    #             # Fallback: Just skip division this step, it will try a different
    #             # random seed/angle on the next timestep!
    #             return None


