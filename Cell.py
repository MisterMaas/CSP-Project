from __future__ import annotations
import numpy as np
import numpy.random as random
import random as rnd
from typing import TYPE_CHECKING
import math
if TYPE_CHECKING:
    from GRNModel import Model

class Cell:
    """"
    GRN Parameters
    """
    # Represents how many unique genes
    # aka. how many gene id's there are.
    AmountOfTypes : int

    # Int that represents the number of different
    # gene types
    NumberOfGenes : int

    # Boolean that keeps track of wether a cell is
    # stable. This is added for optimalisation
    IsStable : bool

    Model : Model

    # Fitness that is caculated depending on
    # the hamming distance
    HammingDistance : int
    Fitness : float

    # Genome is a 4xn matrix, where:
    # Row 1 represents the gene ID
    # Row 2 represents the gene Threshold
    # Row 3 represents the gene's last expression
    # Row 4 represents the gene's current expression
    # n is the number of genes in the genome
    Genome: np.array

    # ExpressionPattern is a 1x20 matrix, where:
    # Each index is 1 when one of the corresponding
    # gene's is active, otherwise it's 0.
    ExpressionPattern: np.array

    # GRN is a nxn matrix, where:
    # Element a_ij is 0 when there is no edge from j to i
    # Element a_ij is 1 when j has an activating effect on i
    # Element a_ij is -1 when j has an inhibiting effect on i
    GRN: np.array

    """"
    Moving parameters
    """

    # We keep a boolean that keeps track of wether a
    # cell is part of a larger organism. We do this
    # via a boolean. All cells are initially
    # unicellular
    UniCellular = True

    # We want to keep track of the initial position of the cell
    iPos : int
    jPos : int


    # We further want to be able to distinghuish the different
    # cells
    ID : int

    """"
    Devision
    """
    # We keep track of the current resource level
    # this is 10 on initialisation
    CRL = 10

    # When a cell is in division, it takes some amount of
    # time. Therefore we have to check how far it
    # is in the devision process. All cells
    # are initialized with division steps equal to 0
    Division_Steps = 0

    def __init__(self, model, number_of_genes = 20):
        def InitiateGenome(number_of_genes):
            """
            Initiates the "pearl string" genome
            """
            # First we create a array from 1 to the number
            # of genes, representing the ID of the gene types.
            identification = np.arange(number_of_genes, dtype=int) + 1
            # The last expressions are just zeros on initiation
            last_expression = np.zeros(number_of_genes, dtype=int)
            # Then we create a random expression pattern
            # We also save this expression pattern to
            # the expression pattern
            expression = np.zeros(number_of_genes)
            self.ExpressionPattern = np.zeros(number_of_genes, dtype=int)
            # We do the same for the thresholds
            threshold = np.zeros(number_of_genes, dtype=int)
            for i in identification:
                exp = random.choice([0, 1])
                expression[i - 1] = exp
                self.ExpressionPattern[i - 1] = exp
                threshold[i-1] = random.choice([-2,-1,0,1,2])
            # We stack the identification and expression arrays
            # to get the intended matrix. Note that the last
            # expressions are the same as the current
            self.Genome = np.vstack((identification, threshold, last_expression, expression))

        def InitiateGRN(number_of_gens):
            """"
            Initiates the GRN as a nxn matrix
            """
            # First we initiate a nxn matrix containing
            # only zero values. We do the same for the
            # threshold map. We use a sparse matrix (containing
            # only few non-zero elements) for optimization.
            self.GRN = self.GRN = np.zeros((number_of_genes, number_of_genes), dtype=np.int8)

            # We then go over all the n genes in the genome.
            for i in range(number_of_genes):
                # We create a range from 0 to one, representing
                # the possible genes that the gene can regulate
                possible_edges = np.arange(number_of_genes)

                # The paper specified that, on initiation,
                # the GRN does not have an edge to itself,
                # nor have any parallel edges. We therefore
                # remove the gene itself and all genes that
                # already have an edge to the current gene
                # from the possible edges
                possible_edges = np.delete(possible_edges, i)
                for n, edges in enumerate(self.GRN[:, i]):
                    if edges != 0:
                        possible_edges = possible_edges[possible_edges != n]
                        if possible_edges.size <= 1:
                            return False

                # We then take a random number between 1 and 20.
                # This will represent the amount of binding sites
                # the GRN has after initiation.
                amount_of_binding_sites = random.binomial(len(possible_edges), (1 / len(possible_edges))) + 1
                # Then we make amount_of_binding_sites random choices
                # about where a gene connects to
                for m in range(amount_of_binding_sites):
                    j = random.choice(possible_edges)
                    # The selected index now means that there
                    # is an edge from gene j to gene i. We then
                    # randomly select wether this binding is
                    # activating or inhibiting (-1, 1)
                    self.GRN[j,i] = random.choice([-1, 1])
                    # Gene j is removed from the possible edges
                    possible_edges = np.delete(possible_edges, np.where(possible_edges == j))
            return True

        self.Model = model
        self.IsStable = False
        self.Fitness = 0
        self.HammingDistance = number_of_genes
        self.NumberOfGenes = number_of_genes
        self.AmountOfTypes = number_of_genes
        InitiateGenome(number_of_genes)

        succes = False

        while not succes:
            succes = InitiateGRN(number_of_genes)

    @classmethod
    def CopyCell(cls, parent, i_pos, j_pos, id):
        new_cell = cls.__new__(cls)
        new_cell.Model = parent.Model
        new_cell.NumberOfGenes = parent.NumberOfGenes
        new_cell.HammingDistance = parent.HammingDistance
        new_cell.IsStable = parent.IsStable
        new_cell.Fitness = parent.Fitness
        new_cell.Genome = np.copy(parent.Genome)
        new_cell.ExpressionPattern = np.copy(parent.ExpressionPattern)
        new_cell.GRN = np.copy(parent.GRN)
        new_cell.iPos = i_pos
        new_cell.jPos = j_pos
        new_cell.ID = id
        return new_cell

    def ExecutePropagation(self, propagation_steps = 11):

        def UpdateGenome():
            # First we set the last genome expression
            # to be the current
            self.Genome[2] = self.Genome[3]
            # Then we update the current expression
            # using the last expression.
            update_values = self.Genome[2] @ self.GRN
            # We take the threshold values
            thres = self.Genome[1]
            # And update the genome using vectorisation logic
            new_expr = np.where(update_values > thres, 1,
                                np.where(update_values == thres, self.Genome[2], 0))
            # We set this to the new expression of the
            # gene
            self.Genome[3] = new_expr

        def UpdateExpressionPattern():
            # We go over all the genes in the genome
            # and check their expression. If a gene
            # of one type is active in the genome,
            # it will be active int the expression patern.
            # Otherwise it will be 0.
            target_length = len(self.Model.Target)
            # We initialize the exprission pattern as a vector
            # only containing zeros
            self.ExpressionPattern = np.zeros(target_length)
            # We make a vector containing all active genes
            active_ids = self.Genome[0, self.Genome[3] == 1].astype(int) - 1
            # We transform this into a boolean array which says wether
            # index is valid
            valid = active_ids[(active_ids >= 0) & (active_ids < target_length)]
            # For all these valid expressions we fill in a 1
            self.ExpressionPattern[valid] = 1
            # We return false if there are no valid expressions
            return valid.size > 0

        # We do not preform propagation when the
        # cell is stable:
        if self.IsStable: return True

        # We perform the propagation steps
        for i in range(propagation_steps):
            # We first update the genome
            UpdateGenome()
            # UpdateExpressionPattern returns True if the
            # cell has died
            if not UpdateExpressionPattern():
                # If a cell has died, we return false
                return False
            # When the last expression pattern
            # and the current expression pattern are the
            # same, the GRN is stable and we exit the loop
            if np.array_equal(self.Genome[2], self.Genome[3]):
                self.IsStable = True
                break

        # After performing all the propagation steps, we can
        # update the fitness
        self.UpdateFitness()
        self.IsStable = True
        # If a cell is alive the function returns True
        return True

    def UpdateFitness(self):
        # First we have to calculate the hamming distance
        self.HammingDistance = np.sum(self.ExpressionPattern != self.Model.Target)
        max_possible_distance = len(self.Model.Target)
        self.Fitness = (1 - (self.HammingDistance / max_possible_distance)) ** self.Model.FitnessPower

    def Mutate(self, mutation_factor=1,
               p_g_dup=2e-4, p_g_del=3e-4,
               p_thres=5e-6, p_b_dup=2e-5,
               p_b_del=3e-5, p_innov=1e-5,
               p_wei_c=2e-5, p_per_c=2e-5,
               del_per=3e-5, dup_per=2e-5):

        def gene_duplication():
            idx = random.randint(0, self.Genome.shape[1])
            col = random.randint(0, self.Genome.shape[1])
            # Copy a random gene column
            donor_col = self.Genome[:, idx].copy()
            self.Genome = np.insert(self.Genome, col, donor_col, axis=1)
            # Copy its GRN column and row
            self.GRN = np.insert(self.GRN, col, self.GRN[:, idx], axis=1)
            self.GRN = np.insert(self.GRN, col, self.GRN[idx, :], axis=0)
            self.NumberOfGenes += 1
            return True

        def gene_deletion():
            idx = random.randint(0, self.Genome.shape[1])
            # Check whether the gene type at idx appears more than once.
            # If it's the last copy of that type, deleting it would make
            # that type go extinct, so we abort.
            gene_id = self.Genome[0, idx]
            # We select a gene
            self.Genome = np.delete(self.Genome, idx, axis=1)
            self.GRN = np.delete(self.GRN, idx, axis=0)
            self.GRN = np.delete(self.GRN, idx, axis=1)
            self.NumberOfGenes -= 1
            # We check if there are duplecates in the
            # genes
            copies = np.sum(self.Genome[0] == gene_id)
            # If there are no duplacates, the cell dies
            if copies < 1:
                return False
            else:
                return True

        def threshold_change():
            idx = random.randint(0, self.Genome.shape[1])
            self.Genome[1, idx] = random.choice([-2, -1, 0, 1, 2])
            return True

        def binding_duplication():
            # Pick a donor gene and copy one of its incoming edges to another gene
            donor = random.randint(0, self.NumberOfGenes)
            incoming = np.nonzero(self.GRN[:, donor])[0]
            if len(incoming) == 0:
                return
            regulator = incoming[random.randint(0, len(incoming))]
            weight = self.GRN[regulator, donor]
            recipient = random.randint(0, self.NumberOfGenes)
            self.GRN[regulator, recipient] = weight
            return True

        def binding_deletion():
            gene = random.randint(0, self.NumberOfGenes)
            incoming = np.nonzero(self.GRN[:, gene])[0]
            # Keep at least 1 binding, matching the C# guard
            if len(incoming) <= 1:
                return True
            regulator = incoming[random.randint(0, len(incoming))]
            self.GRN[regulator, gene] = 0
            return True

        def innovation():
            regulator = random.randint(0, self.NumberOfGenes)
            regulated = random.randint(0, self.NumberOfGenes)
            self.GRN[regulator, regulated] = random.choice([-1, 1])
            return True

        def weight_change():
            gene = random.randint(0, self.NumberOfGenes)
            incoming = np.nonzero(self.GRN[:, gene])[0]
            if len(incoming) == 0:
                return True
            regulator = incoming[random.randint(0, len(incoming))]
            self.GRN[regulator, gene] *= -1
            return True

        def performance_change():
            gene = random.randint(0, self.NumberOfGenes)
            incoming = np.nonzero(self.GRN[:, gene])[0]
            if len(incoming) == 0:
                return True
            old_regulator = incoming[random.randint(0, len(incoming))]
            new_regulator = random.randint(0, self.NumberOfGenes)
            weight = self.GRN[old_regulator, gene]
            self.GRN[old_regulator, gene] = 0
            self.GRN[new_regulator, gene] = weight
            return True

        def delete_percentage(p=0.15):
            # We decide how many cells will be deleted
            portion = math.floor(self.NumberOfGenes*p)
            # Then we indicate a starting index
            start = random.randint(0, self.NumberOfGenes - portion)
            delete_indexes = np.arange(portion) + start
            # Find the index of insertion
            insert = random.randint(0, self.NumberOfGenes)
            # We perform the deletion
            self.Genome = np.delete(self.Genome, delete_indexes, axis=1)
            self.GRN = np.delete(self.GRN, delete_indexes, axis=0)
            self.GRN = np.delete(self.GRN, delete_indexes, axis=1)
            self.NumberOfGenes -= portion
            # We check if there are duplecates in the
            # genes
            # If there are no duplacates, the cell dies
            unique = np.unique(self.Genome[0])
            return len(unique) == self.NumberOfGenes

        def duplicate_percentage(p=0.15):
            # We decide how many cells will be deleted
            portion = math.floor(self.NumberOfGenes*p)
            # Then we indicate a starting index
            start = random.randint(0, self.NumberOfGenes - portion)
            copy_indexes = np.arange(portion) + start
            # Take a insertion index
            insert = random.randint(0, self.NumberOfGenes)
            # Repeat the insertion point once per column/row being inserted
            insert_indices = [insert] * portion
            # Copy a random gene column
            donor_col = self.Genome[:, copy_indexes].copy()
            self.Genome = np.insert(self.Genome, insert_indices, donor_col, axis=1)
            # Copy its GRN column and row
            self.GRN = np.insert(self.GRN, insert_indices, self.GRN[:, copy_indexes], axis=1)
            self.GRN = np.insert(self.GRN, insert_indices, self.GRN[copy_indexes, :], axis=0)

            self.NumberOfGenes += portion
            self.IsStable = False
            return True

        # We want to make a distribution that shows the probability
        # that a certain mutation occurs. These probabilities
        # are taken from the paper of Cromback and Hogeweg
        distribution = [p_g_dup, p_g_del, p_thres, p_b_dup,
                        p_b_del, p_innov, p_wei_c, p_per_c,
                        del_per, dup_per]

        # We make a list of all the possible mutations. NOTE
        # that this MUST have the same order as the distribution
        mutations = [gene_duplication, gene_deletion, threshold_change,
                     binding_duplication, binding_deletion, innovation,
                     weight_change, performance_change, delete_percentage, duplicate_percentage]

        # Mutations can occur that delete the last gene of
        # a type. This would mean that the cell dies. Before
        # applying the mutation, the cell is alive.
        alive = True
        # Boolean to track wether mutation has occured
        mutated = False

        for i, mutation_function in enumerate(mutations):
            # We now take a random number and compare it
            # to the probabilites
            r = random.random()

            # We ignore the first element in compare
            if r < distribution[i] * mutation_factor:
                # Each mutation returns true if the
                # cell is still alive after mutation
                # and false if it died.
                alive = alive and mutation_function()

                # When the cell has mutated,
                # the GRN is unstable again.
                self.IsStable = False
                mutated = True

        # If a mutation has occured, the cell get's a new ID:
        if mutated:
            self.Model.TypeOffCells += 1
            self.ID = self.Model.TypeOffCells

        # When the cell is still alive,
        # we execute propagtion steps.
        if alive:
            # if the Propagation might cause the cell
            # to die
            alive = self.ExecutePropagation()

        # Mutation returns a bolean which
        # states wether the cell is alivbe or dead.
        return alive

    def propose_move(self, directions):
        di, dj = directions
        return (self.iPos + di) % self.Model.xSize, (self.jPos + dj) % self.Model.ySize



