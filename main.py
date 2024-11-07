class SudokuSolver:

    # Sudoku configuration
    SUDOKU_SIZE = 9
    # Algorithm configuration
    HEURISTIC_GOAL = SUDOKU_SIZE * SUDOKU_SIZE * 9 
    POPULATION_NUM = 1024
    MUTATION_RATE = 0.05

    def __init__(self, matrix):
        self.mat = matrix
        self.fixed_mat = self.MaskFixedMatrix()
        self.ppl_list = []

    # Tai
    def AlgorithmRun(self):
        self.InitialPopulationGenerator()

        for idx in range(0, self.POPULATION_NUM):
            if self.HeuristicFunction(self.ppl_list[idx]) == self.HEURISTIC_GOAL:
                print(self.ppl_list[idx])
                return

        while True:
            sel_list = []
            # TODO: chon ra 1024 genes tu 1024 genes ban dau (Selection phase) -> sel_list

            cross_list = []
            # TODO: chon ngau nhien lai tao lan luot 1024 successors -> cross_list

            # rand_temp = random(self.MUTATION_RATE)
            # TODO: Lan luot chon  1024 thang trong cross_list quyet dinh co mutation hay ko
            # TODO: -> Cap nhat tat ca vo self.ppl_list
            pass

    # Tai
    def InitialPopulationGenerator(self):
        # TODO: generate population
        # ppl_list.append(new_ppl)
        for idx in range(0, self.POPULATION_NUM):
            for col_idx in range(0, self.SUDOKU_SIZE):
                for row_idx in range(0, self.SUDOKU_SIZE):
                    pass
            pass
        pass

    # Duong
    def HeuristicFunction(self, mat):
        score = 0

        # Check rows
        for row in mat:
            score += len(set(row))

        # Check columns
        for col in range(self.SUDOKU_SIZE):
            col_set = set()
            for row in range(self.SUDOKU_SIZE):
                col_set.add(mat[row][col])
            score += len(col_set)

        # Check 3x3 grids
        for grid_row in range(0, self.SUDOKU_SIZE, 3):
            for grid_col in range(0, self.SUDOKU_SIZE, 3):
                grid_set = set()
                for row in range(3):
                    for col in range(3):
                        grid_set.add(mat[grid_row + row][grid_col + col])
                score += len(grid_set)

        return score

    # Tai
    def SelectionFunction(self):
        # Use self.ppl_list
        pass

    # Duong
    def CrossoverFunction(self, mat1, mat2):
        pass

    # Duong
    def MutationFunction(self, mat):
        pass

    # Tai
    def Tournament(self, mat1, mat2):
        # return mat_winner
        pass

    # Miscellaneous
    def MaskFixedMatrix(self):
        ret_mat = []
        for i in range(len(self.mat)):
            for j in range(len(self.mat[i])):
                if self.mat[i][j] != 0:
                    ret_mat[i][j] = 1
        return ret_mat


initial_mat = [[1, 2, 3, 4, 5, 6, 7, 8, 9],
               [1, 2, 3, 4, 5, 6, 7, 8, 9],
               [1, 2, 3, 4, 5, 6, 7, 8, 9],
               [1, 2, 3, 4, 5, 6, 7, 8, 9],
               [1, 2, 3, 4, 5, 6, 7, 8, 9],
               [1, 2, 3, 4, 5, 6, 7, 8, 9],
               [1, 2, 3, 4, 5, 6, 7, 8, 9],
               [1, 2, 3, 4, 5, 6, 7, 8, 9],
               [1, 2, 3, 4, 5, 6, 7, 8, 9]]

# An example of a solved matrix
sovle_mat =   [[4, 9, 6, 1, 5, 7, 8, 2, 3],
               [2, 1, 8, 3, 9, 6, 7, 5, 4],
               [7, 5, 3, 2, 8, 4, 1, 6, 9],
               [6, 4, 9, 8, 3, 1, 2, 7, 5],
               [5, 3, 1, 6, 7, 2, 9, 4, 8],
               [8, 2, 7, 5, 4, 9, 6, 3, 1],
               [9, 6, 2, 4, 1, 5, 3, 8, 7],
               [1, 8, 5, 7, 6, 3, 4, 9, 2],
               [3, 7, 4, 9, 2, 8, 5, 1, 6]]

if __name__ == '__main__':
    print('PyCharm')

