class SudokuSolver:

    # Sudoku configuration
    SUDOKU_SIZE = 9
    # Algorithm configuration
    HEURISTIC_GOAL = SUDOKU_SIZE * 3
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
        # return score
        pass

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

if __name__ == '__main__':
    print('PyCharm')

