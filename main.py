import random
import copy


class SudokuSolver:

    # Sudoku configuration
    SUDOKU_SIZE = 9
    # Algorithm configuration
    HEURISTIC_GOAL = SUDOKU_SIZE * SUDOKU_SIZE * 3
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
        for idx in range(0, self.POPULATION_NUM):
            ppl_mat = copy.deepcopy(self.mat)
            for row_idx in range(0, self.SUDOKU_SIZE):
                # Find all zero positions
                zero_positions = [i for i, x in enumerate(ppl_mat[row_idx]) if x == 0]
                # Find the numbers that have appeared (1 -> 9)
                existing_numbers = set(ppl_mat[row_idx]) - {0}
                # Find all missing numbers
                missing_numbers = list(set(range(1, 10)) - existing_numbers)
                # Select the number to fill in (randomly)
                fill_num = random.randint(0, min(len(zero_positions), 2))
                numbers_to_insert = random.sample(missing_numbers, fill_num)
                positions_to_fill = random.sample(zero_positions, fill_num)
                # Fill in
                for pos, num in zip(positions_to_fill, numbers_to_insert):
                    ppl_mat[row_idx][pos] = num
            self.ppl_list.append(ppl_mat)

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
        ret_mat = [[0 for _ in range(self.SUDOKU_SIZE)] for _ in range(self.SUDOKU_SIZE)]
        for i in range(len(self.mat)):
            for j in range(len(self.mat[i])):
                if self.mat[i][j] != 0:
                    ret_mat[i][j] = 1
        return ret_mat


class SudokuGenerator:
    def __init__(self):
        self.filled_sudoku_board = self.generate_sudoku()
        self.unfilled_sudoku_board = self.remove_numbers()

    def generate_sudoku(self):
        board = [[0 for _ in range(9)] for _ in range(9)]
        self.fill_board(board)
        return board

    def is_valid(self, board, row, col, num):
        # Kiểm tra hàng
        if num in board[row]:
            return False
        # Kiểm tra cột
        if num in [board[r][col] for r in range(9)]:
            return False
        # Kiểm tra ô vuông 3x3
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if board[start_row + i][start_col + j] == num:
                    return False

        return True

    def fill_board(self, board):
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    nums = list(range(1, 10))
                    random.shuffle(nums)
                    for num in nums:
                        if self.is_valid(board, row, col, num):
                            board[row][col] = num
                            if self.fill_board(board):
                                return True
                            board[row][col] = 0
                    return False
        return True

    def remove_numbers(self):
        tmp_board = [[0 for _ in range(9)] for _ in range(9)]
        for row in range(9):
            num_to_remove = random.randint(4, 6)  # Số ô cần xóa trong hàng này
            positions = random.sample(range(9), num_to_remove)  # Chọn vị trí ngẫu nhiên để xóa
            for col in range(0, 9):
                if col in positions:
                    tmp_board[row][col] = 0
                else:
                    tmp_board[row][col] = self.filled_sudoku_board[row][col]
        return tmp_board


sudoku_board = SudokuGenerator()
filled_sudoku_board = sudoku_board.filled_sudoku_board
unfilled_sudoku_board = sudoku_board.unfilled_sudoku_board
sudoku_solver = SudokuSolver(unfilled_sudoku_board)

print('[INFO]: Filled Sudoku')
for row in range(len(filled_sudoku_board[0])):
    print(filled_sudoku_board[row])
print('\n')

print('[INFO]: Unfilled Sudoku')
for row in range(len(unfilled_sudoku_board[0])):
    print(unfilled_sudoku_board[row])
print('\n')

print('[INFO]: Population')
sudoku_solver.InitialPopulationGenerator()
for idx in range(len(sudoku_solver.ppl_list)):
    for row in range(len(sudoku_solver.ppl_list[idx])):
        print(sudoku_solver.ppl_list[idx][row])
    print('\n')

if __name__ == '__main__':
    print('PyCharm')

