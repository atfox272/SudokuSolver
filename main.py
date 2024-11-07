import random
import copy


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
        score = 0

        # Kiểm tra hàng
        for row in mat:
            score += len(set(row))

        # Kiểm tra cột
        for col in range(self.SUDOKU_SIZE):
            col_set = set()
            for row in range(self.SUDOKU_SIZE):
                col_set.add(mat[row][col])
            score += len(col_set)

        # Kiểm tra các ô 3x3
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
    def CrossoverFunction(self, population_list):
        new_population = []

        while len(population_list) > 1:
            # Lấy danh sách các chỉ số của population_list
            indices = list(range(len(population_list)))

            # Chọn ngẫu nhiên hai chỉ số khác nhau từ danh sách
            idx1, idx2 = random.sample(indices, 2)

            mat1 = population_list.pop(idx1)
            # Nếu idx2 lớn hơn idx1, giảm idx2 đi 1 vì danh sách đã bị thay đổi sau khi pop idx1
            if idx2 > idx1:
                idx2 -= 1

            # Lấy ma trận thứ hai từ danh sách
            mat2 = population_list.pop(idx2)

            # Chọn điểm cắt ngẫu nhiên theo cột (3 hoặc 6)
            crossover_point = random.choice([3, 6])

            # Tạo hai ma trận con mới
            offspring1 = [row[:crossover_point] + row[crossover_point:] for row in mat1]
            offspring2 = [row[:crossover_point] + row[crossover_point:] for row in mat2]

            # Hoán đổi các phần của hai ma trận
            for i in range(self.SUDOKU_SIZE):
                offspring1[i] = mat1[i][:crossover_point] + mat2[i][crossover_point:]
                offspring2[i] = mat2[i][:crossover_point] + mat1[i][crossover_point:]

            # Thêm hai ma trận con mới vào danh sách dân số mới
            new_population.append(offspring1)
            new_population.append(offspring2)

    # Nếu còn lại một ma trận trong danh sách, thêm nó vào dân số mới
        if population_list:
            new_population.append(population_list.pop())

        return new_population

    # Duong
    def MutationFunction(self, mat):
        # Tạo tỉ lệ ngẫu nhiên
        if random.random() > self.MUTATION_RATE:
            return mat
        # Chọn ngẫu nhiên một hàng
        row = random.randint(0, self.SUDOKU_SIZE - 1)
        
        non_fixed_indices = []
        # Tìm các chỉ số không cố định trong hàng
        for col in range(self.SUDOKU_SIZE):
            if self.fixed_mat[row][col] == 0:
                non_fixed_indices.append(col)

        if not non_fixed_indices:
            return mat

        empty_indices = []
        # Tìm các chỉ số trống trong hàng
        for col in non_fixed_indices:
            if mat[row][col] == 0:
                empty_indices.append(col)

        # Nếu có chỉ số trống, chọn ngẫu nhiên một chỉ số và gán giá trị ngẫu nhiên từ 1 đến 9
        if empty_indices:
            col = random.choice(empty_indices)
            mat[row][col] = random.randint(1, 9)
        else:
            # Nếu không có chỉ số trống, chọn ngẫu nhiên một chỉ số không cố định và gán giá trị
            col = random.choice(non_fixed_indices)
            mat[row][col] = random.randint(1, 9)

        return mat

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

