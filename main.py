import random
import copy


class SudokuSolver:
    # Sudoku configuration
    SUDOKU_SIZE = 9
    # Solver configuration
    GENERATION_RESET_MODE = 1
    # Algorithm configuration
    HEURISTIC_GOAL = SUDOKU_SIZE * SUDOKU_SIZE * 3
    MUTATION_RATE = 0.1
    POPULATION_NUM = 2048
    TOURNAMENT_SIZE = 2
    # Deep configuration
    GENERATION_LIMIT = 300

    def __init__(self, matrix):
        self.mat = matrix
        self.fixed_mat = self.MaskFixedMatrix()
        self.ppl_list = []
        self.generation_count = 0
        self.highest_score_record = 0
        self.highest_score_mat = [[0 for _ in range(self.SUDOKU_SIZE)] for _ in range(self.SUDOKU_SIZE)]
        self.prev_highest_score_record = 0
        self.prev_highest_score_mat = [[0 for _ in range(self.SUDOKU_SIZE)] for _ in range(self.SUDOKU_SIZE)]

    # Tai
    def AlgorithmRun(self):
        generation_reset_limit = self.GENERATION_RESET_MODE*100 + 1
        for i in range(generation_reset_limit):
            # Clear data
            self.ClearBuffer()

            # Initial population phase
            self.InitialPopulation()
            # Check if population list has the goal
            for idx in range(0, self.POPULATION_NUM):
                # Record highest score
                if self.HeuristicFunction(self.ppl_list[idx]) > self.highest_score_record:
                    self.prev_highest_score_record = self.highest_score_record
                    self.prev_highest_score_mat = copy.deepcopy(self.highest_score_mat)
                    self.highest_score_record = self.HeuristicFunction(self.ppl_list[idx])
                    self.highest_score_mat = copy.deepcopy(self.ppl_list[idx])
                if self.HeuristicFunction(self.ppl_list[idx]) == self.HEURISTIC_GOAL:
                    print(f'[INFO]: Found the goal at {self.generation_count}th')
                    self.PrintInfo()
                    return
            while True:
                if self.generation_count >= self.GENERATION_LIMIT:
                    print(f'[INFO]: The number of generations is exceed limitation')
                    self.PrintInfo()
                    if self.GENERATION_RESET_MODE == 1:
                        break
                    else:
                        return
                # Selection phase
                sel_list = self.SelectionFunction()

                # Crossover phase
                cross_list = self.CrossoverFunction(sel_list)

                # Mutation phase
                for idx in range(0, self.POPULATION_NUM):
                    cross_list[idx] = self.MutationFunction(cross_list[idx])
                self.ppl_list = cross_list

                # Check if population list has the goal
                for idx in range(self.POPULATION_NUM):
                    # Record highest score
                    if self.HeuristicFunction(self.ppl_list[idx]) > self.highest_score_record:
                        self.prev_highest_score_record = self.highest_score_record
                        self.prev_highest_score_mat = copy.deepcopy(self.highest_score_mat)
                        self.highest_score_record = self.HeuristicFunction(self.ppl_list[idx])
                        self.highest_score_mat = copy.deepcopy(self.ppl_list[idx])
                    # Check goal state
                    if self.HeuristicFunction(self.ppl_list[idx]) == self.HEURISTIC_GOAL:
                        print(f'[INFO]: Found the goal at {self.generation_count}th')
                        self.PrintInfo()
                        return
                self.generation_count += 1

    # Tai
    def InitialPopulation(self):
        for idx in range(0, self.POPULATION_NUM):
            ppl_mat = copy.deepcopy(self.mat)
            for col_idx in range(0, self.SUDOKU_SIZE):
                # Find all zero positions in the column
                zero_positions = [row for row in range(self.SUDOKU_SIZE) if ppl_mat[row][col_idx] == 0]
                zero_count = len(zero_positions)
                # Find the number that have appeared
                existing_numbers = {ppl_mat[row][col_idx] for row in range(self.SUDOKU_SIZE) if ppl_mat[row][col_idx] != 0}
                # Find all empty
                positions_to_replace = zero_positions
                # Random number 1 -> 9 (not appear in the column) to insert
                numbers_to_insert = random.sample([num for num in range(1, 10) if num not in existing_numbers],zero_count)
                for pos, num in zip(positions_to_replace, numbers_to_insert):
                    ppl_mat[pos][col_idx] = num
                # 1 <= Zero positions <= 2
            self.ppl_list.append(ppl_mat)

    # Duong
    def HeuristicFunction(self, mat):
        score = 0

        # Kiểm tra hàng - bỏ qua số 0
        for row in mat:
            score += len({x for x in row if x != 0})

        # Kiểm tra cột - bỏ qua số 0
        for col in range(self.SUDOKU_SIZE):
            col_set = {mat[row][col] for row in range(self.SUDOKU_SIZE) if mat[row][col] != 0}
            score += len(col_set)

        # Kiểm tra các ô 3x3 - bỏ qua số 0
        for grid_row in range(0, self.SUDOKU_SIZE, 3):
            for grid_col in range(0, self.SUDOKU_SIZE, 3):
                grid_set = set()
                for row in range(3):
                    for col in range(3):
                        value = mat[grid_row + row][grid_col + col]
                        if value != 0:
                            grid_set.add(value)
                score += len(grid_set)
        return score

    # Tai
    def SelectionFunction(self):
        sel_list = []
        # Host 1024 tournaments
        for ppl_idx in range(self.POPULATION_NUM):
            # 3-way Tournament: Select 3 individuals from 1024 randomly
            selected_indices = random.sample(range(len(self.ppl_list)), self.TOURNAMENT_SIZE)
            tournament = [self.ppl_list[i] for i in selected_indices]
            # Select the best individual in the Tournament
            best_individual = max(tournament, key=self.HeuristicFunction)
            # Append to the selected list
            sel_list.append(best_individual)
        return sel_list

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

            # Chọn ngẫu nhiên giữa cắt theo hàng hoặc cột
            # if random.random() < 0.5:
                # Cắt theo cột
            crossover_point = random.choice([3, 6])

            # Tạo hai ma trận con mới
            offspring1 = [row[:crossover_point] + row[crossover_point:] for row in mat1]
            offspring2 = [row[:crossover_point] + row[crossover_point:] for row in mat2]

            # Hoán đổi các phần của hai ma trận
            for i in range(self.SUDOKU_SIZE):
                offspring1[i] = mat1[i][:crossover_point] + mat2[i][crossover_point:]
                offspring2[i] = mat2[i][:crossover_point] + mat1[i][crossover_point:]
            # else:
            #     # Cắt theo hàng
            #     crossover_point = random.choice([3, 6])

            #     # Tạo hai ma trận con mới
            #     offspring1 = mat1[:crossover_point] + mat2[crossover_point:]
            #     offspring2 = mat2[:crossover_point] + mat1[crossover_point:]

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

        if random.random() < 0.5:
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
                if len(non_fixed_indices) >= 2:
                    # Nếu không có chỉ số trống, chọn ngẫu nhiên một chỉ số không cố định và gán giá trị
                    col1, col2 = random.sample(non_fixed_indices, 2)
                    mat[row][col1], mat[row][col2] = mat[row][col2], mat[row][col1]
        else:
            # Chọn ngẫu nhiên một cột
            col = random.randint(0, self.SUDOKU_SIZE - 1)

            non_fixed_indices = []
            # Tìm các chỉ số không cố định trong cột
            for row in range(self.SUDOKU_SIZE):
                if self.fixed_mat[row][col] == 0:
                    non_fixed_indices.append(row)

            if not non_fixed_indices:
                return mat

            empty_indices = []
            # Tìm các chỉ số trống trong cột
            for row in non_fixed_indices:
                if mat[row][col] == 0:
                    empty_indices.append(row)

            # Nếu có chỉ số trống, chọn ngẫu nhiên một chỉ số và gán giá trị ngẫu nhiên từ 1 đến 9
            if empty_indices:
                row = random.choice(empty_indices)
                mat[row][col] = random.randint(1, 9)
            else:
                if len(non_fixed_indices) >= 2:
                    # Nếu không có chỉ số trống, chọn ngẫu nhiên một chỉ số không cố định và gán giá trị
                    row1, row2 = random.sample(non_fixed_indices, 2)
                    mat[row1][col], mat[row2][col] = mat[row2][col], mat[row1][col]

        return mat

    # Miscellaneous
    def MaskFixedMatrix(self):
        ret_mat = [[0 for _ in range(self.SUDOKU_SIZE)] for _ in range(self.SUDOKU_SIZE)]
        for i in range(len(self.mat)):
            for j in range(len(self.mat[i])):
                if self.mat[i][j] != 0:
                    ret_mat[i][j] = 1
        return ret_mat

    def PrintInfo(self):
        print(f'[INFO]: Total generations: {self.generation_count}')
        print(f'[INFO]: Highest score record: {self.highest_score_record}')
        print(f'[INFO]: Highest score matrix:')
        for row in range(len(self.highest_score_mat)):
            print(self.highest_score_mat[row])
        print(f'[INFO]: Previous highest score record: {self.prev_highest_score_record}')
        print(f'[INFO]: Previous highest score matrix:')
        for row in range(len(self.prev_highest_score_mat)):
            print(self.prev_highest_score_mat[row])
        print('\n')

    def ClearBuffer(self):
        self.ppl_list = []
        self.generation_count = 0
        self.highest_score_record = 0
        self.highest_score_mat = [[0 for _ in range(self.SUDOKU_SIZE)] for _ in range(self.SUDOKU_SIZE)]
        self.prev_highest_score_record = 0
        self.prev_highest_score_mat = [[0 for _ in range(self.SUDOKU_SIZE)] for _ in range(self.SUDOKU_SIZE)]


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
            num_to_remove = random.randint(4, 7)  # Số ô cần xóa trong hàng này
            # num_to_remove = 5  # Số ô cần xóa trong hàng này
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
# unfilled_sudoku_board = [[0, 0, 0, 6, 4, 0, 0, 0, 7],
#                          [0, 0, 0, 9, 7, 1, 0, 3, 2],
#                          [1, 0, 0, 2, 0, 3, 5, 0, 0],
#                          [5, 1, 0, 0, 6, 0, 7, 9, 8],
#                          [0, 0, 0, 5, 0, 9, 0, 0, 0],
#                          [9, 6, 4, 0, 3, 0, 0, 2, 5],
#                          [0, 0, 1, 8, 0, 6, 0, 0, 4],
#                          [3, 5, 0, 1, 9, 4, 0, 0, 0],
#                          [6, 0, 0, 0, 5, 7, 0, 0, 0]
#                          ]
sudoku_solver = SudokuSolver(unfilled_sudoku_board)

print('[INFO]: Filled Sudoku')
for row in range(len(filled_sudoku_board[0])):
    print(filled_sudoku_board[row])
print('\n')

print('[INFO]: Unfilled Sudoku')
for row in range(len(unfilled_sudoku_board[0])):
    print(unfilled_sudoku_board[row])
print('\n')

sudoku_solver.AlgorithmRun()

# print(selected_list)
# for idx in range(len(selected_list)):
#     for row in range(len(selected_list[idx])):
#         print(selected_list[idx][row])
#     print('\n')


if __name__ == '__main__':
    print('PyCharm')

