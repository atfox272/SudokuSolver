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

