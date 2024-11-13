import copy
import time
import random
import threading
import sys
import os 

WAIT_TIME = 30  # Thời gian chờ tối đa


# Class Matrix
class Matrix:

    # Hàm khởi tạo|| parent, move depth: tham số tùy chọn
    def __init__(self, state, size, parent=None, move=None, depth=0):
        self.state = state
        self.size = size
        self.parent = parent
        self.move = move
        self.depth = depth
    
    # Hàm goal_test() để kiểm tra trạng thái đích
    def goal_test(self):
        goal_state = []
        n = self.size * self.size # Tổng số ô trong ma trận
        for i in range(self.size):
            row = []
            for j in range(self.size):
                row.append((i * self.size + j + 1) % n)
            goal_state.append(row)
        return self.state == goal_state
    
    # Hàm find_blank() để tìm vị trí của ô trống
    def find_blank(self):
        for i in range(len(self.state)):
            for j in range(len(self.state[i])):
                if self.state[i][j] == 0:
                    return i, j
                
    # Hàm neighbors() để trả về các trạng thái kề
    def neighbors(self):
        neighbors = []
        x, y = self.find_blank()
        directions = {
            'right': (x, y + 1),
            'down': (x + 1, y),
            'up': (x - 1, y),
            'left': (x, y - 1)}
        for move, (new_x, new_y) in directions.items():
            if 0 <= new_x < self.size and 0 <= new_y < self.size:
                new_state = copy.deepcopy(self.state)
                new_state[x][y], new_state[new_x][new_y] = new_state[new_x][new_y], new_state[x][y]
                # Thêm trạng thái mới vào danh sách các trạng thái kề:
                neighbors.append(Matrix(new_state, self.size, self, move, self.depth + 1))
        return neighbors

# Class Solver
class Solver:
    def __init__(self, initial_state, size):
        self.initial_state = initial_state
        self.size = size
        self.stack = [Matrix(initial_state, size)]  # Stack chứa các trạng thái
        self.visited = set()                        # Set chứa các trạng thái đã duyệt  
        self.solution_found = threading.Event()     # Cờ báo hiệu tìm thấy lời giải

    # Hàm set_solution() để lưu đường đi tới ma trận đích
    def set_solution(self, matrix):
        self.solution = []
        while matrix:
            self.solution.append(matrix)
            matrix = matrix.parent
        self.solution.reverse()
    
    # Hàm print_solution() để in ra đường đi tới ma trận đích
    def print_solution(self):
        if self.solution:
            print("\nCác bước giải bài toán:")
            step_count = 0
            for matrix in self.solution:
                for row in matrix.state:
                    print(' '.join(map(str, row)))
                print()
                step_count += 1
            print(f"Tổng các bước: {step_count - 1}")
        else:
            print("Không tìm thấy lời giải")
    
class DFS(Solver):
    def __init__(self, initial_state, size):
        super(DFS, self).__init__(initial_state, size)

    def solve(self):
        visited = set()                                 # Set chứa các trạng thái đã duyệt
        stack = [Matrix(self.initial_state, self.size)] # Stack chứa các trạng thái
        while stack: # Duyệt đến khi stack rỗng
            if self.solution_found.is_set():  # Kiểm tra trạng thái của biến cờ
                return
            matrix = stack.pop()
            if matrix.goal_test():
                self.set_solution(matrix)
                self.solution_found.set()                       # Đặt cờ để báo hiệu đã tìm thấy lời giải
                return
            if tuple(map(tuple, matrix.state)) not in visited:  # Kiểm tra trạng thái đã duyệt chưa
                 visited.add(tuple(map(tuple, matrix.state)))
                 for neighbor in reversed(matrix.neighbors()):
                    if tuple(map(tuple, neighbor.state)) not in visited:
                        stack.append(neighbor)
    
    def is_solvable(self):
        flat_list = [num for row in self.initial_state for num in row if num != 0]
        inversions = 0
        for i in range(len(flat_list)):
            for j in range(i + 1, len(flat_list)):
                if flat_list[i] > flat_list[j]:
                    inversions += 1
        if self.size % 2 == 1:
            return inversions % 2 == 0
        else:
            blank_row = self.size - next(i for i, row in enumerate(self.initial_state) if 0 in row)
            if blank_row % 2 == 0:
                return inversions % 2 == 1
            else:
                return inversions % 2 == 0

# Global functions
def generate_random_puzzle(size):
    puzzle = list(range(size * size))
    random.shuffle(puzzle)
    return [puzzle[i * size:(i + 1) * size] for i in range(size)]

# Main function
def main():
    while True:
        try:
            size = int(input("Nhập kích cỡ bài toán và 1 ma trận ngẫu nhiên sẽ được tạo: "))
            if size <= 0:
                raise ValueError("Phải là số nguyên dương")
            break
        except ValueError as e:
            print(f"Sai input {e}. Nhập lại.")
    
    initial_state = generate_random_puzzle(size)

    print("\nTrạng thái khởi đầu")
    for row in initial_state:
        print(' '.join(map(str, row)))

    solver = DFS(initial_state, size)
    if not solver.is_solvable():
        print("Bài toán không thể giải")
        return
    
    def run_solver():
        solver.solve()

    def timer():
        if not solver.solution_found.wait(WAIT_TIME):  # Chờ tối đa 60 giây hoặc cho đến khi tìm thấy lời giải
            print("Vượt quá thời gian hạn định, dừng giải thuật")
            os._exit(1)
    
    solver_thread = threading.Thread(target=run_solver)
    timer_thread = threading.Thread(target=timer)

    start_time = time.time()
    solver_thread.start()
    timer_thread.start()

    solver_thread.join()
    end_time = time.time()

    solver.print_solution()

    print("\nTrạng thái đích")
    goal_state = [[(i * size + j + 1) % (size * size) for j in range(size)] for i in range(size)]
    for row in goal_state:
        print(' '.join(map(str, row)))

    print(f"Tổng thời gian {end_time - start_time:.2f} giây")

if __name__ == "__main__":
    main()