class ArrayQueue:
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

    def __str__(self):
        return str(self.items)

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedListQueue:
    def __init__(self):
        self.front = None
        self.rear = None

    def enqueue(self, item):
        new_node = Node(item)
        if self.rear is None:
            self.front = self.rear = new_node
            return
        self.rear.next = new_node
        self.rear = new_node

    def dequeue(self):
        if self.is_empty():
            return None
        temp = self.front
        self.front = temp.next
        if self.front is None:
            self.rear = None
        return temp.data

    def is_empty(self):
        return self.front is None

    def size(self):
        current = self.front
        count = 0
        while current:
            count += 1
            current = current.next

    def __str__(self):
        items = []
        current = self.front
        while current:
            items.append(str(current.data))
            current = current.next
        return " -> ".join(items)

from collections import deque

class StandardLibraryQueue:
    def __init__(self):
        self.items = deque()

    def enqueue(self, item):
        self.items.appendleft(item)

    def dequeue(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

    def size(self):
        return len(self.items)

    def __str__(self):
        return str(list(self.items))

def calculate_trapped_water_case_a(matrix):
    if not matrix or len(matrix) < 3 or len(matrix[0]) < 3:
        return 0

    m, n = len(matrix), len(matrix[0])
    water = 0

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            min_surrounding = min(
                matrix[i - 1][j],
                matrix[i + 1][j],
                matrix[i][j - 1],
                matrix[i][j + 1]
            )
            if matrix[i][j] < min_surrounding:
                water += min_surrounding - matrix[i][j]

    return water

def calculate_trapped_water_case_b(matrix, i0, j0, V):
    if not matrix or i0 < 0 or j0 < 0 or i0 >= len(matrix) or j0 >= len(matrix[0]):
        return matrix, 0

    m, n = len(matrix), len(matrix[0])
    water_matrix = [[0 for _ in range(n)] for _ in range(m)]
    total_water = 0

    heap = []
    visited = [[False for _ in range(n)] for _ in range(m)]

    heap.append((matrix[i0][j0], i0, j0))
    visited[i0][j0] = True

    import heapq
    heapq.heapify(heap)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while heap and V > 0:
        current_height, i, j = heapq.heappop(heap)

        min_neighbor = float('inf')
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n:
                if not visited[ni][nj]:
                    heapq.heappush(heap, (matrix[ni][nj], ni, nj))
                    visited[ni][nj] = True
                if matrix[ni][nj] + water_matrix[ni][nj] > current_height:
                    min_neighbor = min(min_neighbor, matrix[ni][nj] + water_matrix[ni][nj])

        if min_neighbor == float('inf'):
            break

        add = min_neighbor - current_height
        if add > V:
            add = V

        water_matrix[i][j] += add
        V -= add
        total_water += add

        heapq.heappush(heap, (current_height + add, i, j))

    result = [[matrix[i][j] + water_matrix[i][j] for j in range(n)] for i in range(m)]
    return result, total_water

def print_matrix(matrix):
    for row in matrix:
        print(" ".join(map(str, row)))
    print()


def test_queue_implementations():
    print("\nТестирование реализаций очереди:")

    aq = ArrayQueue()
    aq.enqueue(1)
    aq.enqueue(2)
    aq.enqueue(3)
    print(f"ArrayQueue: {aq}, размер: {aq.size()}")
    print(f"Извлечено: {aq.dequeue()}, после извлечения: {aq}")

    llq = LinkedListQueue()
    llq.enqueue(1)
    llq.enqueue(2)
    llq.enqueue(3)
    print(f"LinkedListQueue: {llq}, размер: {llq.size()}")
    print(f"Извлечено: {llq.dequeue()}, после извлечения: {llq}")

    slq = StandardLibraryQueue()
    slq.enqueue(1)
    slq.enqueue(2)
    slq.enqueue(3)
    print(f"StandardLibraryQueue: {slq}, размер: {slq.size()}")
    print(f"Извлечено: {slq.dequeue()}, после извлечения: {slq}")


def test_water_volume():
    print("\nТестирование решения задачи о воде:")

    matrix_a = [
        [3, 3, 3, 3, 3],
        [3, 2, 1, 2, 3],
        [3, 2, 1, 2, 3],
        [3, 2, 2, 2, 3],
        [3, 3, 3, 3, 3]
    ]
    print("Матрица для случая a:")
    print_matrix(matrix_a)
    water_a = calculate_trapped_water_case_a(matrix_a)
    print(f"Объем воды после погружения: {water_a}")

    matrix_b = [
        [3, 3, 3, 3, 3],
        [3, 2, 1, 2, 3],
        [3, 2, 1, 2, 3],
        [3, 2, 2, 2, 3],
        [3, 3, 3, 3, 3]
    ]
    print("\nМатрица для случая b:")
    print_matrix(matrix_b)
    i0, j0 = 2, 2
    V = 5
    print(f"Заливаем {V} единиц воды в позицию ({i0}, {j0})")
    result, remaining = calculate_trapped_water_case_b(matrix_b, i0, j0, V)
    print("Результат:")
    print_matrix(result)
    print(f"Осталось воды: {V - remaining}")
    print(f"Вытекло воды: {remaining}")


def performance_comparison():
    print("\nСравнение производительности:")

    import timeit

    array_queue_time = timeit.timeit(
        'aq.enqueue(1); aq.dequeue()',
        setup='from __main__ import ArrayQueue; aq = ArrayQueue()',
        number=10000
    )
    print(f"ArrayQueue: {array_queue_time:.5f} сек на 10000 операций")

    linked_list_queue_time = timeit.timeit(
        'llq.enqueue(1); llq.dequeue()',
        setup='from __main__ import LinkedListQueue; llq = LinkedListQueue()',
        number=10000
    )
    print(f"LinkedListQueue: {linked_list_queue_time:.5f} сек на 10000 операций")

    standard_library_queue_time = timeit.timeit(
        'slq.enqueue(1); slq.dequeue()',
        setup='from __main__ import StandardLibraryQueue; slq = StandardLibraryQueue()',
        number=10000
    )
    print(f"StandardLibraryQueue: {standard_library_queue_time:.5f} сек на 10000 операций")

if __name__ == "__main__":
    print("Автор: Колесников Сергей Николаевич, группа 020303-АИСа-о24")
    test_queue_implementations()
    test_water_volume()
    performance_comparison()