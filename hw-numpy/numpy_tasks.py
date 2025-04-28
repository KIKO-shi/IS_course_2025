import numpy as np

def uniform_intervals(a, b, n):
    """1. создает numpy массив - равномерное разбиение интервала от a до b на n отрезков."""
    return np.linspace(a, b, n)
def test1():
    assert np.allclose(uniform_intervals(-1.2, 2.4, 7), np.array([-1.2, -0.6,  0. ,  0.6,  1.2,  1.8,  2.4]))
# Запуск теста
test1()
print("done.")

def cyclic123_array(n): 
    """2. Генерирует numpy массив длины  3𝑛 , заполненный циклически числами 1, 2, 3, 1, 2, 3, 1...."""
    return np.tile([1, 2, 3], n)

def test2():
    assert np.allclose(cyclic123_array(4), np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])) 
# Запуск теста
test2()
print("done.")

def first_n_odd_number(n):
    """3. Создает массив первых n нечетных целых чисел"""
    return np.arange(1, 2*n, 2)

def test3():
    assert np.allclose(first_n_odd_number(3), np.array([1, 3, 5]))
# Запуск теста
test3()
print("done.")

def zeros_array_with_border(n):
    """4. Создает массив нулей размера n x n с "рамкой" из единиц по краям."""
    arr = np.zeros((n, n))
    arr[[0, -1], :] = 1
    arr[:, [0, -1]] = 1
    return arr

def test4():
    assert np.allclose(zeros_array_with_border(4), np.array([[1., 1., 1., 1.],
                                                             [1., 0., 0., 1.],
                                                             [1., 0., 0., 1.],
                                                             [1., 1., 1., 1.]]))
# Запуск теста
test4()
print("done.")

def chess_board(n):
    """5. Создаёт массив n x n с шахматной доской из нулей и единиц"""
    return np.indices((n, n)).sum(axis=0) % 2

def test5():
    expected = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert np.allclose(chess_board(3), expected)
# Запуск теста
test5()
print("done.")

def matrix_with_sum_index(n):
    """6. Создаёт 𝑛 × 𝑛  матрицу с (𝑖,𝑗)-элементами равным 𝑖+𝑗."""
    return np.add.outer(np.arange(n), np.arange(n))

def test6():
    assert np.allclose(matrix_with_sum_index(3), np.array([[0, 1, 2],
                                                           [1, 2, 3],
                                                           [2, 3, 4]]))
# Запуск теста
test6()
print("done.")

def cos_sin_as_two_rows(a, b, dx):
    """7. Вычислите $cos(x)$ и $sin(x)$ на интервале [a, b) с шагом dx, 
    а затем объедините оба массива чисел как строки в один массив. """
    x = np.arange(a, b, dx)
    return np.cos(x), np.sin(x)

def test7():
    assert np.allclose(cos_sin_as_two_rows(0, 1, 0.25), np.array([[1.        , 0.96891242, 0.87758256, 0.73168887],
                                                                  [0.        , 0.24740396, 0.47942554, 0.68163876]]))
# Запуск теста
test7()
print("done.")

def compute_mean_rowssum_columnssum(A):
    """8. Для numpy массива A вычисляет среднее всех элементов, сумму строк и сумму столбцов."""
    return A.mean(), A.sum(axis=1), A.sum(axis=0)

def test8():
    np.random.seed(42)
    A = np.random.rand(3, 5)
    mean, rows_sum, columns_sum = compute_mean_rowssum_columnssum(A)

    assert np.abs(mean - 0.49456456164468965) < 1e-12
    assert np.allclose(rows_sum, np.sum(A, axis=1))
    assert np.allclose(columns_sum, np.sum(A, axis=0))
# Запуск теста
test8()
print("done.")


def sort_array_by_column(A, j):
    """ 9. Сортирует строки numpy массива A по j-му столбцу в порядке возрастания."""
    return A[A[:, j].argsort()]

def test9():
    np.random.seed(42)
    A = np.random.rand(5, 5)
    assert np.allclose(sort_array_by_column(A, 1), np.array([[0.15599452, 0.05808361, 0.86617615, 0.60111501, 0.70807258],
                                                             [0.61185289, 0.13949386, 0.29214465, 0.36636184, 0.45606998],
                                                             [0.18340451, 0.30424224, 0.52475643, 0.43194502, 0.29122914],
                                                             [0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864],
                                                             [0.02058449, 0.96990985, 0.83244264, 0.21233911, 0.18182497]]))
# Запуск теста
test9()
print("done.")

def compute_integral(a, b, f, dx, method):
    """10. Считает определённый интеграл функции f на отрезке [a, b] с шагом dx 3-мя методами:  
    method == 'rectangular' - методом прямоугольника   
    method == 'trapezoidal' - методом трапеций   
    method == 'simpson' - методом Симпсона  
    """
    x = np.arange(a, b, dx)
    y = f(x)
    rect = np.sum(y[:-1]) * dx
    trap = np.sum((y[:-1] + y[1:]) / 2 * dx)
    simpson = dx / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]))
    
    return rect, trap, simpson
def test10():
    f1 = lambda x: (x**2 + 3) / (x - 2)
    assert np.allclose(compute_integral(3, 4, f1, 0.001, method="rectangular"), 10.352030263919616, rtol=0.01)
    assert np.allclose(compute_integral(3, 4, f1, 0.001, method="trapezoidal"), 10.352030263919616, rtol=0.01)
    assert np.allclose(compute_integral(3, 4, f1, 0.001, method="simpson"), 10.352030263919616, rtol=0.001)

    f2 = lambda x: np.cos(x)**3
    assert np.allclose(compute_integral(0, np.pi/2, f2, 0.001, method="rectangular"), 2/3, rtol=0.01)
    assert np.allclose(compute_integral(0, np.pi/2, f2, 0.001, method="trapezoidal"), 2/3, rtol=0.01)
    assert np.allclose(compute_integral(0, np.pi/2, f2, 0.001, method="simpson"), 2/3, rtol=0.001)

#Запуск тестов
test10()
print("done.") 
