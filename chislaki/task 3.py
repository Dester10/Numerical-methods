import numpy as np
import matplotlib.pyplot as plt

# Задаем коэффициенты системы уравнений
A = np.array([[-7, 3, 2], [2, -5, 2], [1, 1, -4]])
b = np.array([-2, 0, -3])

# Метод простых итераций
def simple_iteration(A, b, tol=1e-4, max_iter=1000):
    x = np.zeros_like(b)
    D = np.diag(np.diag(A))
    LU = A - D
    history = []
    for _ in range(max_iter):
        x_new = np.dot(np.linalg.inv(D), b - np.dot(LU, x))
        history.append(np.copy(x_new))  # Сохраняем историю значений
        if np.linalg.norm(x_new - x) < tol:
            return x_new, history
        x = x_new
    return x, history

# Метод Зейделя
def gauss_seidel(A, b, tol=1e-4, max_iter=1000):
    n = len(b)
    x = np.zeros(n)
    history = []
    for _ in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            x_new[i] = (b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
        history.append(np.copy(x_new))  # Сохраняем историю значений
        if np.linalg.norm(x_new - x) < tol:
            return x_new, history
        x = x_new
    return x, history

# Решение системы уравнений
x_simple_iteration, history_simple_iteration = simple_iteration(A, b)
x_gauss_seidel, history_gauss_seidel = gauss_seidel(A, b)

print("Метод простых итераций:")
print(x_simple_iteration)
print("\nМетод Зейделя:")
print(x_gauss_seidel)

# Визуализация процесса найденных значений
def plot_iterations(history, method_name):
    # Преобразуем список истории в массив для удобства работы
    history = np.array(history)
    iterations = np.arange(len(history))
    
    plt.figure(figsize=(10, 6))
    for i in range(history.shape[1]):
        plt.plot(iterations, history[:, i], label=f'Variable {i+1}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title(f'Convergence of {method_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Построение графиков
plot_iterations(history_simple_iteration, "Simple Iteration Method")
plot_iterations(history_gauss_seidel, "Gauss-Seidel Method")